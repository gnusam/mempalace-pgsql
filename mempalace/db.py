"""
db.py — PostgreSQL + pgvector storage layer for MemPalace
=========================================================

Replaces ChromaDB + SQLite with a single PostgreSQL database.
Uses sentence-transformers for GPU-accelerated embeddings.
"""

import os
import hashlib
import json
import logging
import re
from datetime import datetime, date
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

logger = logging.getLogger("mempalace.db")

DEFAULT_DSN = "postgresql://mempalace:mempalace@localhost:5432/mempalace"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Lazy-loaded model
_model = None


_POWER_SUPPLY_PATH = Path("/sys/class/power_supply")


def _on_ac_power() -> bool:
    """True if plugged into AC mains, False if on battery.

    Returns True (assume plugged) when no Mains adapter is reported — desktop
    boxes, or sysfs unreadable from inside a container without /sys mounted.
    """
    try:
        for ps in _POWER_SUPPLY_PATH.iterdir():
            try:
                if (ps / "type").read_text().strip() == "Mains":
                    return (ps / "online").read_text().strip() == "1"
            except OSError:
                continue
    except OSError:
        pass
    return True


def _select_device(cuda_available: bool) -> str:
    """Pick the embedding device.

    MEMPALACE_DEVICE env var overrides auto-detect. Auto picks GPU when
    plugged in, CPU on battery — frees the dGPU to hit D3cold runtime
    suspend instead of holding ~260 MiB VRAM idle on a laptop.
    """
    override = os.environ.get("MEMPALACE_DEVICE", "").lower()
    if override == "cpu":
        return "cpu"
    if override == "cuda":
        return "cuda" if cuda_available else "cpu"
    if cuda_available and _on_ac_power():
        return "cuda"
    return "cpu"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = _select_device(torch.cuda.is_available())
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(f"Loaded {EMBEDDING_MODEL} on {device}")
    return _model


_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _sanitize_pg_str(s):
    """Make a string storable by PostgreSQL: drop NUL, replace lone surrogates.

    PostgreSQL cannot store NUL (0x00) in ``text`` or ``jsonb`` — psycopg
    rejects it outright ("PostgreSQL text fields cannot contain NUL (0x00)
    bytes"), and inside metadata it JSON-escapes to ``\\u0000``, which the
    jsonb cast rejects ("unsupported Unicode escape sequence"). A lone UTF-16
    surrogate (U+D800–U+DFFF) has no UTF-8 encoding, so psycopg raises
    UnicodeEncodeError before the query even reaches the server. Either way a
    single polluted transcript aborts the whole mine run and leaves every
    file after it unmined.

    Strip NUL, replace surrogates with U+FFFD (mirroring upstream's ChromaDB
    document handling): rejecting would re-abort the mine, dropping the
    drawer would lose recall. The encode probe keeps the common clean-string
    path allocation-free.
    """
    if "\x00" in s:
        s = s.replace("\x00", "")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = _LONE_SURROGATE_RE.sub("�", s)
    return s


def _sanitize_pg(value):
    """Recursively apply _sanitize_pg_str to strings in dicts/lists/tuples.

    Metadata is sanitized *before* json.dumps: with the default
    ``ensure_ascii=True`` both NUL and lone surrogates serialize to ``\\uXXXX``
    escapes that the server-side jsonb cast rejects, so cleaning the
    serialized string would be too late. Non-string scalars pass through.
    """
    if isinstance(value, str):
        return _sanitize_pg_str(value)
    if isinstance(value, dict):
        return {_sanitize_pg(k): _sanitize_pg(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_pg(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_pg(v) for v in value)
    return value


def embed(texts):
    """Embed a list of texts. Returns list of numpy arrays."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return [np.array(e, dtype=np.float32) for e in embeddings]


# Half-open validity interval for as-of KG queries: a fact whose valid_to
# equals the query instant has already ended at that instant, so the interval
# is [valid_from, valid_to) and the upper bound is strict (>). This lets a
# fact and its successor share a boundary instant without an as-of query
# returning both. Date-only valid_to (length 10, TEXT lexicographic compare)
# still expands to the end of that day, so a standalone date-only fact stays
# valid through its whole final day exactly as before.
_VALID_TO_EXPR = (
    "(CASE WHEN LENGTH(t.valid_to) = 10 THEN t.valid_to || 'T23:59:59' ELSE t.valid_to END)"
)


class PalaceDB:
    """Single database for drawers (vectors) + knowledge graph."""

    def __init__(self, dsn=None):
        if dsn:
            self.dsn = dsn
        else:
            env = os.environ.get("DATABASE_URL")
            if env:
                self.dsn = env
            else:
                from .config import MempalaceConfig

                self.dsn = MempalaceConfig().database_url
        self._conn = None

    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            register_vector(self._conn)
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def reset(self):
        """Close the current connection so the next conn() call reconnects.

        Used by callers after a failure that may have left the connection in an
        aborted-transaction state (e.g. statement_timeout during a forced seq
        scan). The next operation gets a fresh, healthy connection.
        """
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
        finally:
            self._conn = None

    def init_schema(self):
        """Create tables if they don't exist."""
        schema_path = Path(__file__).parent / "init_schema.sql"
        with open(schema_path) as f:
            self.conn().cursor().execute(f.read())

    # ── Drawers ──────────────────────────────────────────────────────────

    @staticmethod
    def _mining_drawer_id(wing, room, source_file, chunk_index):
        """Deterministic per file+chunk slot ID (mining path)."""
        digest = hashlib.md5((source_file + str(chunk_index)).encode()).hexdigest()[:16]
        return f"drawer_{wing}_{room}_{digest}"

    @staticmethod
    def _registry_sentinel_id(source_file):
        """Deterministic ID of the 0-chunk registry sentinel for a file."""
        return f"_reg_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"

    @staticmethod
    def _scope_clause(ingest_mode, extract_mode, include_registry=False):
        """SQL fragment + params selecting one mining scope's drawers.

        A "scope" is the set of drawers one mining pass owns for a
        source_file: project mining writes no ingest_mode key, convo mining
        writes ingest_mode='convos' plus its extract_mode (legacy convo rows
        without extract_mode count as 'exchange'). Scoping both the staleness
        check and the stale-row purge to the active pass is what lets
        exchange-mode and general-mode drawers coexist for the same
        transcript without deleting or invalidating each other — the
        over-match class upstream hit in PR #2089, where a legacy
        no-extract_mode rule purged the sweeper's own drawers.

        With ``include_registry`` (the freshness check) the file's 0-chunk
        registry sentinel counts toward any scope — "this file yielded
        nothing at mtime m" is scope-independent information. The purge in
        replace_file_drawers leaves it out: sentinel lifecycle belongs to
        register_empty_file and the explicit by-id delete when chunks land.
        """
        if ingest_mode is None:
            clause = "((metadata->>'ingest_mode') IS NULL)"
            params = []
        else:
            clause = (
                "(metadata->>'ingest_mode' = %s"
                " AND COALESCE(metadata->>'extract_mode', 'exchange') = %s)"
            )
            params = [ingest_mode, extract_mode or "exchange"]
        if include_registry:
            clause = f"({clause} OR metadata->>'ingest_mode' = 'registry')"
        return clause, params

    def replace_file_drawers(
        self,
        wing,
        chunks,
        source_file,
        agent="mempalace",
        source_mtime=None,
        ingest_mode=None,
        extract_mode=None,
    ):
        """Atomically replace one scope's drawers for a file with a new chunk set.

        ``chunks`` is a list of dicts with ``room``, ``content``,
        ``chunk_index`` and optional ``metadata``. The stale-row purge and
        every insert commit as ONE transaction, so a mine killed mid-file
        leaves the previous state fully intact instead of a partial batch
        that file_already_mined() would mistake for a complete mine, and a
        failed purge aborts the whole attempt instead of letting old and new
        rows coexist (adapted from upstream PR #2088 — upstream needs a
        chunk_total completion marker because ChromaDB commits batch by
        batch; PostgreSQL lets us make the whole file atomic instead).

        Rows outside the (ingest_mode, extract_mode) scope survive: see
        _scope_clause. An empty ``chunks`` list purges the scope's rows —
        a file whose current content yields nothing must not keep serving
        its old drawers. The file's registry sentinel is dropped whenever
        real chunks land, so a formerly-empty file doesn't keep a stale
        sentinel that fails the freshness check forever.

        ``source_mtime`` must be the mtime paired with the content the
        caller actually read and chunked — never a later re-stat (upstream
        PR #2088's TOCTOU: an append landing between read and re-stat gets
        stamped as already-mined and is silently skipped forever).
        """
        source_file = _sanitize_pg_str(source_file)
        contents = [_sanitize_pg_str(c["content"]) for c in chunks]
        embeddings = embed(contents) if contents else []
        now = datetime.now()
        rows = []
        for chunk, content, emb in zip(chunks, contents, embeddings):
            meta_dict = dict(chunk.get("metadata") or {})
            if ingest_mode:
                meta_dict.setdefault("ingest_mode", ingest_mode)
                if extract_mode:
                    meta_dict.setdefault("extract_mode", extract_mode)
            if source_mtime is not None:
                meta_dict["source_mtime"] = source_mtime
            rows.append(
                (
                    self._mining_drawer_id(wing, chunk["room"], source_file, chunk["chunk_index"]),
                    chunk["room"],
                    content,
                    emb,
                    chunk["chunk_index"],
                    json.dumps(_sanitize_pg(meta_dict)),
                )
            )

        scope_sql, scope_params = self._scope_clause(ingest_mode, extract_mode)
        keep_ids = [r[0] for r in rows]
        conn = self.conn()
        old_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            cur = conn.cursor()
            delete_sql = f"DELETE FROM drawers WHERE source_file = %s AND {scope_sql}"
            delete_params = [source_file, *scope_params]
            if keep_ids:
                delete_sql += " AND NOT (id = ANY(%s))"
                delete_params.append(keep_ids)
            cur.execute(delete_sql, delete_params)
            if keep_ids:
                cur.execute(
                    "DELETE FROM drawers WHERE id = %s",
                    (self._registry_sentinel_id(source_file),),
                )
            for drawer_id, room, content, emb, chunk_index, meta in rows:
                cur.execute(
                    """INSERT INTO drawers (id, wing, room, content, embedding, source_file,
                       chunk_index, added_by, filed_at, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET
                         wing        = EXCLUDED.wing,
                         room        = EXCLUDED.room,
                         content     = EXCLUDED.content,
                         embedding   = EXCLUDED.embedding,
                         source_file = EXCLUDED.source_file,
                         chunk_index = EXCLUDED.chunk_index,
                         added_by    = EXCLUDED.added_by,
                         filed_at    = EXCLUDED.filed_at,
                         metadata    = EXCLUDED.metadata""",
                    (
                        drawer_id,
                        wing,
                        room,
                        content,
                        emb,
                        source_file,
                        chunk_index,
                        agent,
                        now,
                        meta,
                    ),
                )
            conn.commit()
            return keep_ids
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit = old_autocommit

    def add_drawer(
        self, wing, room, content, source_file="", chunk_index=0, agent="mempalace", metadata=None
    ):
        # Mining path: ID = hash(source_file + chunk_index) — deterministic
        # per file+chunk slot so re-mining a modified file targets the same
        # row and can update it via the ON CONFLICT DO UPDATE clause below.
        #
        # MCP / diary path: ID = hash(content) — deterministic per content,
        # so re-filing identical content is idempotent (same ID → upsert
        # becomes a no-op UPDATE with the same values). This replaces the
        # old content[:200] + datetime.now() hash which produced a fresh ID
        # every call and let the same content pile up as duplicates; it is
        # also the fix for upstream findings #6 (TOCTOU) and #13 (non-
        # deterministic IDs).
        # Sanitize before hashing so the drawer ID stays consistent with the
        # stored (cleaned) content across re-mines of the same file.
        content = _sanitize_pg_str(content)
        source_file = _sanitize_pg_str(source_file) if source_file else source_file
        if source_file:
            drawer_id = self._mining_drawer_id(wing, room, source_file, chunk_index)
        else:
            digest = hashlib.md5(content.encode()).hexdigest()[:16]
            drawer_id = f"drawer_{wing}_{room}_{digest}"
        emb = embed([content])[0]
        # Stamp source_mtime on mining drawers so file_already_mined() can
        # detect when a file has been edited since it was last mined. We
        # merge into any caller-supplied metadata rather than replacing it.
        meta_dict = dict(metadata or {})
        if source_file and "source_mtime" not in meta_dict:
            try:
                meta_dict["source_mtime"] = os.path.getmtime(source_file)
            except OSError:
                pass
        meta = json.dumps(_sanitize_pg(meta_dict))
        cur = self.conn().cursor()
        try:
            # Upsert so that re-mining a modified file actually updates the
            # row instead of being silently dropped (the old ON CONFLICT DO
            # NOTHING was the data-stagnation bug from upstream finding #11).
            # For content-based MCP IDs the upsert is a no-op when content
            # is unchanged and idempotent when it is.
            cur.execute(
                """INSERT INTO drawers (id, wing, room, content, embedding, source_file,
                   chunk_index, added_by, filed_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                     wing        = EXCLUDED.wing,
                     room        = EXCLUDED.room,
                     content     = EXCLUDED.content,
                     embedding   = EXCLUDED.embedding,
                     source_file = EXCLUDED.source_file,
                     chunk_index = EXCLUDED.chunk_index,
                     added_by    = EXCLUDED.added_by,
                     filed_at    = EXCLUDED.filed_at,
                     metadata    = EXCLUDED.metadata""",
                (
                    drawer_id,
                    wing,
                    room,
                    content,
                    emb,
                    source_file,
                    chunk_index,
                    agent,
                    datetime.now(),
                    meta,
                ),
            )
            return drawer_id
        except Exception:
            self.conn().rollback()
            raise

    def register_empty_file(
        self,
        source_file,
        wing,
        agent="mempalace",
        source_mtime=None,
        purge_stale=False,
        ingest_mode="convos",
        extract_mode=None,
    ):
        """Insert a no-embedding sentinel so file_already_mined() returns True
        for files that produce zero chunks (port of upstream 87e8baf, PR #732).

        Without this, files that normalize to nothing or produce zero chunks
        are re-read on every mine run — file_already_mined() requires a row
        with a matching source_file AND a stored source_mtime, but 0-chunk
        early exits never write one.

        ``source_mtime`` should be the mtime the caller captured with the
        read (same TOCTOU rationale as replace_file_drawers); falls back to
        a getmtime() probe for legacy callers.

        With ``purge_stale`` the caller asserts the file's current content
        genuinely yields nothing, so the scope's old drawers no longer have
        a source: they are deleted in the SAME transaction as the sentinel
        upsert (upstream PR #2089 — a rebuild must not leave rows its own
        pass no longer owns, and a failed purge must roll back the sentinel
        too, or the file would read as freshly mined while stale drawers
        keep serving). Callers on possibly-transient failure paths (e.g. a
        normalize() parse error) must leave ``purge_stale`` off: a hiccup
        must not destroy mined data.

        The embedding column is left NULL so the sentinel doesn't cost an
        embedding pass and is invisible to vector search.
        """
        if source_mtime is None:
            try:
                source_mtime = os.path.getmtime(source_file)
            except OSError:
                return None
        sentinel_id = self._registry_sentinel_id(source_file)
        meta = json.dumps({"source_mtime": source_mtime, "ingest_mode": "registry"})
        conn = self.conn()
        old_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            cur = conn.cursor()
            if purge_stale:
                scope_sql, scope_params = self._scope_clause(ingest_mode, extract_mode)
                cur.execute(
                    f"DELETE FROM drawers WHERE source_file = %s AND {scope_sql}",
                    (source_file, *scope_params),
                )
            cur.execute(
                """INSERT INTO drawers (id, wing, room, content, embedding,
                       source_file, chunk_index, added_by, filed_at, metadata)
                   VALUES (%s, %s, %s, %s, NULL, %s, 0, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                     metadata    = EXCLUDED.metadata,
                     filed_at    = EXCLUDED.filed_at""",
                (
                    sentinel_id,
                    wing,
                    "_registry",
                    f"[registry] {source_file}",
                    source_file,
                    agent,
                    datetime.now(),
                    meta,
                ),
            )
            conn.commit()
            return sentinel_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit = old_autocommit

    def content_hash_exists(self, file_content_hash, exclude_source_file):
        """True if another file's drawers already carry this content hash.

        Adapted from upstream PR #2050 (prefetch_content_hashes): the same
        conversation re-exported under a new filename must not duplicate
        every drawer under fresh slot IDs. The caller hashes the whole
        normalized transcript; drawers store it as
        ``metadata->>'file_content_hash'`` (indexed by expression, see
        init_schema.sql). The file's own rows are excluded so a touch'd but
        unchanged file still re-mines under its own name.

        Pre-hash legacy drawers carry no hash and never match — dedup
        applies to newly mined files only.
        """
        cur = self.conn().cursor()
        cur.execute(
            "SELECT 1 FROM drawers WHERE metadata->>'file_content_hash' = %s "
            "AND source_file <> %s LIMIT 1",
            (file_content_hash, exclude_source_file),
        )
        return cur.fetchone() is not None

    def find_duplicate_pairs(self, where=None, similarity_threshold=0.9, max_drawers=2000):
        """Return (id_a, id_b, similarity) pairs of near-duplicate drawers.

        Adapted from upstream PR #1951 (read-only find_duplicates tool):
        upstream probes each drawer's nearest neighbors through the backend
        API; PostgreSQL lets us do one bounded self-join with the pgvector
        cosine operator instead. The scope is capped at ``max_drawers``
        most-recent rows (a full self-join over a 275k palace would be
        quadratic), so on large scopes this is a rolling-window audit, not
        an exhaustive one — callers see the cap in the tool response.
        Registry sentinels (embedding IS NULL) are naturally excluded.
        """
        clauses, params = self._build_where(where)
        scope_sql = "SELECT id, embedding FROM drawers WHERE embedding IS NOT NULL"
        if clauses:
            scope_sql += f" AND {clauses}"
        scope_sql += " ORDER BY filed_at DESC LIMIT %s"
        cur = self.conn().cursor()
        cur.execute(
            f"""WITH scope AS ({scope_sql})
                SELECT a.id, b.id, 1 - (a.embedding <=> b.embedding) AS sim
                FROM scope a JOIN scope b ON a.id < b.id
                WHERE 1 - (a.embedding <=> b.embedding) >= %s
                ORDER BY sim DESC""",
            (*params, int(max_drawers), float(similarity_threshold)),
        )
        return [(a, b, float(s)) for a, b, s in cur.fetchall()]

    def files_already_mined(self, file_mtimes, ingest_mode=None, extract_mode=None):
        """Bulk variant of file_already_mined: one query for a whole scan.

        ``file_mtimes`` is a list of ``(source_file, current_mtime)`` pairs
        (the caller stats each file once at scan time). Returns the set of
        source_files whose scoped drawers ALL carry that mtime — the same
        strict rule as file_already_mined, evaluated server-side with a
        VALUES join instead of one round-trip per file (the deferred
        "bulk already-mined pre-fetch" candidate from the 2026-07-29
        upstream audit; on PG the per-file probes dominate scan time for
        large already-mined projects).

        A file whose mtime changes after this snapshot is simply mined (or
        skipped) based on the snapshot — the same race window the per-file
        check always had; the next run converges.
        """
        if not file_mtimes:
            return set()
        scope_sql, scope_params = self._scope_clause(
            ingest_mode, extract_mode, include_registry=True
        )
        cur = self.conn().cursor()
        if scope_params:
            scope_sql = cur.mogrify(scope_sql, scope_params).decode()
        rows = psycopg2.extras.execute_values(
            cur,
            f"""SELECT d.source_file
                FROM drawers d
                JOIN (VALUES %s) AS v(source_file, mtime)
                  ON d.source_file = v.source_file
                WHERE {scope_sql}
                GROUP BY d.source_file
                HAVING bool_and(
                    COALESCE((d.metadata->>'source_mtime')::float8 = v.mtime::float8, FALSE)
                )""",
            [(str(f), float(m)) for f, m in file_mtimes],
            page_size=500,
            fetch=True,
        )
        return {r[0] for r in rows}

    def file_already_mined(self, source_file, ingest_mode=None, extract_mode=None):
        """Fast check: has this file been filed before AND is unchanged?

        Strict variant (adapted from upstream PR #2088): EVERY drawer in the
        file's mining scope must carry the current on-disk mtime, not just an
        arbitrary one (the old LIMIT 1 probe). A partial state — e.g. rows
        left by a pre-atomic-replace mine that died mid-file, or stale tail
        rows from a shrunk file — therefore reads as "not mined" and gets
        cleaned up by the next replace_file_drawers() pass, instead of being
        mistaken for a complete mine and skipped forever.

        The check is scoped by (ingest_mode, extract_mode) — see
        _scope_clause — so convo extract modes don't invalidate each other's
        freshness. The convo scope also counts the registry sentinel: a
        0-chunk file is "mined" for any mode as long as the sentinel's mtime
        is current.

        Returns False (re-mine needed) when:
          - no drawer exists in this file's scope,
          - any scoped drawer lacks source_mtime (pre-mtime build), or
          - any scoped drawer's stored mtime differs from the current one.

        This keeps the bf88daa port's contract: upsert semantics let a
        modified file update its rows, and this check stops the miner from
        short-circuiting modified files as "already mined" at the scan stage.
        """
        try:
            current_mtime = os.path.getmtime(source_file)
        except OSError:
            return False
        scope_sql, scope_params = self._scope_clause(
            ingest_mode, extract_mode, include_registry=True
        )
        cur = self.conn().cursor()
        cur.execute(
            "SELECT bool_and(COALESCE((metadata->>'source_mtime')::float8 = %s, FALSE)) "
            f"FROM drawers WHERE source_file = %s AND {scope_sql}",
            (current_mtime, source_file, *scope_params),
        )
        row = cur.fetchone()
        return bool(row and row[0])

    def get_drawers(self, where=None, limit=None, offset=0, include=None):
        """Get drawers with optional filters. Returns ChromaDB-compatible dict."""
        clauses, params = self._build_where(where)
        sql = "SELECT id, wing, room, content, source_file, chunk_index, added_by, filed_at, metadata FROM drawers"
        if clauses:
            sql += f" WHERE {clauses}"
        sql += " ORDER BY filed_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        if offset:
            sql += f" OFFSET {int(offset)}"

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()

        ids = [r["id"] for r in rows]
        documents = [r["content"] for r in rows]
        metadatas = []
        for r in rows:
            m = {
                "wing": r["wing"],
                "room": r["room"],
                "source_file": r["source_file"] or "",
                "chunk_index": r["chunk_index"],
                "added_by": r["added_by"] or "",
                "filed_at": r["filed_at"].isoformat() if r["filed_at"] else "",
            }
            extra = r["metadata"]
            if extra and isinstance(extra, dict):
                m.update(extra)
            metadatas.append(m)

        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def query(self, query_text, n_results=5, where=None, auto_detect=True, since=None, before=None):
        """Semantic search with optional automatic wing/room name matching.

        If ``auto_detect`` is true and no explicit filter is given, we inspect
        the query text for a wing or room name and auto-scope to it. Callers
        that want to search the whole palace (e.g. duplicate detection, where
        a room name inside the content should NOT constrain the search) must
        pass ``auto_detect=False``.

        ``since`` / ``before`` (adapted from upstream PR #2000) bound the
        drawer's ``filed_at`` timestamp: ``since`` is inclusive from the
        start of that day, ``before`` is exclusive of it (``YYYY-MM-DD``
        strings — PostgreSQL casts them to that day's midnight). Like any
        filter, a date window routes the query through the exact
        sequential-scan path rather than HNSW.
        """
        # Auto-detect wing/room name in query when no filter specified
        if where is None and auto_detect:
            where = self._auto_detect_filter(query_text)

        emb = embed([query_text])[0]
        clauses, params = self._build_where(where)
        if since:
            clauses = f"{clauses} AND filed_at >= %s" if clauses else "filed_at >= %s"
            params.append(since)
        if before:
            clauses = f"{clauses} AND filed_at < %s" if clauses else "filed_at < %s"
            params.append(before)

        conn = self.conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if clauses:
            # HNSW doesn't support pre-filtering. Wrap in a transaction
            # with index scans disabled to force sequential scan.
            #
            # We must restore autocommit (and rollback on failure) even when
            # the query raises, or the connection is left in an aborted-
            # transaction state and every subsequent call fails with
            # "current transaction is aborted, commands ignored until end of
            # transaction block". This used to corrupt the long-lived MCP
            # server connection after the first slow query hit a timeout.
            old_autocommit = conn.autocommit
            conn.autocommit = False
            try:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
                sql = f"""SELECT id, wing, room, content, source_file, chunk_index,
                                 added_by, filed_at, metadata,
                                 embedding <=> %s AS distance
                          FROM drawers WHERE {clauses}
                          ORDER BY distance LIMIT {int(n_results)}"""
                cur.execute(sql, [emb] + params)
                rows = cur.fetchall()
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.autocommit = old_autocommit
        else:
            # Unfiltered HNSW path. The pgvector default `hnsw.ef_search=40`
            # is too low for a 400k+ drawer corpus — greedy nearest-neighbor
            # search gets stuck in a local cluster of noise and misses the
            # true top-k by a wide margin (measured recall@10 ≈ 0% on
            # short queries). Wrap in a transaction and bump ef_search
            # locally so every search gets the wider exploration without
            # affecting other sessions.
            old_autocommit = conn.autocommit
            conn.autocommit = False
            try:
                cur.execute("SET LOCAL hnsw.ef_search = 500")
                sql = f"""SELECT id, wing, room, content, source_file, chunk_index,
                                added_by, filed_at, metadata,
                                embedding <=> %s AS distance
                         FROM drawers
                         ORDER BY distance LIMIT {int(n_results)}"""
                cur.execute(sql, [emb])
                rows = cur.fetchall()
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.autocommit = old_autocommit

        ids, documents, metadatas, distances = [], [], [], []
        for r in rows:
            ids.append(r["id"])
            documents.append(r["content"])
            distances.append(float(r["distance"]))
            m = {
                "wing": r["wing"],
                "room": r["room"],
                "source_file": r["source_file"] or "",
                "chunk_index": r["chunk_index"],
                "added_by": r["added_by"] or "",
                "filed_at": r["filed_at"].isoformat() if r["filed_at"] else "",
            }
            extra = r["metadata"]
            if extra and isinstance(extra, dict):
                m.update(extra)
            metadatas.append(m)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def delete_drawer(self, drawer_id):
        cur = self.conn().cursor()
        cur.execute("DELETE FROM drawers WHERE id = %s", (drawer_id,))
        return cur.rowcount > 0

    def drawer_exists(self, drawer_id):
        cur = self.conn().cursor()
        cur.execute("SELECT 1 FROM drawers WHERE id = %s", (drawer_id,))
        return cur.fetchone() is not None

    def count(self, where=None):
        clauses, params = self._build_where(where)
        sql = "SELECT COUNT(*) FROM drawers"
        if clauses:
            sql += f" WHERE {clauses}"
        cur = self.conn().cursor()
        cur.execute(sql, params)
        return cur.fetchone()[0]

    # ── Compressed (AAAK) ────────────────────────────────────────────────

    def upsert_compressed(self, drawer_id, content, metadata=None):
        content = _sanitize_pg_str(content)
        emb = embed([content])[0]
        meta = json.dumps(_sanitize_pg(metadata or {}))
        cur = self.conn().cursor()
        cur.execute(
            """INSERT INTO compressed (id, content, embedding, metadata)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,
               embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata""",
            (drawer_id, content, emb, meta),
        )

    # ── Knowledge Graph ──────────────────────────────────────────────────

    @staticmethod
    def _entity_id(name):
        return name.lower().replace(" ", "_").replace("'", "")

    def add_entity(self, name, entity_type="unknown", properties=None):
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        cur = self.conn().cursor()
        cur.execute(
            """INSERT INTO entities (id, name, type, properties)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
               type = EXCLUDED.type, properties = EXCLUDED.properties""",
            (eid, name, entity_type, props),
        )
        return eid

    def add_triple(
        self,
        subject,
        predicate,
        obj,
        valid_from=None,
        valid_to=None,
        confidence=1.0,
        source_closet=None,
        source_file=None,
    ):
        # Reject inverted intervals: a triple with valid_to < valid_from
        # would never satisfy `valid_from <= as_of AND valid_to > as_of`,
        # so it would be invisible to every as-of query — silently corrupt.
        # Same-day intervals (point-in-time facts) are explicitly allowed.
        if valid_from is not None and valid_to is not None and valid_to < valid_from:
            raise ValueError(
                f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                "an inverted interval would be invisible to every KG query"
            )

        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred_norm = predicate.lower().replace(" ", "_")

        # Auto-create entities
        self.add_entity(subject)
        self.add_entity(obj)

        # Check for existing active triple
        cur = self.conn().cursor()
        cur.execute(
            "SELECT id FROM triples WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
            (sub_id, pred_norm, obj_id),
        )
        existing = cur.fetchone()
        if existing:
            return existing[0]

        triple_id = hashlib.md5(
            f"{sub_id}:{pred_norm}:{obj_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        cur.execute(
            """INSERT INTO triples (id, subject, predicate, object, valid_from,
               valid_to, confidence, source_closet, source_file)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                triple_id,
                sub_id,
                pred_norm,
                obj_id,
                valid_from,
                valid_to,
                confidence,
                source_closet,
                source_file,
            ),
        )
        return triple_id

    def invalidate(self, subject, predicate, obj, ended=None):
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred_norm = predicate.lower().replace(" ", "_")
        ended = ended or date.today().isoformat()
        cur = self.conn().cursor()
        cur.execute(
            "UPDATE triples SET valid_to=%s WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
            (ended, sub_id, pred_norm, obj_id),
        )

    def query_entity(self, name, as_of=None, direction="both"):
        eid = self._entity_id(name)
        results = []

        if direction in ("outgoing", "both"):
            results += self._query_triples("t.subject = %s", eid, as_of, "outgoing")
        if direction in ("incoming", "both"):
            results += self._query_triples("t.object = %s", eid, as_of, "incoming")
        return results

    def _query_triples(self, filter_clause, entity_id, as_of, direction):
        sql = f"""SELECT t.*, s.name as sub_name, o.name as obj_name
                  FROM triples t
                  JOIN entities s ON t.subject = s.id
                  JOIN entities o ON t.object = o.id
                  WHERE {filter_clause}"""
        params = [entity_id]

        if as_of:
            sql += " AND (t.valid_from IS NULL OR t.valid_from <= %s)"
            sql += f" AND (t.valid_to IS NULL OR {_VALID_TO_EXPR} > %s)"
            params += [as_of, as_of]

        sql += " ORDER BY t.valid_from ASC NULLS LAST"

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "direction": direction,
                    "subject": r["sub_name"],
                    "predicate": r["predicate"],
                    "object": r["obj_name"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "confidence": r["confidence"],
                    "source_closet": r["source_closet"],
                    "current": r["valid_to"] is None,
                }
            )
        return results

    def query_relationship(self, predicate, as_of=None):
        pred_norm = predicate.lower().replace(" ", "_")
        sql = """SELECT t.*, s.name as sub_name, o.name as obj_name
                 FROM triples t
                 JOIN entities s ON t.subject = s.id
                 JOIN entities o ON t.object = o.id
                 WHERE t.predicate = %s"""
        params = [pred_norm]
        if as_of:
            sql += " AND (t.valid_from IS NULL OR t.valid_from <= %s)"
            sql += f" AND (t.valid_to IS NULL OR {_VALID_TO_EXPR} > %s)"
            params += [as_of, as_of]

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "confidence": r["confidence"],
                "current": r["valid_to"] is None,
            }
            for r in cur.fetchall()
        ]

    def timeline(self, entity_name=None):
        if entity_name:
            eid = self._entity_id(entity_name)
            sql = """SELECT t.*, s.name as sub_name, o.name as obj_name
                     FROM triples t
                     JOIN entities s ON t.subject = s.id
                     JOIN entities o ON t.object = o.id
                     WHERE t.subject = %s OR t.object = %s
                     ORDER BY t.valid_from ASC NULLS LAST"""
            params = [eid, eid]
        else:
            sql = """SELECT t.*, s.name as sub_name, o.name as obj_name
                     FROM triples t
                     JOIN entities s ON t.subject = s.id
                     JOIN entities o ON t.object = o.id
                     ORDER BY t.valid_from ASC NULLS LAST
                     LIMIT 100"""
            params = []

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in cur.fetchall()
        ]

    def kg_stats(self):
        cur = self.conn().cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        entities = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM triples")
        triples = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM triples WHERE valid_to IS NULL")
        current = cur.fetchone()[0]
        cur.execute("SELECT DISTINCT predicate FROM triples")
        types = [r[0] for r in cur.fetchall()]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": triples - current,
            "relationship_types": types,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts):
        """Seed the knowledge graph from structured entity facts.

        Bootstraps the graph with known ground truth (people, relationships,
        interests). Ported from knowledge_graph.py.
        """
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(
                name,
                etype,
                {"gender": facts.get("gender", ""), "birthday": facts.get("birthday", "")},
            )

            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name, "child_of", parent.capitalize(), valid_from=facts.get("birthday")
                )

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of", facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of", facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of", facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(), valid_from="2025-01-01")

    def _auto_detect_filter(self, query_text):
        """Check if query contains a wing or room name and return a filter.

        NULL/empty wing/room rows are skipped — ``None.lower()`` used to raise
        AttributeError and propagate up as an opaque KO.
        """
        query_lower = query_text.lower().strip()
        normalized = query_lower.replace(" ", "_").replace("-", "_")
        query_words = set(normalized.split("_"))
        cur = self.conn().cursor()

        # Check exact wing match
        cur.execute("SELECT DISTINCT wing FROM drawers WHERE wing IS NOT NULL AND wing <> ''")
        wings = [r[0] for r in cur.fetchall()]
        for w in wings:
            w_lower = w.lower()
            if w_lower == normalized:
                return {"wing": w}
            # Also match if wing name is a significant word in the query
            if w_lower in query_words and len(w) > 2:
                return {"wing": w}

        # Check exact room match
        cur.execute("SELECT DISTINCT room FROM drawers WHERE room IS NOT NULL AND room <> ''")
        rooms = [r[0] for r in cur.fetchall()]
        for r in rooms:
            if r.lower() == normalized:
                return {"room": r}

        return None

    def _build_where(self, where):
        """Convert ChromaDB-style where dict to SQL WHERE clause."""
        if not where:
            return "", []

        if "$and" in where:
            parts, params = [], []
            for cond in where["$and"]:
                c, p = self._build_where(cond)
                parts.append(c)
                params.extend(p)
            return " AND ".join(f"({p})" for p in parts), params

        clauses, params = [], []
        for key, val in where.items():
            if key.startswith("$"):
                continue
            clauses.append(f"{key} = %s")
            params.append(val)

        return " AND ".join(clauses), params


# ── Module-level singleton ───────────────────────────────────────────────

_db = None


def get_db(dsn=None):
    global _db
    if _db is None or (_db._conn and _db._conn.closed):
        _db = PalaceDB(dsn)
    return _db
