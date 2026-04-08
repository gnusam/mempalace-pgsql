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


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(f"Loaded {EMBEDDING_MODEL} on {device}")
    return _model


def embed(texts):
    """Embed a list of texts. Returns list of numpy arrays."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return [np.array(e, dtype=np.float32) for e in embeddings]


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

    def init_schema(self):
        """Create tables if they don't exist."""
        schema_path = Path(__file__).parent / "init_schema.sql"
        with open(schema_path) as f:
            self.conn().cursor().execute(f.read())

    # ── Drawers ──────────────────────────────────────────────────────────

    def add_drawer(self, wing, room, content, source_file="", chunk_index=0,
                   agent="mempalace", metadata=None):
        # Use content hash when source_file is empty (MCP/diary entries)
        if source_file:
            hash_input = source_file + str(chunk_index)
        else:
            hash_input = content[:200] + datetime.now().isoformat()
        drawer_id = f"drawer_{wing}_{room}_{hashlib.md5(hash_input.encode()).hexdigest()[:16]}"
        emb = embed([content])[0]
        meta = json.dumps(metadata or {})
        cur = self.conn().cursor()
        try:
            cur.execute(
                """INSERT INTO drawers (id, wing, room, content, embedding, source_file,
                   chunk_index, added_by, filed_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""",
                (drawer_id, wing, room, content, emb, source_file,
                 chunk_index, agent, datetime.now(), meta),
            )
            return drawer_id if cur.rowcount > 0 else None
        except Exception:
            self.conn().rollback()
            raise

    def file_already_mined(self, source_file):
        cur = self.conn().cursor()
        cur.execute("SELECT 1 FROM drawers WHERE source_file = %s LIMIT 1", (source_file,))
        return cur.fetchone() is not None

    def get_drawers(self, where=None, limit=None, offset=0, include=None):
        """Get drawers with optional filters. Returns ChromaDB-compatible dict."""
        clauses, params = self._build_where(where)
        sql = f"SELECT id, wing, room, content, source_file, chunk_index, added_by, filed_at, metadata FROM drawers"
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
                "wing": r["wing"], "room": r["room"],
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

    def query(self, query_text, n_results=5, where=None):
        """Semantic search with automatic wing/room name matching.

        If the query matches a wing or room name and no explicit filter
        is given, auto-filter to that wing/room for relevant results.
        """
        # Auto-detect wing/room name in query when no filter specified
        if not where:
            where = self._auto_detect_filter(query_text)

        emb = embed([query_text])[0]
        clauses, params = self._build_where(where)

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if clauses:
            # HNSW doesn't support pre-filtering. Wrap in a transaction
            # with index scans disabled to force sequential scan.
            old_autocommit = self.conn().autocommit
            self.conn().autocommit = False
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
            sql = f"""SELECT id, wing, room, content, source_file, chunk_index,
                             added_by, filed_at, metadata,
                             embedding <=> %s AS distance
                      FROM drawers WHERE {clauses}
                      ORDER BY distance LIMIT {int(n_results)}"""
            query_params = [emb] + params
        else:
            sql = f"""SELECT id, wing, room, content, source_file, chunk_index,
                            added_by, filed_at, metadata,
                            embedding <=> %s AS distance
                     FROM drawers
                     ORDER BY distance LIMIT {int(n_results)}"""
            query_params = [emb]

        cur.execute(sql, query_params)
        rows = cur.fetchall()

        # Restore autocommit if we changed it
        if clauses:
            self.conn().commit()
            self.conn().autocommit = old_autocommit

        ids, documents, metadatas, distances = [], [], [], []
        for r in rows:
            ids.append(r["id"])
            documents.append(r["content"])
            distances.append(float(r["distance"]))
            m = {
                "wing": r["wing"], "room": r["room"],
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
            "ids": [ids], "documents": [documents],
            "metadatas": [metadatas], "distances": [distances],
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
        emb = embed([content])[0]
        meta = json.dumps(metadata or {})
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

    def add_triple(self, subject, predicate, obj, valid_from=None, valid_to=None,
                   confidence=1.0, source_closet=None, source_file=None):
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
            (triple_id, sub_id, pred_norm, obj_id, valid_from, valid_to,
             confidence, source_closet, source_file),
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
            results += self._query_triples(
                "t.subject = %s", eid, as_of, "outgoing"
            )
        if direction in ("incoming", "both"):
            results += self._query_triples(
                "t.object = %s", eid, as_of, "incoming"
            )
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
            sql += " AND (t.valid_to IS NULL OR t.valid_to >= %s)"
            params += [as_of, as_of]

        sql += " ORDER BY t.valid_from ASC NULLS LAST"

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "direction": direction,
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "confidence": r["confidence"],
                "source_closet": r["source_closet"],
                "current": r["valid_to"] is None,
            })
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
            sql += " AND (t.valid_to IS NULL OR t.valid_to >= %s)"
            params += [as_of, as_of]

        cur = self.conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return [{
            "subject": r["sub_name"], "predicate": r["predicate"],
            "object": r["obj_name"], "valid_from": r["valid_from"],
            "valid_to": r["valid_to"], "confidence": r["confidence"],
            "current": r["valid_to"] is None,
        } for r in cur.fetchall()]

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
        return [{
            "subject": r["sub_name"], "predicate": r["predicate"],
            "object": r["obj_name"], "valid_from": r["valid_from"],
            "valid_to": r["valid_to"], "current": r["valid_to"] is None,
        } for r in cur.fetchall()]

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
            "entities": entities, "triples": triples,
            "current_facts": current, "expired_facts": triples - current,
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
                name, etype,
                {"gender": facts.get("gender", ""), "birthday": facts.get("birthday", "")},
            )

            parent = facts.get("parent")
            if parent:
                self.add_triple(name, "child_of", parent.capitalize(),
                                valid_from=facts.get("birthday"))

            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())

            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(name, "is_child_of",
                                facts.get("parent", "").capitalize() or name,
                                valid_from=facts.get("birthday"))
            elif relationship == "husband":
                self.add_triple(name, "is_partner_of",
                                facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(name, "is_sibling_of",
                                facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(name, "is_pet_of",
                                facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")

            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(),
                                valid_from="2025-01-01")

    def _auto_detect_filter(self, query_text):
        """Check if query contains a wing or room name and return a filter."""
        query_lower = query_text.lower().strip()
        query_words = set(query_lower.replace("-", "_").replace(" ", "_").split("_"))
        cur = self.conn().cursor()

        # Check exact wing match
        cur.execute("SELECT DISTINCT wing FROM drawers")
        wings = [r[0] for r in cur.fetchall()]
        for w in wings:
            if w.lower() == query_lower.replace(" ", "_").replace("-", "_"):
                return {"wing": w}
            # Also match if wing name is a significant word in the query
            if w.lower() in query_words and len(w) > 2:
                return {"wing": w}

        # Check exact room match
        cur.execute("SELECT DISTINCT room FROM drawers")
        rooms = [r[0] for r in cur.fetchall()]
        for r in rooms:
            if r.lower() == query_lower.replace(" ", "_").replace("-", "_"):
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
