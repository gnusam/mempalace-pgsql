import os
import tempfile
import shutil
from pathlib import Path

from mempalace import convo_miner
from mempalace.convo_miner import mine_convos, scan_convos
from mempalace.db import PalaceDB


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mempalace:mempalace@localhost:5433/mempalace"
)


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    mine_convos(tmpdir, palace_path=None, wing="test_convo_mining")

    # Verify via PostgreSQL
    db = PalaceDB(DATABASE_URL)
    count = db.count(where={"wing": "test_convo_mining"})
    assert count >= 2

    # Verify search works
    results = db.query("memory persistence", n_results=1, where={"wing": "test_convo_mining"})
    assert len(results["documents"][0]) > 0

    # Cleanup
    cur = db.conn().cursor()
    cur.execute("DELETE FROM drawers WHERE wing = 'test_convo_mining'")
    db.close()
    shutil.rmtree(tmpdir)


def test_scan_convos_skips_symlinks():
    """Symlinks in the convo source tree are refused at scan time."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir)
        real = root / "session.txt"
        real.write_text("> hello\nworld\n", encoding="utf-8")
        os.symlink(real, root / "linked.txt")

        files = [p.name for p in scan_convos(str(root))]
        assert "session.txt" in files
        assert "linked.txt" not in files
    finally:
        shutil.rmtree(tmpdir)


def test_scan_convos_skips_oversized_files(monkeypatch):
    """Convo files exceeding MAX_FILE_SIZE are dropped before open()."""
    monkeypatch.setattr(convo_miner, "MAX_FILE_SIZE", 50)
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir)
        (root / "small.txt").write_text("> q\na\n", encoding="utf-8")  # ~7 bytes
        (root / "big.txt").write_text("> q\n" + "x" * 200 + "\n", encoding="utf-8")  # >50

        files = [p.name for p in scan_convos(str(root))]
        assert "small.txt" in files
        assert "big.txt" not in files
    finally:
        shutil.rmtree(tmpdir)


# --- Upstream 9b60c6e (PR #708): full AI response, not first 8 lines ---


def test_chunk_by_exchange_preserves_full_ai_response():
    """The 8-line truncation `ai_lines[:8]` is gone — every line of the AI
    response must end up in some drawer (split across drawers if needed)."""
    user_turn = "> what's up"
    # 12 short lines so we exceed the previous 8-line cap but stay under CHUNK_SIZE
    ai_lines = [f"line-{i:02d}-content" for i in range(12)]
    text = user_turn + "\n" + "\n".join(ai_lines) + "\n"

    chunks = convo_miner.chunk_exchanges(text)
    combined = " ".join(c["content"] for c in chunks)
    for line in ai_lines:
        assert line in combined, f"missing line: {line}"


def test_chunk_by_exchange_splits_oversize_exchange_across_drawers():
    """Exchanges longer than CHUNK_SIZE split into consecutive drawers
    instead of being silently truncated (upstream 9b60c6e)."""
    user_turn = "> long story incoming"
    # 3000 chars of response — well over the 800 CHUNK_SIZE threshold
    ai_response = "word " * 600
    text = f"{user_turn}\n{ai_response}\n\n> next\nshort\n\n> follow\nup\n"

    chunks = convo_miner.chunk_exchanges(text)
    # At least 4 chunks: ≥3 from the oversize exchange (3000/800 = 3.75) +
    # the two short follow-ups
    assert len(chunks) >= 4
    # Total chunked content is at least as long as the original AI response
    total_len = sum(len(c["content"]) for c in chunks)
    assert total_len >= len(ai_response)


# --- Upstream 87e8baf (PR #732): 0-chunk files get a sentinel ---


def test_register_empty_file_purge_stale_removes_only_scoped_rows():
    """When a file's content now yields nothing, its old drawers for THIS
    extract mode must go (in the same transaction as the sentinel), while
    other scopes' rows survive — upstream PR #2089's over-match class."""
    tmpdir = tempfile.mkdtemp()
    try:
        f = Path(tmpdir) / "now_empty.jsonl"
        f.write_text("hi\n", encoding="utf-8")
        src = str(f)

        db = PalaceDB(DATABASE_URL)
        wing = "test_register_purge"
        try:
            db.add_drawer(
                wing,
                "general",
                "old exchange drawer " * 10,
                source_file=src,
                chunk_index=0,
                metadata={"ingest_mode": "convos", "extract_mode": "exchange"},
            )
            db.add_drawer(
                wing,
                "decision",
                "general-mode drawer " * 10,
                source_file=src,
                chunk_index=0,
                metadata={"ingest_mode": "convos", "extract_mode": "general"},
            )
            db.register_empty_file(src, wing=wing, purge_stale=True, extract_mode="exchange")
            cur = db.conn().cursor()
            cur.execute(
                "SELECT metadata->>'extract_mode', metadata->>'ingest_mode' "
                "FROM drawers WHERE source_file = %s",
                (src,),
            )
            remaining = cur.fetchall()
            modes = {r[1] for r in remaining}
            assert ("exchange", "convos") not in [(r[0], r[1]) for r in remaining], (
                "stale exchange-mode drawers must be purged"
            )
            assert ("general", "convos") in [(r[0], r[1]) for r in remaining], (
                "the other extract mode's drawers must survive the purge"
            )
            assert "registry" in modes
            assert db.file_already_mined(src, ingest_mode="convos", extract_mode="exchange")
        finally:
            cur = db.conn().cursor()
            cur.execute("DELETE FROM drawers WHERE wing = %s", (wing,))
            db.close()
    finally:
        shutil.rmtree(tmpdir)


def test_register_empty_file_without_purge_keeps_existing_rows():
    """The transient-failure path (normalize() raising) registers the
    sentinel but must NOT delete mined data — a parse hiccup is not proof
    the content is gone (upstream PR #2088's failed-purge lesson)."""
    tmpdir = tempfile.mkdtemp()
    try:
        f = Path(tmpdir) / "hiccup.jsonl"
        f.write_text("hi\n", encoding="utf-8")
        src = str(f)

        db = PalaceDB(DATABASE_URL)
        wing = "test_register_keep"
        try:
            db.add_drawer(
                wing,
                "general",
                "previously mined drawer " * 10,
                source_file=src,
                chunk_index=0,
                metadata={"ingest_mode": "convos", "extract_mode": "exchange"},
            )
            db.register_empty_file(src, wing=wing)
            cur = db.conn().cursor()
            cur.execute(
                "SELECT COUNT(*) FROM drawers WHERE source_file = %s "
                "AND metadata->>'ingest_mode' = 'convos'",
                (src,),
            )
            assert cur.fetchone()[0] == 1
        finally:
            cur = db.conn().cursor()
            cur.execute("DELETE FROM drawers WHERE wing = %s", (wing,))
            db.close()
    finally:
        shutil.rmtree(tmpdir)


def test_register_empty_file_stores_caller_mtime_not_a_restat():
    """The sentinel must carry the mtime captured with the read, not a fresh
    getmtime() at register time (upstream PR #2088's TOCTOU gap)."""
    tmpdir = tempfile.mkdtemp()
    try:
        f = Path(tmpdir) / "toctou.jsonl"
        f.write_text("hi\n", encoding="utf-8")
        src = str(f)

        db = PalaceDB(DATABASE_URL)
        wing = "test_register_mtime"
        try:
            db.register_empty_file(src, wing=wing, source_mtime=1234.5)
            cur = db.conn().cursor()
            cur.execute(
                "SELECT metadata->>'source_mtime' FROM drawers WHERE source_file = %s",
                (src,),
            )
            assert float(cur.fetchone()[0]) == 1234.5
        finally:
            cur = db.conn().cursor()
            cur.execute("DELETE FROM drawers WHERE wing = %s", (wing,))
            db.close()
    finally:
        shutil.rmtree(tmpdir)


def test_mine_convos_files_atomically_with_read_time_mtime(monkeypatch):
    """mine_convos must file each transcript through one scoped
    replace_file_drawers call carrying the mtime captured before
    normalize() read the file — session logs are appended to while being
    mined (upstream PRs #2088/#2089)."""

    class RecordingDB:
        def __init__(self):
            self.replace_calls = []

        def file_already_mined(self, source_file, ingest_mode=None, extract_mode=None):
            return False

        def register_empty_file(self, *args, **kwargs):
            raise AssertionError("should not be called: chunks exist")

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
            self.replace_calls.append(
                {
                    "chunks": chunks,
                    "source_mtime": source_mtime,
                    "ingest_mode": ingest_mode,
                    "extract_mode": extract_mode,
                }
            )
            return [f"id_{c['chunk_index']}" for c in chunks]

    tmpdir = tempfile.mkdtemp()
    try:
        f = Path(tmpdir) / "session.jsonl"
        f.write_text("USER: hello\nAI: world\n" * 50, encoding="utf-8")
        real_mtime = f.stat().st_mtime

        db = RecordingDB()
        monkeypatch.setattr(convo_miner, "get_db_instance", lambda palace_path=None: db)
        monkeypatch.setattr(convo_miner, "normalize", lambda path: "USER: hello\nAI: world\n" * 50)
        monkeypatch.setattr(
            convo_miner,
            "chunk_exchanges",
            lambda content: [
                {"content": "exchange one " * 20, "chunk_index": 0},
                {"content": "exchange two " * 20, "chunk_index": 1},
            ],
        )
        # Any later re-stat would observe this bogus value instead.
        monkeypatch.setattr(os.path, "getmtime", lambda path: real_mtime + 9999)

        mine_convos(tmpdir, palace_path=None, wing="test_convo_atomic")

        assert len(db.replace_calls) == 1
        call = db.replace_calls[0]
        assert len(call["chunks"]) == 2
        assert call["ingest_mode"] == "convos"
        assert call["extract_mode"] == "exchange"
        assert call["source_mtime"] == real_mtime
    finally:
        shutil.rmtree(tmpdir)


def test_register_empty_file_makes_file_already_mined_true():
    """A file that produces zero chunks must register a no-embedding sentinel
    so file_already_mined() returns True on the next mine run."""
    tmpdir = tempfile.mkdtemp()
    try:
        empty = Path(tmpdir) / "empty.txt"
        empty.write_text("hi\n", encoding="utf-8")  # well under MIN_CHUNK_SIZE

        db = PalaceDB(DATABASE_URL)
        wing = "test_register_empty"
        try:
            assert db.file_already_mined(str(empty)) is False
            sentinel_id = db.register_empty_file(str(empty), wing=wing)
            assert sentinel_id is not None
            assert db.file_already_mined(str(empty)) is True
        finally:
            cur = db.conn().cursor()
            cur.execute("DELETE FROM drawers WHERE wing = %s", (wing,))
            db.close()
    finally:
        shutil.rmtree(tmpdir)
