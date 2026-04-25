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
