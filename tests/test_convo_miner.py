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
