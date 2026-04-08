import os
import tempfile
import shutil
from mempalace.convo_miner import mine_convos
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
