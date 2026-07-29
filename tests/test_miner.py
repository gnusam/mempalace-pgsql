import os
import shutil
import tempfile
from pathlib import Path

import yaml

from mempalace.db import PalaceDB
from mempalace import miner
from mempalace.miner import mine, scan_project, process_file


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mempalace:mempalace@localhost:5433/mempalace"
)


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scanned_files(project_root: Path, **kwargs):
    files = scan_project(str(project_root), **kwargs)
    return sorted(path.relative_to(project_root).as_posix() for path in files)


def test_project_mining():
    tmpdir = tempfile.mkdtemp()
    db = None
    try:
        project_root = Path(tmpdir).resolve()
        os.makedirs(project_root / "backend")

        write_file(
            project_root / "backend" / "app.py", "def main():\n    print('hello world')\n" * 20
        )
        with open(project_root / "mempalace.yaml", "w") as f:
            yaml.dump(
                {
                    "wing": "test_mine_project",
                    "rooms": [
                        {"name": "backend", "description": "Backend code"},
                        {"name": "general", "description": "General"},
                    ],
                },
                f,
            )

        mine(str(project_root), palace_path=None)

        db = PalaceDB(DATABASE_URL)
        assert db.count(where={"wing": "test_mine_project"}) > 0
    finally:
        if db is not None:
            cur = db.conn().cursor()
            cur.execute("DELETE FROM drawers WHERE wing = 'test_mine_project'")
            db.close()
        shutil.rmtree(tmpdir)


def test_scan_project_respects_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "ignored.py\ngenerated/\n")
        write_file(project_root / "src" / "app.py", "print('hello')\n" * 20)
        write_file(project_root / "ignored.py", "print('ignore me')\n" * 20)
        write_file(project_root / "generated" / "artifact.py", "print('artifact')\n" * 20)

        assert scanned_files(project_root) == ["src/app.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_respects_nested_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.log\n")
        write_file(project_root / "subrepo" / ".gitignore", "tasks/\n")
        write_file(project_root / "subrepo" / "src" / "main.py", "print('main')\n" * 20)
        write_file(project_root / "subrepo" / "tasks" / "task.py", "print('task')\n" * 20)
        write_file(project_root / "subrepo" / "debug.log", "debug\n" * 20)

        assert scanned_files(project_root) == ["subrepo/src/main.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_nested_gitignore_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "*.csv\n")
        write_file(project_root / "subrepo" / ".gitignore", "!keep.csv\n")
        write_file(project_root / "drop.csv", "a,b,c\n" * 20)
        write_file(project_root / "subrepo" / "keep.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root) == ["subrepo/keep.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_allows_gitignore_negation_when_parent_dir_is_visible():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/*\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == ["generated/keep.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_does_not_reinclude_file_from_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n!generated/keep.py\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root) == []
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_disable_gitignore():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "data/\n")
        write_file(project_root / "data" / "stuff.csv", "a,b,c\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["data/stuff.csv"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_ignored_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "docs/\n")
        write_file(project_root / "docs" / "guide.md", "# Guide\n" * 20)

        assert scanned_files(project_root, include_ignored=["docs"]) == ["docs/guide.md"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_specific_ignored_file():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "generated/\n")
        write_file(project_root / "generated" / "drop.py", "print('drop')\n" * 20)
        write_file(project_root / "generated" / "keep.py", "print('keep')\n" * 20)

        assert scanned_files(project_root, include_ignored=["generated/keep.py"]) == [
            "generated/keep.py"
        ]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_can_include_exact_file_without_known_extension():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".gitignore", "README\n")
        write_file(project_root / "README", "hello\n" * 20)

        assert scanned_files(project_root, include_ignored=["README"]) == ["README"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_include_override_beats_skip_dirs():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)

        assert scanned_files(
            project_root,
            respect_gitignore=False,
            include_ignored=[".pytest_cache"],
        ) == [".pytest_cache/cache.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skip_dirs_still_apply_without_override():
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / ".pytest_cache" / "cache.py", "print('cache')\n" * 20)
        write_file(project_root / "main.py", "print('main')\n" * 20)

        assert scanned_files(project_root, respect_gitignore=False) == ["main.py"]
    finally:
        shutil.rmtree(tmpdir)


# ─── Anti-noise filters (minification, symlink skip, file-size ceiling) ──


def test_scan_project_skips_minified_bundles_by_name():
    """Filename patterns like *.min.js, *-bundle.js, *.umd.js are dropped."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / "app.js", "console.log('hello')\n" * 20)
        write_file(project_root / "vendor.min.js", "var a=1;" * 50)
        write_file(project_root / "style.min.css", "body{margin:0}" * 50)
        write_file(project_root / "swagger-ui-bundle.js", "var b=2;" * 50)
        write_file(project_root / "framework.umd.js", "var c=3;" * 50)

        assert scanned_files(project_root) == ["app.js"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skips_symlinks():
    """Symlinks point anywhere; refuse to follow them at scan time."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        real = project_root / "real.py"
        write_file(real, "print('real')\n" * 20)
        os.symlink(real, project_root / "linked.py")

        assert scanned_files(project_root) == ["real.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skips_oversized_files(monkeypatch):
    """Files exceeding MAX_FILE_SIZE are dropped before being opened."""
    monkeypatch.setattr(miner, "MAX_FILE_SIZE", 100)  # 100 bytes cap
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        write_file(project_root / "small.py", "x = 1\n")  # well under 100 bytes
        write_file(project_root / "big.py", "y = 2\n" * 50)  # ~300 bytes, over cap

        assert scanned_files(project_root) == ["small.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_scan_project_skips_new_lockfiles_by_name():
    """yarn.lock, composer.lock, etc are added to the skip list."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()

        write_file(project_root / "app.py", "print('hello')\n" * 20)
        write_file(project_root / "yarn.lock", "lock content\n" * 20)
        write_file(project_root / "composer.lock", "lock content\n" * 20)
        write_file(project_root / "Cargo.lock", "lock content\n" * 20)
        write_file(project_root / "poetry.lock", "lock content\n" * 20)
        write_file(project_root / "pnpm-lock.yaml", "lock content\n" * 20)

        assert scanned_files(project_root) == ["app.py"]
    finally:
        shutil.rmtree(tmpdir)


def test_process_file_skips_machine_generated_long_lines():
    """Files with average line length > MAX_AVG_LINE_LENGTH are dropped."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        minified = project_root / "blob.json"
        # One 2000-char line → avg line length 2000, well over the 400 cap
        write_file(minified, "x" * 2000)

        rooms = [{"name": "general", "description": "general", "keywords": []}]
        count, room = process_file(
            filepath=minified,
            project_path=project_root,
            db=None,
            wing="test",
            rooms=rooms,
            agent="test",
            dry_run=True,
        )
        assert count == 0
        # Upstream 091c2fe (PR #687): early-exit paths return "general" so
        # the dry-run summary printer doesn't crash on a None dict key.
        assert room == "general"
    finally:
        shutil.rmtree(tmpdir)


class RecordingDB:
    """Test double for PalaceDB exposing just what process_file touches."""

    def __init__(self):
        self.replace_calls = []

    def file_already_mined(self, source_file, ingest_mode=None, extract_mode=None):
        return False

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
                "wing": wing,
                "chunks": chunks,
                "source_file": source_file,
                "source_mtime": source_mtime,
            }
        )
        return [f"id_{c['chunk_index']}" for c in chunks]


def test_process_file_files_all_chunks_via_one_atomic_replace():
    """All of a file's chunks go through a single replace_file_drawers call
    (one transaction), not per-chunk autocommitted inserts — a mine killed
    mid-file must leave the previous state intact (upstream PR #2088)."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        source = project_root / "code.py"
        write_file(source, "def hello():\n    return 'world'\n" * 40)

        db = RecordingDB()
        rooms = [{"name": "general", "description": "general", "keywords": []}]
        count, room = process_file(
            filepath=source,
            project_path=project_root,
            db=db,
            wing="test",
            rooms=rooms,
            agent="test",
            dry_run=False,
        )
        assert len(db.replace_calls) == 1
        assert count == len(db.replace_calls[0]["chunks"]) > 0
        assert all(c["room"] == room for c in db.replace_calls[0]["chunks"])
    finally:
        shutil.rmtree(tmpdir)


def test_process_file_stamps_read_time_mtime_not_a_later_restat(monkeypatch):
    """The stored source_mtime must be the one captured with the read, not a
    later os.path.getmtime() — a file appended to in between would otherwise
    be stamped as current and its tail permanently skipped (upstream PR
    #2088's TOCTOU gap)."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        source = project_root / "session.md"
        write_file(source, "some conversation content\n" * 40)
        real_mtime = source.stat().st_mtime

        # Any later re-stat would observe this bogus value instead.
        monkeypatch.setattr(os.path, "getmtime", lambda path: real_mtime + 9999)

        db = RecordingDB()
        rooms = [{"name": "general", "description": "general", "keywords": []}]
        process_file(
            filepath=source,
            project_path=project_root,
            db=db,
            wing="test",
            rooms=rooms,
            agent="test",
            dry_run=False,
        )
        assert db.replace_calls[0]["source_mtime"] == real_mtime
    finally:
        shutil.rmtree(tmpdir)


def test_process_file_purges_scope_when_content_yields_nothing():
    """A file whose current content falls below the chunk minimum must purge
    its old drawers on re-mine, not silently keep serving them (upstream PR
    #2088's dangling-closet analog)."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        source = project_root / "shrunk.md"
        write_file(source, "tiny")

        db = RecordingDB()
        rooms = [{"name": "general", "description": "general", "keywords": []}]
        count, room = process_file(
            filepath=source,
            project_path=project_root,
            db=db,
            wing="test",
            rooms=rooms,
            agent="test",
            dry_run=False,
        )
        assert count == 0
        assert len(db.replace_calls) == 1
        assert db.replace_calls[0]["chunks"] == []
    finally:
        shutil.rmtree(tmpdir)


def test_process_file_accepts_normal_line_lengths():
    """Sanity check: ordinary multi-line files still pass the line-length gate."""
    tmpdir = tempfile.mkdtemp()
    try:
        project_root = Path(tmpdir).resolve()
        normal = project_root / "code.py"
        write_file(normal, "def hello():\n    return 'world'\n" * 40)

        rooms = [{"name": "general", "description": "general", "keywords": []}]
        count, room = process_file(
            filepath=normal,
            project_path=project_root,
            db=None,
            wing="test",
            rooms=rooms,
            agent="test",
            dry_run=True,
        )
        assert count > 0
        assert room == "general"
    finally:
        shutil.rmtree(tmpdir)
