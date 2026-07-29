import os
import json
import tempfile
from mempalace.config import MempalaceConfig


def test_default_config():
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert "palace" in cfg.palace_path
    assert cfg.collection_name == "mempalace_drawers"


def test_config_from_file():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palace_path": "/custom/palace"}, f)
    cfg = MempalaceConfig(config_dir=tmpdir)
    assert cfg.palace_path == "/custom/palace"


def test_env_override():
    os.environ["MEMPALACE_PALACE_PATH"] = "/env/palace"
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.palace_path == "/env/palace"
    del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_expanduser():
    os.environ["MEMPALACE_PALACE_PATH"] = "~/mempalace-test"
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        assert "~" not in cfg.palace_path
        assert cfg.palace_path.endswith("mempalace-test")
        assert cfg.palace_path == os.path.expanduser("~/mempalace-test")
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_abspath_collapses_traversal():
    os.environ["MEMPALACE_PALACE_PATH"] = "/tmp/palace/../mempalace-test"
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        assert ".." not in cfg.palace_path
        assert cfg.palace_path == "/tmp/mempalace-test"
    finally:
        del os.environ["MEMPALACE_PALACE_PATH"]


def test_env_path_legacy_alias_normalized():
    os.environ.pop("MEMPALACE_PALACE_PATH", None)
    os.environ["MEMPAL_PALACE_PATH"] = "~/legacy-alias/../mempalace-test"
    try:
        cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
        assert "~" not in cfg.palace_path
        assert ".." not in cfg.palace_path
        assert cfg.palace_path == os.path.expanduser("~/mempalace-test")
    finally:
        del os.environ["MEMPAL_PALACE_PATH"]


def test_init():
    tmpdir = tempfile.mkdtemp()
    cfg = MempalaceConfig(config_dir=tmpdir)
    cfg.init()
    assert os.path.exists(os.path.join(tmpdir, "config.json"))


# ── sanitize_iso_date (port of upstream 4d98b05 + abe8576) ────────────────


def test_iso_date_accepts_full_date():
    from mempalace.config import sanitize_iso_date

    assert sanitize_iso_date("2026-03-15") == "2026-03-15"


def test_iso_date_strips_whitespace():
    from mempalace.config import sanitize_iso_date

    assert sanitize_iso_date(" 2026-03-15 ") == "2026-03-15"


def test_iso_date_passes_through_none_and_empty():
    from mempalace.config import sanitize_iso_date

    assert sanitize_iso_date(None) is None
    assert sanitize_iso_date("") == ""


def test_iso_date_rejects_partial_dates():
    import pytest

    from mempalace.config import sanitize_iso_date

    for bad in ("2026", "2026-03"):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            sanitize_iso_date(bad)


def test_iso_date_rejects_natural_language():
    import pytest

    from mempalace.config import sanitize_iso_date

    with pytest.raises(ValueError, match="as_of"):
        sanitize_iso_date("March 2026", "as_of")


def test_iso_date_rejects_out_of_range_components():
    import pytest

    from mempalace.config import sanitize_iso_date

    for bad in ("2026-13-01", "2026-00-10", "2026-01-32"):
        with pytest.raises(ValueError):
            sanitize_iso_date(bad)


def test_iso_date_rejects_non_string():
    import pytest

    from mempalace.config import sanitize_iso_date

    with pytest.raises(ValueError, match="must be a string"):
        sanitize_iso_date(20260315)
