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
