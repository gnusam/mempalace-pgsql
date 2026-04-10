import os
import json
import tempfile

import pytest

from mempalace import normalize as normalize_mod
from mempalace.normalize import normalize


def test_plain_text():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("Hello world\nSecond line\n")
    f.close()
    result = normalize(f.name)
    assert "Hello world" in result
    os.unlink(f.name)


def test_claude_json():
    data = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    result = normalize(f.name)
    assert "Hi" in result
    os.unlink(f.name)


def test_empty():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.close()
    result = normalize(f.name)
    assert result.strip() == ""
    os.unlink(f.name)


def test_normalize_refuses_oversized_file(monkeypatch):
    """Files larger than MAX_NORMALIZE_FILE_SIZE raise IOError via the size guard."""
    # Lower the ceiling to 10 bytes so we can trigger the guard without
    # writing half a gigabyte to disk.
    monkeypatch.setattr(normalize_mod, "MAX_NORMALIZE_FILE_SIZE", 10)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("a" * 1000)  # 1000 bytes, well over the 10-byte cap
    f.close()
    try:
        with pytest.raises(IOError, match="too large"):
            normalize(f.name)
    finally:
        os.unlink(f.name)
