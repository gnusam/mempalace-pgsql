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


# --- strip_noise: harness chrome removal (upstream PRs #1909 + #2064) ---


def test_strips_each_known_noise_tag():
    from mempalace.normalize import _NOISE_TAGS, strip_noise

    for tag in _NOISE_TAGS:
        text = f"> Real.\n<{tag}>noise body</{tag}>\n> Also real."
        out = strip_noise(text)
        assert tag not in out, f"{tag} leaked into output"
        assert "Real." in out


def test_strips_block_with_blank_lines_in_body():
    """A task-notification carrying a subagent's multi-paragraph result
    (blank lines inside the block) must not leak wholesale into drawers."""
    from mempalace.normalize import strip_noise

    text = (
        "> User:\n"
        "<task-notification>\n"
        "<task-id>abc123</task-id>\n"
        "<result># Report\n"
        "\n"
        "Paragraph one.\n"
        "\n"
        "Paragraph two.</result>\n"
        "</task-notification>\n"
        "> Real message."
    )
    out = strip_noise(text)
    assert "task-notification" not in out
    assert "Paragraph one" not in out
    assert "Real message." in out


def test_dangling_open_tag_does_not_merge_with_later_block():
    """A line-anchored dangling open tag must not merge with a later
    complete block of the same tag and eat the real content between."""
    from mempalace.normalize import strip_noise

    text = (
        "<system-reminder>dangling, never closed\n"
        "> User: real content here\n"
        "\n"
        "<system-reminder>complete block</system-reminder>\n"
        "> Tail."
    )
    out = strip_noise(text)
    assert "real content here" in out
    assert "Tail." in out
    assert "complete block" not in out


def test_strips_indented_slash_command_chrome():
    """Claude Code emits slash-command chrome with indented tags; the line
    anchor must tolerate leading whitespace, and command-args is noise too."""
    from mempalace.normalize import strip_noise

    text = (
        "<command-name>/model</command-name>\n"
        "            <command-message>model</command-message>\n"
        "            <command-args></command-args>\n"
        "> Real message."
    )
    out = strip_noise(text)
    for tag in ("command-name", "command-message", "command-args"):
        assert tag not in out, f"{tag} leaked"
    assert "Real message." in out


def test_strips_ansi_escapes_but_not_prose_naming_them():
    from mempalace.normalize import strip_noise

    text = "> ok\n\x1b[1mBOLD\x1b[0m and \x1b]8;;https://x\x07link\x1b]8;;\x07 done"
    out = strip_noise(text)
    assert "\x1b" not in out
    assert "BOLD" in out
    assert "link" in out
    # Prose that merely *names* a sequence survives (no ESC byte).
    named = "the [1m code and ESC[0m marker"
    assert strip_noise(named) == named


def test_normalize_strips_chrome_from_claude_code_jsonl():
    """End-to-end: a Claude Code session whose user turn embeds a
    system-reminder block comes out of normalize() without it."""
    entries = [
        {
            "type": "user",
            "message": {"content": "hello there\n<system-reminder>secret chrome</system-reminder>"},
        },
        {"type": "assistant", "message": {"content": "hi, how can I help?"}},
        {"type": "user", "message": {"content": "explain memory palaces"}},
        {"type": "assistant", "message": {"content": "gladly — they are spatial mnemonics"}},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        out = normalize(path)
        assert "system-reminder" not in out
        assert "secret chrome" not in out
        assert "hello there" in out
        assert "spatial mnemonics" in out
    finally:
        os.unlink(path)
