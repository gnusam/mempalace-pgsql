#!/usr/bin/env python3
"""
normalize.py — Convert any chat export format to MemPalace transcript format.

Supported:
    - Plain text with > markers (pass through)
    - Claude.ai JSON export
    - ChatGPT conversations.json
    - Claude Code JSONL
    - OpenAI Codex CLI JSONL
    - Slack JSON export
    - Plain text (pass through for paragraph chunking)

No API key. No internet. Everything local.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


# Hard ceiling on the size of a single file that normalize() will slurp
# into memory. Exposed as a module constant so tests can monkey-patch it.
# Ported from upstream 0720fb8 (PR #399).
MAX_NORMALIZE_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


# ── Harness-chrome noise stripping ───────────────────────────────────────
#
# Claude Code (and its hooks) inject XML-ish envelopes and terminal escape
# codes into transcripts. These waste drawer space and pollute search
# results. Combined port of upstream strip_noise plus open PRs #1909 (full
# slash-command envelope + ANSI, issue #1333) and #2064 (indent-tolerant
# anchor, multi-paragraph bodies) — reconciled here since this fork never
# carried the base strip_noise layer.
#
# Verbatim is sacred — every tag pattern is anchored to a line start and a
# tag body may not swallow another opening of the same tag, so a stray
# unclosed tag can never merge with a later block and eat the real content
# between them. When in doubt, leave text alone.

_NOISE_TAGS = (
    "system-reminder",
    "command-message",
    "command-name",
    "command-args",
    "local-command-caveat",
    "local-command-stdout",
    "local-command-stderr",
    "task-notification",
    "user-prompt-submit-hook",
    "hook_output",
)


def _tag_pattern(name: str) -> "re.Pattern[str]":
    # Opening tag must begin a line — only a `> ` blockquote marker (added by
    # _messages_to_transcript) and/or indentation may precede it, since Claude
    # Code emits slash-command chrome with indented tags. Body is lazy and
    # may cross blank lines — task-notification and system-reminder blocks
    # legitimately carry multi-paragraph payloads (subagent results, recalled
    # memories) — but may not contain another opening of the same tag, so a
    # dangling open tag can never merge with a later block and eat the content
    # between them. Closing tag eats optional trailing whitespace + newline.
    return re.compile(
        rf"(?m)^(?:> )?[ \t]*<{name}(?:\s[^>]*)?>"
        rf"(?:(?!<{name}[\s>])[\s\S])*?"
        rf"</{name}>[ \t]*\n?"
    )


_NOISE_TAG_PATTERNS = tuple(_tag_pattern(name) for name in _NOISE_TAGS)

# ANSI escape sequences that Claude Code's Bash tool preserves verbatim from
# terminal output (upstream #1333). Each escape is several BPE tokens, so they
# bloat embeddings and pollute search. Unlike the patterns above these are NOT
# line-anchored — they are anchored on the literal ESC byte (0x1B), a control
# character that never appears in legitimate prose, so text that merely
# *names* a sequence like "[1m" or "ESC[0m" survives untouched. Both follow
# ECMA-48 and are ReDoS-safe by construction (disjoint / negated character
# classes, no overlapping nested quantifiers).
#
# CSI (Control Sequence Introducer): ESC [ , parameter bytes (0x30-0x3F),
# intermediate bytes (0x20-0x2F), one final byte (0x40-0x7E). Covers all SGR
# color/style codes (the dominant Bash-output noise) plus cursor/erase.
_ANSI_CSI_RE = re.compile(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")
# OSC (Operating System Command): ESC ] , string payload, terminated by BEL
# (0x07) or ST (ESC \). Covers hyperlinks (ESC]8;;URL BEL) and window titles.
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Claude Code collapsed-output chrome, line-anchored.
_COLLAPSED_LINES_RE = re.compile(r"(?m)^(?:> )?…\s*\+\d+ lines.*\n?")


def strip_noise(text: str) -> str:
    """Remove system tags, hook output, and Claude Code UI chrome from text.

    Only known-noise shapes are touched; everything else passes through
    verbatim.
    """
    for pattern in _NOISE_TAG_PATTERNS:
        text = pattern.sub("", text)
    text = _COLLAPSED_LINES_RE.sub("", text)
    # Strip the collapsed-output chrome "[N tokens] (ctrl+o to expand)".
    # Narrow shape — a bare "(ctrl+o to expand)" in user prose stays intact.
    text = re.sub(r"\s*\[\d+\s+tokens?\]\s*\(ctrl\+o to expand\)", "", text)
    # Strip ANSI escape sequences from terminal / Bash-tool output (#1333).
    # Applied after tag removal so escapes nested inside a stripped envelope
    # (e.g. <local-command-stdout>) are already gone with the tag.
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_CSI_RE.sub("", text)
    # Collapse runs of blank lines created by the removals
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize(filepath: str) -> str:
    """
    Load a file and normalize to transcript format if it's a chat export.
    Plain text files pass through unchanged.
    """
    try:
        # Safety limit: refuse to slurp pathologically large files into
        # memory. Ported from upstream 0720fb8 (PR #399, fixes
        # milla-jovovich/mempalace#396), by @bensig.
        file_size = os.path.getsize(filepath)
        if file_size > MAX_NORMALIZE_FILE_SIZE:
            raise IOError(f"File too large ({file_size // (1024 * 1024)} MB): {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}")

    if not content.strip():
        return content

    # Already has > markers — pass through (minus harness chrome)
    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return strip_noise(content)

    # Try JSON normalization
    ext = Path(filepath).suffix.lower()
    if ext in (".json", ".jsonl") or content.strip()[:1] in ("{", "["):
        normalized = _try_normalize_json(content)
        if normalized:
            return strip_noise(normalized)

    return strip_noise(content)


def _try_normalize_json(content: str) -> Optional[str]:
    """Try all known JSON chat schemas."""

    normalized = _try_claude_code_jsonl(content)
    if normalized:
        return normalized

    normalized = _try_codex_jsonl(content)
    if normalized:
        return normalized

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    for parser in (_try_claude_ai_json, _try_chatgpt_json, _try_slack_json):
        normalized = parser(data)
        if normalized:
            return normalized

    return None


def _try_claude_code_jsonl(content: str) -> Optional[str]:
    """Claude Code JSONL sessions."""
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if msg_type in ("human", "user"):
            text = _extract_content(message.get("content", ""))
            if text:
                messages.append(("user", text))
        elif msg_type == "assistant":
            text = _extract_content(message.get("content", ""))
            if text:
                messages.append(("assistant", text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_codex_jsonl(content: str) -> Optional[str]:
    """OpenAI Codex CLI sessions (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl).

    Uses only event_msg entries (user_message / agent_message) which represent
    the canonical conversation turns. response_item entries are skipped because
    they include synthetic context injections and duplicate the real messages.
    """
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages = []
    has_session_meta = False
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry_type == "session_meta":
            has_session_meta = True
            continue

        if entry_type != "event_msg":
            continue

        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type", "")
        msg = payload.get("message")
        if not isinstance(msg, str):
            continue
        text = msg.strip()
        if not text:
            continue

        if payload_type == "user_message":
            messages.append(("user", text))
        elif payload_type == "agent_message":
            messages.append(("assistant", text))

    if len(messages) >= 2 and has_session_meta:
        return _messages_to_transcript(messages)
    return None


def _try_claude_ai_json(data) -> Optional[str]:
    """Claude.ai JSON export: flat messages list or privacy export with chat_messages."""
    if isinstance(data, dict):
        data = data.get("messages", data.get("chat_messages", []))
    if not isinstance(data, list):
        return None

    # Privacy export: array of conversation objects with chat_messages inside each
    if data and isinstance(data[0], dict) and "chat_messages" in data[0]:
        all_messages = []
        for convo in data:
            if not isinstance(convo, dict):
                continue
            chat_msgs = convo.get("chat_messages", [])
            for item in chat_msgs:
                if not isinstance(item, dict):
                    continue
                role = item.get("role", "")
                text = _extract_content(item.get("content", ""))
                if role in ("user", "human") and text:
                    all_messages.append(("user", text))
                elif role in ("assistant", "ai") and text:
                    all_messages.append(("assistant", text))
        if len(all_messages) >= 2:
            return _messages_to_transcript(all_messages)
        return None

    # Flat messages list
    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        text = _extract_content(item.get("content", ""))
        if role in ("user", "human") and text:
            messages.append(("user", text))
        elif role in ("assistant", "ai") and text:
            messages.append(("assistant", text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_chatgpt_json(data) -> Optional[str]:
    """ChatGPT conversations.json with mapping tree."""
    if not isinstance(data, dict) or "mapping" not in data:
        return None
    mapping = data["mapping"]
    messages = []
    # Find root: prefer node with parent=None AND no message (synthetic root)
    root_id = None
    fallback_root = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            if node.get("message") is None:
                root_id = node_id
                break
            elif fallback_root is None:
                fallback_root = node_id
    if not root_id:
        root_id = fallback_root
    if root_id:
        current_id = root_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message")
            if msg:
                role = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = " ".join(str(p) for p in parts if isinstance(p, str) and p).strip()
                if role == "user" and text:
                    messages.append(("user", text))
                elif role == "assistant" and text:
                    messages.append(("assistant", text))
            children = node.get("children", [])
            current_id = children[0] if children else None
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_slack_json(data) -> Optional[str]:
    """
    Slack channel export: [{"type": "message", "user": "...", "text": "..."}]
    Optimized for 2-person DMs. In channels with 3+ people, alternating
    speakers are labeled user/assistant to preserve the exchange structure.
    """
    if not isinstance(data, list):
        return None
    messages = []
    seen_users = {}
    last_role = None
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        user_id = item.get("user", item.get("username", ""))
        text = item.get("text", "").strip()
        if not text or not user_id:
            continue
        if user_id not in seen_users:
            # Alternate roles so exchange chunking works with any number of speakers
            if not seen_users:
                seen_users[user_id] = "user"
            elif last_role == "user":
                seen_users[user_id] = "assistant"
            else:
                seen_users[user_id] = "user"
        last_role = seen_users[user_id]
        messages.append((seen_users[user_id], text))
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _extract_content(content) -> str:
    """Pull text from content — handles str, list of blocks, or dict."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return content.get("text", "").strip()
    return ""


def _messages_to_transcript(messages: list, spellcheck: bool = True) -> str:
    """Convert [(role, text), ...] to transcript format with > markers."""
    if spellcheck:
        try:
            from mempalace.spellcheck import spellcheck_user_text

            _fix = spellcheck_user_text
        except ImportError:
            _fix = None
    else:
        _fix = None

    lines = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            if _fix is not None:
                text = _fix(text)
            lines.append(f"> {text}")
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                lines.append(messages[i + 1][1])
                i += 2
            else:
                i += 1
        else:
            lines.append(text)
            i += 1
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python normalize.py <filepath>")
        sys.exit(1)
    filepath = sys.argv[1]
    result = normalize(filepath)
    quote_count = sum(1 for line in result.split("\n") if line.strip().startswith(">"))
    print(f"\nFile: {os.path.basename(filepath)}")
    print(f"Normalized: {len(result)} chars | {quote_count} user turns detected")
    print("\n--- Preview (first 20 lines) ---")
    print("\n".join(result.split("\n")[:20]))
