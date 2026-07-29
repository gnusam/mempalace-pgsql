#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the palace.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to palace.

Same palace as project mining. Different ingest strategy.
"""

import hashlib
import os
import sys
from pathlib import Path
from collections import defaultdict

from .db import get_db

from .normalize import normalize


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".mempalace",
    "tool-results",
    "memory",
}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800  # chars per drawer — align with miner.py

# Hard ceiling on per-file size at scan time. Ported from upstream 1d19dfc
# (PR #252) — protects against OOM on pathological transcript files.
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(content: str) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.
    """
    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines)
    else:
        return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: list) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim. When the combined
    user-turn + response exceeds CHUNK_SIZE the response is split across
    consecutive drawers so nothing is silently discarded.
    """
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines)
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            if len(content) > CHUNK_SIZE:
                first_part = content[:CHUNK_SIZE]
                if len(first_part.strip()) > MIN_CHUNK_SIZE:
                    chunks.append({"content": first_part, "chunk_index": len(chunks)})
                remainder = content[CHUNK_SIZE:]
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    if len(part.strip()) > MIN_CHUNK_SIZE:
                        chunks.append({"content": part, "chunk_index": len(chunks)})
            elif len(content.strip()) > MIN_CHUNK_SIZE:
                chunks.append(
                    {
                        "content": content,
                        "chunk_index": len(chunks),
                    }
                )
        else:
            i += 1

    return chunks


def _chunk_by_paragraph(content: str) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                chunks.append({"content": group, "chunk_index": len(chunks)})
        return chunks

    for para in paragraphs:
        if len(para) > MIN_CHUNK_SIZE:
            chunks.append({"content": para, "chunk_index": len(chunks)})

    return chunks


# =============================================================================
# ROOM DETECTION — topic-based for conversations
# =============================================================================

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def detect_convo_room(content: str) -> str:
    """Score conversation content against topic keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for room, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[room] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# =============================================================================
# PALACE OPERATIONS
# =============================================================================


def get_db_instance(palace_path: str = None):
    return get_db()


def file_already_mined(db, source_file: str, extract_mode: str = "exchange") -> bool:
    # Scoped to this pass's extract mode so exchange-mode and general-mode
    # drawers for the same transcript track their own freshness (see
    # PalaceDB._scope_clause, upstream PR #2089's over-match class).
    return db.file_already_mined(source_file, ingest_mode="convos", extract_mode=extract_mode)


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str) -> list:
    """Find all potential conversation files."""
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files. Ported from upstream
                # 1d19dfc (PR #252).
                if filepath.is_symlink():
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def mine_convos(
    convo_dir: str,
    palace_path: str,
    wing: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
):
    """Mine a directory of conversation files into the palace.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions
    """

    convo_path = Path(convo_dir).expanduser().resolve()
    if not wing:
        wing = convo_path.name.lower().replace(" ", "_").replace("-", "_")

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine — Conversations")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print("  Palace:  PostgreSQL")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'-' * 55}\n")

    db = get_db_instance(palace_path) if not dry_run else None

    total_drawers = 0
    files_skipped = 0
    room_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        source_file = str(filepath)

        # Skip if already filed
        if not dry_run and file_already_mined(db, source_file, extract_mode):
            files_skipped += 1
            continue

        # Capture the mtime BEFORE normalize() reads the file, so the stored
        # value is paired with the content actually chunked. Claude Code
        # session logs are appended to while being mined — a later re-stat
        # would stamp the drawers as covering the appended tail and the next
        # mine would skip it forever (upstream PR #2088's TOCTOU gap).
        try:
            source_mtime = filepath.stat().st_mtime
        except OSError:
            source_mtime = None

        # Normalize format
        try:
            content = normalize(str(filepath))
        except (OSError, ValueError):
            # Possibly transient failure: register the sentinel so the run
            # doesn't re-read the file forever, but do NOT purge existing
            # drawers — a parse hiccup must not destroy mined data
            # (upstream PR #2088's failed-purge lesson).
            if not dry_run:
                db.register_empty_file(source_file, wing, agent, source_mtime=source_mtime)
            continue

        if not content or len(content.strip()) < MIN_CHUNK_SIZE:
            # Genuinely empty content: this pass's old drawers (if any) no
            # longer have a source — purge them along with registering the
            # sentinel (upstream PR #2089: a rebuild must not leave rows
            # its own pass no longer owns).
            if not dry_run:
                db.register_empty_file(
                    source_file,
                    wing,
                    agent,
                    source_mtime=source_mtime,
                    purge_stale=True,
                    extract_mode=extract_mode,
                )
            continue

        # Content dedup (adapted from upstream PR #2050): the same
        # conversation re-exported under a new filename must not duplicate
        # every drawer under fresh file+chunk slot IDs. Hash the whole
        # normalized transcript; if another file already carries it, skip
        # this one and register the sentinel so it isn't re-checked every
        # run. The file's own rows are excluded, so a renamed-in-place or
        # touch'd file still re-mines under its own name.
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if not dry_run and db.content_hash_exists(content_hash, source_file):
            db.register_empty_file(source_file, wing, agent, source_mtime=source_mtime)
            files_skipped += 1
            print(f"  = [{i:4}/{len(files)}] {filepath.name[:50]:50} duplicate content, skipped")
            continue

        # Chunk — either exchange pairs or general extraction
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(content)
            # Each chunk already has memory_type; use it as the room name
        else:
            chunks = chunk_exchanges(content)

        if not chunks:
            if not dry_run:
                db.register_empty_file(
                    source_file,
                    wing,
                    agent,
                    source_mtime=source_mtime,
                    purge_stale=True,
                    extract_mode=extract_mode,
                )
            continue

        # Detect room from content (general mode uses memory_type instead)
        if extract_mode != "general":
            room = detect_convo_room(content)
        else:
            room = None  # set per-chunk below

        if dry_run:
            if extract_mode == "general":
                from collections import Counter

                type_counts = Counter(c.get("memory_type", "general") for c in chunks)
                types_str = ", ".join(f"{t}:{n}" for t, n in type_counts.most_common())
                print(f"    [DRY RUN] {filepath.name} → {len(chunks)} memories ({types_str})")
            else:
                print(f"    [DRY RUN] {filepath.name} → room:{room} ({len(chunks)} drawers)")
            total_drawers += len(chunks)
            # Track room counts
            if extract_mode == "general":
                for c in chunks:
                    room_counts[c.get("memory_type", "general")] += 1
            else:
                room_counts[room] += 1
            continue

        if extract_mode != "general":
            room_counts[room] += 1

        # File the whole transcript through one atomic replace: stale-row
        # purge (scoped to this extract mode) and every insert commit
        # together, so a mine killed mid-file leaves the previous state
        # intact and re-chunked exchanges don't leave orphaned tail rows
        # (upstream PRs #2088/#2089, adapted — see
        # PalaceDB.replace_file_drawers).
        payload = []
        for chunk in chunks:
            chunk_room = chunk.get("memory_type", room) if extract_mode == "general" else room
            if extract_mode == "general":
                room_counts[chunk_room] += 1
            payload.append(
                {
                    "room": chunk_room,
                    "content": chunk["content"],
                    "chunk_index": chunk["chunk_index"],
                    "metadata": {"file_content_hash": content_hash},
                }
            )
        filed_ids = db.replace_file_drawers(
            wing,
            payload,
            source_file,
            agent=agent,
            source_mtime=source_mtime,
            ingest_mode="convos",
            extract_mode=extract_mode,
        )
        drawers_added = len(filed_ids)

        total_drawers += drawers_added
        print(f"  ✓ [{i:4}/{len(files)}] {filepath.name[:50]:50} +{drawers_added}")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    if room_counts:
        print("\n  By room:")
        for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
