#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Semantic search against the palace (PostgreSQL + pgvector).
Returns verbatim text — the actual words, never summaries.
"""

import logging
from pathlib import Path

from .db import get_db

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. query failure)."""


def search(
    query: str, palace_path: str = None, wing: str = None, room: str = None, n_results: int = 5
):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    db = get_db()

    # Leave where=None when no explicit filter is given so db.query() runs
    # its auto-detect path (catches "datahub" → wing=datahub, etc). Passing
    # an empty dict here silently disables auto-detect because db.query
    # tests `if where is None`.
    where = None
    if wing and room:
        where = {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        where = {"wing": wing}
    elif room:
        where = {"room": room}

    try:
        results = db.query(query, n_results=n_results, where=where)
    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = round(max(0.0, 1 - dist), 3)
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {similarity}")
        print()
        for line in doc.strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(
    query: str,
    palace_path: str = None,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    since: str = None,
    before: str = None,
) -> dict:
    """
    Programmatic search — returns a dict instead of printing.
    Used by the MCP server and other callers that need data.

    ``since`` / ``before``: optional ``YYYY-MM-DD`` bounds on filed_at
    (since inclusive, before exclusive — see PalaceDB.query, upstream
    PR #2000).
    """
    db = get_db()

    # Leave where=None when no explicit filter is given so db.query() runs
    # its auto-detect path (catches "datahub" → wing=datahub, etc). Passing
    # an empty dict here silently disables auto-detect because db.query
    # tests `if where is None`.
    where = None
    if wing and room:
        where = {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        where = {"wing": wing}
    elif room:
        where = {"room": room}

    try:
        results = db.query(query, n_results=n_results, where=where, since=since, before=before)
    except Exception as e:
        logger.error("Search error: %s", e)
        return {
            "error": "Search failed",
            "hint": "Check that PostgreSQL is running and the palace has been mined.",
        }

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for drawer_id, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                # Round-trippable ID (adapted from upstream PR #2090): lets
                # a search hit feed mempalace_delete_drawer / dedup flows
                # directly. Previously hits carried no ID at all.
                "drawer_id": drawer_id,
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(max(0.0, 1 - dist), 3),
            }
        )

    return {
        "query": query,
        "filters": {"wing": wing, "room": room, "since": since, "before": before},
        "results": hits,
    }
