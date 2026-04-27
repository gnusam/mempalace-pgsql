"""
palace_graph.py — Graph traversal layer for MemPalace
======================================================

Builds a navigable graph from the palace structure:
  - Nodes = rooms (named ideas)
  - Edges = shared rooms across wings (tunnels)
  - Edge types = halls (the corridors)

No external graph DB needed — built from PostgreSQL metadata.
"""

# PEP 604 (``str | None``) needs 3.10+ at runtime; CI still runs 3.9, so
# defer annotation evaluation to keep the union syntax working there.
from __future__ import annotations

import logging
from collections import defaultdict, Counter
from .db import get_db

logger = logging.getLogger("mempalace_graph")


def _normalize_wing(wing: str | None) -> str | None:
    """Normalize a wing name for consistent lookup.

    Wings are stored with spaces and hyphens replaced by underscores
    (see ``room_detector_local.derive_wing_name`` and
    ``convo_miner._derive_wing``). Callers that pass the raw directory
    name (``mempalace-public``) would silently miss against the
    canonical form (``mempalace_public``); this helper aligns the
    lookup key with the stored metadata.
    """
    if wing is None:
        return None
    return wing.lower().replace(" ", "_").replace("-", "_")


def build_graph():
    """
    Build the palace graph from drawer metadata.

    Returns:
        nodes: dict of {room: {wings: set, halls: set, count: int}}
        edges: list of {room, wing_a, wing_b, hall} — one per tunnel crossing
    """
    db = get_db()
    room_data = defaultdict(lambda: {"wings": set(), "halls": set(), "count": 0, "dates": set()})

    offset = 0
    while True:
        batch = db.get_drawers(limit=1000, offset=offset)
        if not batch["ids"]:
            break
        for meta in batch["metadatas"]:
            room = meta.get("room", "")
            wing = meta.get("wing", "")
            hall = meta.get("hall", "")
            date_val = meta.get("date", "")
            if room and room != "general" and wing:
                room_data[room]["wings"].add(wing)
                if hall:
                    room_data[room]["halls"].add(hall)
                if date_val:
                    room_data[room]["dates"].add(date_val)
                room_data[room]["count"] += 1
        offset += len(batch["ids"])

    edges = []
    for room, data in room_data.items():
        wings = sorted(data["wings"])
        if len(wings) >= 2:
            for i, wa in enumerate(wings):
                for wb in wings[i + 1 :]:
                    for hall in data["halls"]:
                        edges.append(
                            {
                                "room": room,
                                "wing_a": wa,
                                "wing_b": wb,
                                "hall": hall,
                                "count": data["count"],
                            }
                        )

    nodes = {}
    for room, data in room_data.items():
        nodes[room] = {
            "wings": sorted(data["wings"]),
            "halls": sorted(data["halls"]),
            "count": data["count"],
            "dates": sorted(data["dates"])[-5:] if data["dates"] else [],
        }

    return nodes, edges


def traverse(start_room: str, max_hops: int = 2):
    nodes, edges = build_graph()

    if start_room not in nodes:
        return {
            "error": f"Room '{start_room}' not found",
            "suggestions": _fuzzy_match(start_room, nodes),
        }

    start = nodes[start_room]
    visited = {start_room}
    results = [
        {
            "room": start_room,
            "wings": start["wings"],
            "halls": start["halls"],
            "count": start["count"],
            "hop": 0,
        }
    ]

    frontier = [(start_room, 0)]
    while frontier:
        current_room, depth = frontier.pop(0)
        if depth >= max_hops:
            continue

        current = nodes.get(current_room, {})
        current_wings = set(current.get("wings", []))

        for room, data in nodes.items():
            if room in visited:
                continue
            shared_wings = current_wings & set(data["wings"])
            if shared_wings:
                visited.add(room)
                results.append(
                    {
                        "room": room,
                        "wings": data["wings"],
                        "halls": data["halls"],
                        "count": data["count"],
                        "hop": depth + 1,
                        "connected_via": sorted(shared_wings),
                    }
                )
                if depth + 1 < max_hops:
                    frontier.append((room, depth + 1))

    results.sort(key=lambda x: (x["hop"], -x["count"]))
    return results[:50]


def find_tunnels(wing_a: str = None, wing_b: str = None):
    nodes, edges = build_graph()

    norm_a = _normalize_wing(wing_a)
    norm_b = _normalize_wing(wing_b)

    tunnels = []
    for room, data in nodes.items():
        wings = data["wings"]
        if len(wings) < 2:
            continue
        if norm_a and norm_a not in wings:
            continue
        if norm_b and norm_b not in wings:
            continue
        tunnels.append(
            {
                "room": room,
                "wings": wings,
                "halls": data["halls"],
                "count": data["count"],
                "recent": data["dates"][-1] if data["dates"] else "",
            }
        )

    if not tunnels and (wing_a or wing_b):
        logger.warning(
            "No tunnels found for wing filter(s): wing_a=%r (normalized=%r), wing_b=%r (normalized=%r)",
            wing_a,
            norm_a,
            wing_b,
            norm_b,
        )

    tunnels.sort(key=lambda x: -x["count"])
    return tunnels[:50]


def graph_stats():
    nodes, edges = build_graph()

    tunnel_rooms = sum(1 for n in nodes.values() if len(n["wings"]) >= 2)
    wing_counts = Counter()
    for data in nodes.values():
        for w in data["wings"]:
            wing_counts[w] += 1

    return {
        "total_rooms": len(nodes),
        "tunnel_rooms": tunnel_rooms,
        "total_edges": len(edges),
        "rooms_per_wing": dict(wing_counts.most_common()),
        "top_tunnels": [
            {"room": r, "wings": d["wings"], "count": d["count"]}
            for r, d in sorted(nodes.items(), key=lambda x: -len(x[1]["wings"]))[:10]
            if len(d["wings"]) >= 2
        ],
    }


def _fuzzy_match(query: str, nodes: dict, n: int = 5):
    query_lower = query.lower()
    scored = []
    for room in nodes:
        if query_lower in room:
            scored.append((room, 1.0))
        elif any(word in room for word in query_lower.split("-")):
            scored.append((room, 0.5))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:n]]
