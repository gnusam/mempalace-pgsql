"""
Tests for wing-name normalization in palace_graph.find_tunnels.

Wings are stored on disk with hyphens/spaces collapsed to underscores
(see room_detector_local.derive_wing_name and convo_miner._derive_wing).
Without lookup-time normalization, callers passing the raw directory
name ("mempalace-public") would silently miss against the canonical
form ("mempalace_public") in metadata.

Adapted from upstream MemPalace tests/test_palace_graph_tunnels.py
(commit 3474641, PR #1194 by @wahajahmed010) — the fork's palace_graph
only exposes find_tunnels (no JSON tunnel store, no create/list/follow
helpers), so the upstream tests for those code paths don't apply.
"""

import logging

from mempalace import palace_graph


def _seed(metadatas):
    """Build a fake DB whose get_drawers returns one batch of given metas."""

    class _FakeDB:
        def __init__(self):
            self._served = False

        def get_drawers(self, limit=None, offset=0):
            if self._served:
                return {"ids": [], "metadatas": []}
            self._served = True
            ids = [f"drawer_{i}" for i in range(len(metadatas))]
            return {"ids": ids, "metadatas": list(metadatas)}

    return _FakeDB()


def test_normalize_wing_helper():
    assert palace_graph._normalize_wing(None) is None
    assert palace_graph._normalize_wing("My-Project") == "my_project"
    assert palace_graph._normalize_wing("Some Wing") == "some_wing"
    assert palace_graph._normalize_wing("wing_code") == "wing_code"


def test_find_tunnels_matches_hyphenated_wing(monkeypatch):
    metas = [
        {"room": "auth", "wing": "mempalace_public", "hall": "code", "date": "2026-04-26"},
        {"room": "auth", "wing": "wing_people", "hall": "code", "date": "2026-04-26"},
    ]
    monkeypatch.setattr(palace_graph, "get_db", lambda: _seed(metas))

    by_hyphen = palace_graph.find_tunnels(wing_a="mempalace-public")
    by_under = palace_graph.find_tunnels(wing_a="mempalace_public")

    assert len(by_hyphen) == 1
    assert len(by_under) == 1
    assert by_hyphen[0]["room"] == "auth"


def test_find_tunnels_normalizes_both_filters(monkeypatch):
    metas = [
        {"room": "auth", "wing": "my_project", "hall": "code", "date": "2026-04-26"},
        {"room": "auth", "wing": "your_project", "hall": "code", "date": "2026-04-26"},
    ]
    monkeypatch.setattr(palace_graph, "get_db", lambda: _seed(metas))

    result = palace_graph.find_tunnels(wing_a="My-Project", wing_b="Your Project")
    assert len(result) == 1
    assert result[0]["room"] == "auth"


def test_find_tunnels_warns_on_empty_result(monkeypatch, caplog):
    monkeypatch.setattr(palace_graph, "get_db", lambda: _seed([]))
    with caplog.at_level(logging.WARNING, logger="mempalace_graph"):
        result = palace_graph.find_tunnels(wing_a="nonexistent-wing")
    assert result == []
    assert "No tunnels found" in caplog.text


def test_find_tunnels_no_filter_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(palace_graph, "get_db", lambda: _seed([]))
    with caplog.at_level(logging.WARNING, logger="mempalace_graph"):
        palace_graph.find_tunnels()
    assert "No tunnels found" not in caplog.text
