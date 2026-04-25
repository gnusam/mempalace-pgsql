"""
Tests for the pure-function protocol layer of mempalace.mcp_server.

Exercises the handle_request() dispatch without touching the database or
the individual tool handlers — the focus is on the JSON-RPC plumbing:
initialize, tools/list, and the tools/call argument-parsing path.

A previous test_mcp_server.py was intentionally dropped at the PG
migration (commit abbab4f) because it depended on ChromaDB / legacy KG
APIs. This file is scoped narrower: no DB, no tool execution.
"""

import pytest

from mempalace import mcp_server
from mempalace.mcp_server import SUPPORTED_PROTOCOL_VERSIONS, handle_request


# ── initialize: protocol version negotiation (ported from upstream 950d52b)


def test_initialize_echoes_supported_client_version():
    """When the client asks for a version we support, echo it back."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
    }
    response = handle_request(request)
    assert response["result"]["protocolVersion"] == "2025-11-25"


def test_initialize_falls_back_to_newest_on_unknown_client_version():
    """Unknown client versions fall back to the newest version we know."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "initialize",
        "params": {"protocolVersion": "1999-01-01", "capabilities": {}},
    }
    response = handle_request(request)
    # Newest is the first entry in the supported list
    assert response["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]


def test_initialize_defaults_when_client_omits_version():
    """No protocolVersion in params → default to the oldest supported version,
    so legacy clients that don't negotiate still get a sensible response."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "initialize",
        "params": {},
    }
    response = handle_request(request)
    # When the key is absent, the implementation uses the oldest (index -1)
    # as the fallback default, which IS in the supported set so it gets
    # echoed back directly.
    assert response["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]


def test_initialize_reports_server_info():
    """The initialize response advertises the server name and version."""
    request = {"jsonrpc": "2.0", "id": 4, "method": "initialize", "params": {}}
    response = handle_request(request)
    info = response["result"]["serverInfo"]
    assert info["name"] == "mempalace"
    assert isinstance(info["version"], str)


# ── tools/call: null-arguments safety (ported from upstream 0720fb8)


def test_tools_call_accepts_null_arguments(monkeypatch):
    """`"arguments": null` must not crash the dispatcher.

    Replace a real tool with a dummy handler so we can observe the
    dispatch without touching the DB. The point of this test is to
    exercise the `params.get("arguments") or {}` path — before the fix,
    the .items() iteration on None would crash inside the dispatcher.
    """
    calls = []

    def fake_handler(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setitem(
        mcp_server.TOOLS,
        "mempalace_status",
        {
            "description": "stub",
            "input_schema": {"type": "object", "properties": {}},
            "handler": fake_handler,
        },
    )

    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "mempalace_status", "arguments": None},
    }
    response = handle_request(request)

    assert "error" not in response
    assert calls == [{}]  # handler received an empty kwargs dict, not None


def test_tools_call_missing_arguments_key_still_works(monkeypatch):
    """Absent `arguments` key is equivalent to an empty dict (always was)."""
    calls = []

    def fake_handler(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setitem(
        mcp_server.TOOLS,
        "mempalace_status",
        {
            "description": "stub",
            "input_schema": {"type": "object", "properties": {}},
            "handler": fake_handler,
        },
    )

    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "mempalace_status"},
    }
    response = handle_request(request)

    assert "error" not in response
    assert calls == [{}]


# ── _normalize_optional_filter: empty/whitespace = no filter (upstream #1097
#    pattern, factored out across every LLM-callable tool)


def test_normalize_optional_filter_returns_none_for_empty_strings():
    from mempalace.mcp_server import _normalize_optional_filter

    assert _normalize_optional_filter("") is None
    assert _normalize_optional_filter("   ") is None
    assert _normalize_optional_filter("\t\n") is None


def test_normalize_optional_filter_strips_then_returns_value():
    from mempalace.mcp_server import _normalize_optional_filter

    assert _normalize_optional_filter("wing_code") == "wing_code"
    assert _normalize_optional_filter("  wing_code  ") == "wing_code"


def test_normalize_optional_filter_passes_none_through():
    from mempalace.mcp_server import _normalize_optional_filter

    assert _normalize_optional_filter(None) is None


# ── tool_search: empty/whitespace wing/room treated as no filter (upstream #1097)


def test_tool_search_empty_wing_room_means_no_filter(monkeypatch):
    """LLMs often fill optional params with "" instead of omitting them.

    Empty or whitespace-only wing/room must be normalized to None before
    reaching search_memories — otherwise the search gets silently scoped
    to a non-existent empty filter and returns 0 results.
    """
    captured = {}

    def fake_search_memories(query, palace_path=None, wing=None, room=None, n_results=5):
        captured["wing"] = wing
        captured["room"] = room
        return {"results": []}

    monkeypatch.setattr(mcp_server, "search_memories", fake_search_memories)

    mcp_server.tool_search("hello", wing="", room="   ")
    assert captured == {"wing": None, "room": None}

    mcp_server.tool_search("hello", wing="wing_code", room="")
    assert captured == {"wing": "wing_code", "room": None}


# ── tool_diary_write / tool_diary_read: optional wing param (upstream #659/#1145)


class _FakeDB:
    """Minimal stand-in for PalaceDB capturing add_drawer / get_drawers calls."""

    def __init__(self):
        self.adds = []
        self.queries = []
        self.fake_results = {"ids": [], "documents": [], "metadatas": []}

    def add_drawer(self, **kwargs):
        self.adds.append(kwargs)
        return f"drawer_{len(self.adds)}"

    def get_drawers(self, where=None, limit=None, offset=0, include=None):
        self.queries.append(where)
        return self.fake_results


def test_diary_write_uses_provided_wing(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_write(agent_name="claude", entry="hello", wing="wing_mempalace")
    assert fake.adds[-1]["wing"] == "wing_mempalace"


def test_diary_write_falls_back_when_wing_omitted(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_write(agent_name="Claude", entry="hello")
    assert fake.adds[-1]["wing"] == "wing_claude"


def test_diary_write_falls_back_on_whitespace_wing(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_write(agent_name="Claude", entry="hello", wing="   ")
    assert fake.adds[-1]["wing"] == "wing_claude"


def test_diary_read_filters_by_provided_wing_and_agent(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_read(agent_name="claude", wing="wing_mempalace")
    where = fake.queries[-1]
    conds = where["$and"]
    # wing + room + added_by clauses
    assert {"wing": "wing_mempalace"} in conds
    assert {"room": "diary"} in conds
    assert {"added_by": "claude"} in conds


def test_diary_read_empty_wing_spans_all_wings_for_agent(monkeypatch):
    """Upstream PR #1145 (1fd16da): wing='' must span every wing this agent
    has written diary entries to — no wing filter, only room + agent."""
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_read(agent_name="claude", wing="")
    where = fake.queries[-1]
    conds = where["$and"]
    # No wing clause when wing is empty
    assert all("wing" not in c for c in conds)
    assert {"room": "diary"} in conds
    assert {"added_by": "claude"} in conds


def test_diary_read_whitespace_wing_spans_all_wings_for_agent(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(mcp_server, "_get_db", lambda: fake)
    mcp_server.tool_diary_read(agent_name="claude", wing="   ")
    where = fake.queries[-1]
    conds = where["$and"]
    assert all("wing" not in c for c in conds)


# ── tool_list_rooms / tool_find_tunnels: same empty-filter normalization


def test_tool_list_rooms_empty_wing_lists_all_rooms(monkeypatch):
    """tool_list_rooms with wing="   " must hit the unfiltered branch
    (counts rooms across all wings) instead of running
    `WHERE wing = '   '` and returning nothing."""
    captured = {"sql": None, "params": None}

    class _FakeCursor:
        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

    class _FakeDB:
        def conn(self):
            return _FakeConn()

    monkeypatch.setattr(mcp_server, "_get_db", lambda: _FakeDB())

    result = mcp_server.tool_list_rooms(wing="   ")
    assert result["wing"] == "all"
    assert "WHERE wing" not in captured["sql"]
    assert captured["params"] is None


def test_tool_list_rooms_with_real_wing_filters(monkeypatch):
    captured = {"sql": None, "params": None}

    class _FakeCursor:
        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

    class _FakeDB:
        def conn(self):
            return _FakeConn()

    monkeypatch.setattr(mcp_server, "_get_db", lambda: _FakeDB())

    result = mcp_server.tool_list_rooms(wing="  wing_code  ")
    assert result["wing"] == "wing_code"  # stripped
    assert "WHERE wing" in captured["sql"]
    assert captured["params"] == ("wing_code",)


def test_tool_find_tunnels_normalizes_both_wings(monkeypatch):
    captured = {"args": None}

    def fake_find_tunnels(a, b):
        captured["args"] = (a, b)
        return {"tunnels": []}

    monkeypatch.setattr(mcp_server, "find_tunnels", fake_find_tunnels)

    mcp_server.tool_find_tunnels(wing_a="", wing_b="   ")
    assert captured["args"] == (None, None)

    mcp_server.tool_find_tunnels(wing_a=" wing_code ", wing_b="wing_user")
    assert captured["args"] == ("wing_code", "wing_user")


def test_tools_call_unknown_tool_returns_error():
    """Unknown tool name yields a JSON-RPC error, not a crash."""
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "definitely_not_a_tool", "arguments": {}},
    }
    response = handle_request(request)
    assert "error" in response
    assert "Unknown tool" in response["error"]["message"]


def test_tools_list_returns_registered_tools():
    """tools/list enumerates every entry in TOOLS with name/description/schema."""
    request = {"jsonrpc": "2.0", "id": 8, "method": "tools/list", "params": {}}
    response = handle_request(request)
    tools = response["result"]["tools"]
    names = {t["name"] for t in tools}
    # A couple of well-known tools should be present
    assert "mempalace_status" in names
    assert "mempalace_search" in names
    # Shape check
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
