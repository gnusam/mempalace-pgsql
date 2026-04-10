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
