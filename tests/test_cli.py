"""
Tests for the pure-function pieces of mempalace.cli.

Scoped narrowly to avoid touching the database or filesystem-heavy
subcommands — this file only exercises output-only commands like
`mempalace mcp`.
"""

import argparse

from mempalace.cli import cmd_mcp


def test_cmd_mcp_prints_docker_compose_setup(capsys):
    """`mempalace mcp` prints the Docker Compose invocation for this fork."""
    args = argparse.Namespace(palace=None)
    cmd_mcp(args)

    out = capsys.readouterr().out

    # Docker Compose invocation is the primary fork delivery model
    assert "docker compose run" in out
    assert "--entrypoint python mempalace -m mempalace.mcp_server" in out
    # Claude Code hookup line
    assert "claude mcp add mempalace" in out
    # No palace flag supplied → "Optional custom palace" section should appear
    assert "Optional custom palace" in out


def test_cmd_mcp_honours_palace_flag(capsys):
    """When --palace is set, the printed commands embed the resolved path."""
    args = argparse.Namespace(palace="/tmp/my_palace")
    cmd_mcp(args)

    out = capsys.readouterr().out

    # The resolved palace path should appear after --palace in the command
    assert "--palace /tmp/my_palace" in out
    # When an explicit palace is given, the "Optional custom palace" section
    # is intentionally suppressed (user already picked one)
    assert "Optional custom palace" not in out


def test_cmd_mcp_quotes_palace_path_with_spaces(capsys):
    """Paths with spaces are shell-quoted via shlex.quote."""
    args = argparse.Namespace(palace="/tmp/palace with spaces")
    cmd_mcp(args)

    out = capsys.readouterr().out
    # shlex.quote wraps strings containing spaces in single quotes
    assert "'/tmp/palace with spaces'" in out


# --- Subagent transcripts: mined by default, --exclude-subagents opts out ---
# (upstream 5f425ba4 / PR #1217 defaults to skipping them; this fork keeps them)


def _parse_mine(monkeypatch, *argv):
    """Run main() with a recording cmd_mine and return the parsed Namespace."""
    from mempalace import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_mine", seen.append)
    monkeypatch.setattr("sys.argv", ["mempalace", "mine", "/some/dir", *argv])
    cli.main()
    return seen[0]


def test_mine_subagents_included_by_default(monkeypatch):
    assert _parse_mine(monkeypatch, "--mode", "convos").include_subagents is True


def test_mine_exclude_subagents_flag_parses(monkeypatch):
    args = _parse_mine(monkeypatch, "--mode", "convos", "--exclude-subagents")
    assert args.include_subagents is False


def test_cmd_mine_passes_include_subagents_to_convo_miner(monkeypatch):
    from mempalace import cli, convo_miner

    calls = []
    monkeypatch.setattr(convo_miner, "mine_convos", lambda **kw: calls.append(kw))
    for flag in (False, True):
        args = argparse.Namespace(
            palace="/tmp/p",
            include_ignored=[],
            mode="convos",
            dir="/some/dir",
            wing=None,
            agent="mempalace",
            limit=0,
            dry_run=True,
            extract="exchange",
            include_subagents=flag,
        )
        cli.cmd_mine(args)
    assert [c["include_subagents"] for c in calls] == [False, True]
