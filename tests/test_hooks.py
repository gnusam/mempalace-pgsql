"""
test_hooks.py — Tests for the bash hooks and the docker mine wrapper.

The hooks honor MEMPAL_MINE_CMD so tests can substitute a stub that records
its argv instead of spinning up Docker; mine-wrapper.sh itself is tested
with a fake `docker` binary injected via PATH.
"""

import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOKS = REPO / "hooks"


@pytest.fixture()
def hook_env(tmp_path):
    """Isolated HOME + argv-recording mine stub for hook invocations."""
    home = tmp_path / "home"
    home.mkdir()
    calls = tmp_path / "mine_calls.log"
    stub = tmp_path / "mine_stub.sh"
    stub.write_text('#!/bin/bash\necho "$@" >> "%s"\n' % calls)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, HOME=str(home), MEMPAL_MINE_CMD=str(stub))
    return env, home, calls


def _run_hook(script, payload, env):
    return subprocess.run(
        ["bash", str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )


def _wait_for(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def _make_transcript(home, n_user_messages):
    tdir = home / ".claude" / "projects" / "-home-sam-dev-someproj"
    tdir.mkdir(parents=True)
    lines = [
        json.dumps({"message": {"role": "user", "content": f"msg {i}"}})
        for i in range(n_user_messages)
    ]
    transcript = tdir / "abc-123.jsonl"
    transcript.write_text("\n".join(lines) + "\n")
    return transcript


# ── SessionEnd hook (port of upstream d09392c) ───────────────────────────


class TestSessionEndHook:
    def test_mines_transcript_in_background_and_returns_immediately(self, hook_env):
        env, home, calls = hook_env
        transcript = _make_transcript(home, 3)
        start = time.monotonic()
        result = _run_hook(
            HOOKS / "mempal_sessionend_hook.sh",
            {"session_id": "abc-123", "transcript_path": str(transcript)},
            env,
        )
        elapsed = time.monotonic() - start
        assert result.returncode == 0
        assert json.loads(result.stdout.strip()) == {}
        assert elapsed < 5, "SessionEnd must return well within its budget"
        assert _wait_for(calls), "detached mine never ran"
        assert calls.read_text().strip() == f"{transcript.parent} --mode convos"

    def test_rejects_traversal_path(self, hook_env):
        env, home, calls = hook_env
        result = _run_hook(
            HOOKS / "mempal_sessionend_hook.sh",
            {"session_id": "abc-123", "transcript_path": "/tmp/../etc/x.jsonl"},
            env,
        )
        assert result.returncode == 0
        assert json.loads(result.stdout.strip()) == {}
        time.sleep(0.3)
        assert not calls.exists(), "mine must not run on an invalid path"

    def test_rejects_non_jsonl_path(self, hook_env):
        env, home, calls = hook_env
        result = _run_hook(
            HOOKS / "mempal_sessionend_hook.sh",
            {"session_id": "abc-123", "transcript_path": "/etc/passwd"},
            env,
        )
        assert result.returncode == 0
        time.sleep(0.3)
        assert not calls.exists()


# ── Stop hook routes its mine through the wrapper ────────────────────────


class TestSaveHookMineRouting:
    def test_below_interval_no_mine_no_block(self, hook_env):
        env, home, calls = hook_env
        transcript = _make_transcript(home, 3)
        result = _run_hook(
            HOOKS / "mempal_save_hook.sh",
            {"session_id": "abc-123", "transcript_path": str(transcript)},
            env,
        )
        assert json.loads(result.stdout.strip()) == {}
        time.sleep(0.3)
        assert not calls.exists()

    def test_at_interval_blocks_and_mines_via_wrapper(self, hook_env):
        env, home, calls = hook_env
        transcript = _make_transcript(home, 20)
        result = _run_hook(
            HOOKS / "mempal_save_hook.sh",
            {"session_id": "abc-123", "transcript_path": str(transcript)},
            env,
        )
        out = json.loads(result.stdout.strip())
        assert out["decision"] == "block"
        assert 'wing="wing_someproj"' in out["reason"]
        assert _wait_for(calls), "mine never ran at save interval"
        assert calls.read_text().strip() == f"{transcript.parent} --mode convos"


class TestPreCompactHookMineRouting:
    def test_always_blocks_and_mines_synchronously(self, hook_env):
        env, home, calls = hook_env
        transcript = _make_transcript(home, 2)
        result = _run_hook(
            HOOKS / "mempal_precompact_hook.sh",
            {"session_id": "abc-123", "transcript_path": str(transcript)},
            env,
        )
        out = json.loads(result.stdout.strip())
        assert out["decision"] == "block"
        # synchronous: the call is recorded before the hook returns
        assert calls.exists()
        assert calls.read_text().strip() == f"{transcript.parent} --mode convos"


# ── mine-wrapper.sh path translation (docker stubbed via PATH) ───────────


@pytest.fixture()
def wrapper_env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    calls = tmp_path / "docker_calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    # `docker image inspect` must succeed so the wrapper skips the build;
    # every invocation is recorded for assertions.
    fake_docker.write_text('#!/bin/bash\necho "$@" >> "%s"\nexit 0\n' % calls)
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IEXEC)
    env = dict(
        os.environ,
        HOME=str(home),
        PATH=f"{bin_dir}:{os.environ['PATH']}",
    )
    return env, home, calls


def _run_wrapper(args, env):
    return subprocess.run(
        ["bash", str(REPO / "mine-wrapper.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )


class TestMineWrapperPathTranslation:
    def _docker_run_line(self, calls):
        runs = [ln for ln in calls.read_text().splitlines() if ln.startswith("run ")]
        assert len(runs) == 1, "expected exactly one docker run"
        return runs[0]

    def test_transcript_dir_maps_to_transcripts_mount(self, wrapper_env):
        env, home, calls = wrapper_env
        tdir = home / ".claude" / "projects" / "-proj"
        tdir.mkdir(parents=True)
        result = _run_wrapper([str(tdir), "--mode", "convos"], env)
        assert result.returncode == 0, result.stderr
        run = self._docker_run_line(calls)
        assert f"{home}/.claude/projects:/transcripts:ro" in run
        assert "mine /transcripts/-proj --mode convos" in run

    def test_dev_dir_maps_to_projects_mount(self, wrapper_env):
        env, home, calls = wrapper_env
        pdir = home / "dev" / "myapp"
        pdir.mkdir(parents=True)
        result = _run_wrapper([str(pdir), "--mode", "projects"], env)
        assert result.returncode == 0, result.stderr
        run = self._docker_run_line(calls)
        assert f"{home}/dev:/projects:ro" in run
        assert "mine /projects/myapp --mode projects" in run

    def test_other_dir_mounts_itself(self, wrapper_env):
        env, home, calls = wrapper_env
        other = home / "elsewhere"
        other.mkdir()
        result = _run_wrapper([str(other), "--mode", "projects"], env)
        assert result.returncode == 0, result.stderr
        run = self._docker_run_line(calls)
        assert f"{other}:/mine-target:ro" in run
        assert "mine /mine-target --mode projects" in run

    def test_no_gpu_flag_ever(self, wrapper_env):
        env, home, calls = wrapper_env
        other = home / "elsewhere"
        other.mkdir()
        _run_wrapper([str(other), "--mode", "projects"], env)
        assert "--gpus" not in calls.read_text()

    def test_missing_path_arg_errors(self, wrapper_env):
        env, home, calls = wrapper_env
        result = _run_wrapper([], env)
        assert result.returncode == 2
