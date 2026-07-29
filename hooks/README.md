# MemPalace Hooks — Auto-Save for Terminal AI Tools

These hook scripts make MemPalace save automatically. No manual "save" commands needed.

## What They Do

| Hook | When It Fires | What Happens |
|------|--------------|-------------|
| **Save Hook** | Every 15 human messages | Blocks the AI, tells it to save key topics/decisions/quotes to the palace |
| **PreCompact Hook** | Right before context compaction | Emergency save — forces the AI to save EVERYTHING before losing context |
| **SessionEnd Hook** | On clean session exit | Final background mine of the transcript, so short sessions under the Stop interval are still captured |

The AI does the actual filing — it knows the conversation context, so it classifies memories into the right wings/halls/closets. The hooks just tell it WHEN to save.

All three hooks mine through Docker via `mine-wrapper.sh` at the repo root: the host python does not need any of the mining dependencies (psycopg2, sentence-transformers) — the wrapper reuses the same image as the MCP server, translates host paths to container mounts, and serializes concurrent mines with a non-blocking flock. SessionEnd cannot block or message the AI (and Claude Code budgets it at ~1.5s), so it only detaches the CLI mine and returns immediately (adapted from upstream `d09392c`, #1341).

## Install — Claude Code

Add to `~/.claude/settings.json` to cover **every** project (recommended — the palace is global, so project-scoped hooks silently skip all your other work), or to a project's `.claude/settings.local.json` to scope it down:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "/absolute/path/to/hooks/mempal_save_hook.sh",
        "timeout": 30
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "/absolute/path/to/hooks/mempal_precompact_hook.sh",
        "timeout": 30
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "/absolute/path/to/hooks/mempal_sessionend_hook.sh",
        "timeout": 10
      }]
    }]
  }
}
```

Do not register the same hooks in both the global and a project settings file — Claude Code merges hook sources, so they would fire twice.

Make them executable:
```bash
chmod +x hooks/*.sh mine-wrapper.sh
```

## Install — Codex CLI (OpenAI)

Add to `.codex/hooks.json`:

```json
{
  "Stop": [{
    "type": "command",
    "command": "/absolute/path/to/hooks/mempal_save_hook.sh",
    "timeout": 30
  }],
  "PreCompact": [{
    "type": "command",
    "command": "/absolute/path/to/hooks/mempal_precompact_hook.sh",
    "timeout": 30
  }]
}
```

## Configuration

Edit `mempal_save_hook.sh` to change:

- **`SAVE_INTERVAL=15`** — How many human messages between saves. Lower = more frequent saves, higher = less interruption.
- **`STATE_DIR`** — Where hook state is stored (defaults to `~/.mempalace/hook_state/`)
- **`MEMPAL_DIR`** — Optional **project directory** (code, notes, docs) to also mine on each save trigger, with `--mode projects`. The hook ALWAYS mines the active conversation transcript automatically with `--mode convos` — `MEMPAL_DIR` is purely additive, never an override. Leave blank if you don't want to ingest project files.

### mempalace CLI

The relevant commands are:

```bash
mempalace mine <dir>               # Mine all files in a directory
mempalace mine <dir> --mode convos # Mine conversation transcripts only
```

The hooks resolve the repo root (and thus `mine-wrapper.sh`) automatically from their own path, so they work regardless of where you install the repo. Set `MEMPAL_MINE_CMD` to substitute another miner command (the test suite uses this to stub the Docker layer).

## How It Works (Technical)

### Save Hook (Stop event)

```
User sends message → AI responds → Claude Code fires Stop hook
                                            ↓
                                    Hook counts human messages in JSONL transcript
                                            ↓
                              ┌─── < 15 since last save ──→ echo "{}" (let AI stop)
                              │
                              └─── ≥ 15 since last save ──→ {"decision": "block", "reason": "save..."}
                                                                    ↓
                                                            AI saves to palace
                                                                    ↓
                                                            AI tries to stop again
                                                                    ↓
                                                            stop_hook_active = true
                                                                    ↓
                                                            Hook sees flag → echo "{}" (let it through)
```

The `stop_hook_active` flag prevents infinite loops: block once → AI saves → tries to stop → flag is true → we let it through.

### PreCompact Hook

```
Context window getting full → Claude Code fires PreCompact
                                        ↓
                                Hook ALWAYS blocks
                                        ↓
                                AI saves everything
                                        ↓
                                Compaction proceeds
```

No counting needed — compaction always warrants a save.

### SessionEnd Hook

```
Session exits cleanly → Claude Code fires SessionEnd (~1.5s budget)
                                ↓
                        Hook validates transcript path
                                ↓
                        nohup mine-wrapper.sh … & disown  (detached)
                                ↓
                        echo "{}" — returns immediately
                                ↓
                        Child finishes the mine after the session is gone
```

The mine is idempotent (mtime-aware `file_already_mined`) and the wrapper's flock skips a run if a Stop-hook mine is still in flight, so racing triggers are harmless.

## Debugging

Check the hook log:
```bash
cat ~/.mempalace/hook_state/hook.log
```

Example output:
```
[14:30:15] Session abc123: 12 exchanges, 12 since last save
[14:35:22] Session abc123: 15 exchanges, 15 since last save
[14:35:22] TRIGGERING SAVE at exchange 15
[14:40:01] Session abc123: 18 exchanges, 3 since last save
```

## Cost

**Zero extra tokens.** The hooks are bash scripts that run locally. They don't call any API. The only "cost" is the AI spending a few seconds organizing memories at each checkpoint — and it's doing that with context it already has loaded.
