#!/bin/bash
# MEMPALACE SESSION-END HOOK — Final flush on clean exit
#
# Claude Code "SessionEnd" hook. Short sessions that exit cleanly below
# SAVE_INTERVAL and without a PreCompact were never saved — this hook
# takes one final mine of the conversation transcript when the session
# ends, so nothing under the Stop-hook threshold is lost.
#
# Adapted from upstream MemPalace d09392c (#1341 by @mvalentsev): Claude
# Code budgets SessionEnd hooks tightly (documented default 1.5s) and a
# cold mine far exceeds that, so the work MUST NOT run in the foreground —
# it would be killed before saving anything. The hook validates the
# transcript path, detaches the mine (nohup + disown), and returns
# immediately; the child finishes after the session has exited. Unlike
# Stop/PreCompact, SessionEnd cannot block or message the AI, so there is
# no diary prompt here — only the CLI mine. The upstream hooks_cli.py
# implementation does not apply; this is the bash-only port, mining
# through Docker via mine-wrapper.sh like the other two hooks.
#
# === INSTALL ===
# Add to ~/.claude/settings.json (global) or .claude/settings.local.json:
#
#   "hooks": {
#     "SessionEnd": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/absolute/path/to/mempal_sessionend_hook.sh",
#         "timeout": 10
#       }]
#     }]
#   }

STATE_DIR="$HOME/.mempalace/hook_state"
mkdir -p "$STATE_DIR"

# Miner command. Resolves to the repo's docker wrapper next to hooks/;
# MEMPAL_MINE_CMD overrides it (used by tests to stub the docker layer).
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINE_CMD="${MEMPAL_MINE_CMD:-$HOOK_DIR/../mine-wrapper.sh}"

# Validate transcript path — same shape check as mempal_save_hook.sh.
is_valid_transcript_path() {
    local path="$1"
    [ -n "$path" ] || return 1
    case "$path" in
        *.json|*.jsonl) ;;
        *) return 1 ;;
    esac
    case "/$path/" in
        */../*) return 1 ;;
    esac
    return 0
}

# Read JSON input from stdin
INPUT=$(cat)

SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','unknown'))" 2>/dev/null)
SESSION_ID=$(echo "$SESSION_ID" | tr -cd 'a-zA-Z0-9_-')
[ -z "$SESSION_ID" ] && SESSION_ID="unknown"
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null)
TRANSCRIPT_PATH="${TRANSCRIPT_PATH/#\~/$HOME}"

echo "[$(date '+%H:%M:%S')] SESSION-END for session $SESSION_ID" >> "$STATE_DIR/hook.log"

# Detach the final mine so we return within the SessionEnd budget. The
# mine-wrapper's flock makes a race with a still-running Stop-hook mine
# harmless (the loser is skipped; the miner is idempotent anyway).
if is_valid_transcript_path "$TRANSCRIPT_PATH" && [ -f "$TRANSCRIPT_PATH" ]; then
    nohup "$MINE_CMD" "$(dirname "$TRANSCRIPT_PATH")" --mode convos \
        >> "$STATE_DIR/hook.log" 2>&1 &
    disown
elif [ -n "$TRANSCRIPT_PATH" ]; then
    echo "[$(date '+%H:%M:%S')] Skipping invalid transcript path: $TRANSCRIPT_PATH" \
        >> "$STATE_DIR/hook.log"
fi

echo "{}"
exit 0
