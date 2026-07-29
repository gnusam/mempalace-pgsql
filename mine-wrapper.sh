#!/usr/bin/env bash
# Docker-routed miner for the MemPalace hooks.
#
# The hooks used to run `python3 -m mempalace mine ...` with the host
# interpreter, which has none of the PG-era dependencies (psycopg2,
# sentence-transformers) — every hook-triggered mine died on import and the
# failure only ever landed in hook.log. The only complete runtime is the
# Docker image the MCP server already uses (see mcp-wrapper.sh), so route
# hook mining through the same image.
#
# Usage: mine-wrapper.sh <host-path> --mode convos|projects
#
# Host→container path translation (mounts added per prefix):
#   $HOME/.claude/projects/...  → /transcripts/...   (~/.claude/projects ro)
#   $HOME/dev/...               → /projects/...      (same convention as compose)
#   anything else               → /mine-target       (the dir itself, ro)
#
# Mining runs on CPU (no --gpus): it's an unattended background job and must
# not reserve the dGPU out from under an interactive session. Concurrent
# invocations (Stop + SessionEnd racing, several sessions ending at once)
# serialize through a non-blocking flock — if a mine is already running the
# new one is skipped, which is safe because the miner is idempotent
# (mtime-aware file_already_mined) and the next trigger will catch up.

set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
PROJECT="mempalace"
NETWORK="${PROJECT}_default"
IMAGE="${PROJECT}-${PROJECT}"
DB_URL="postgresql://mempalace:mempalace@postgres:5432/mempalace"
STATE_DIR="$HOME/.mempalace/hook_state"
mkdir -p "$STATE_DIR"

HOST_PATH="${1:-}"
shift || true
if [ -z "$HOST_PATH" ]; then
    echo "usage: mine-wrapper.sh <host-path> --mode convos|projects" >&2
    exit 2
fi

# ── Serialize concurrent mines (skip, don't queue) ──────────────────
exec 9>"$STATE_DIR/mine.lock"
if ! flock -n 9; then
    echo "[$(date '+%H:%M:%S')] mine-wrapper: mine already running, skipping $HOST_PATH" \
        >> "$STATE_DIR/hook.log"
    exit 0
fi

# ── Host → container path translation ───────────────────────────────
extra_mounts=()
case "$HOST_PATH" in
    "$HOME/.claude/projects"*)
        CONTAINER_PATH="/transcripts${HOST_PATH#"$HOME/.claude/projects"}"
        extra_mounts+=(-v "$HOME/.claude/projects:/transcripts:ro")
        ;;
    "$HOME/dev"*)
        CONTAINER_PATH="/projects${HOST_PATH#"$HOME/dev"}"
        extra_mounts+=(-v "$HOME/dev:/projects:ro")
        ;;
    *)
        CONTAINER_PATH="/mine-target"
        extra_mounts+=(-v "$HOST_PATH:/mine-target:ro")
        ;;
esac

# ── Ensure postgres is up and the image exists ──────────────────────
docker compose -f "$COMPOSE_FILE" up -d postgres >/dev/null 2>&1
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" build mempalace >/dev/null 2>&1
fi

exec docker run --rm -i \
    --name "mempalace-mine-$$" \
    --network "$NETWORK" \
    -e "DATABASE_URL=$DB_URL" \
    -e "PYTHONPATH=/app" \
    -e "MEMPALACE_DEVICE=cpu" \
    -v "${PROJECT}_model-cache:/root/.cache" \
    -v "$COMPOSE_DIR/mempalace:/app/mempalace:ro" \
    "${extra_mounts[@]}" \
    --entrypoint python \
    "$IMAGE" -m mempalace mine "$CONTAINER_PATH" "$@"
