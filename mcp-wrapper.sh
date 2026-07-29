#!/usr/bin/env bash
# Battery-aware, deduplicated MCP wrapper for mempalace.
#
# Replaces the raw `docker compose run` in ~/.claude.json so that:
#   A. On battery → no GPU reservation, MEMPALACE_DEVICE=cpu
#   B. On AC but another MCP container already holds the GPU → CPU fallback
#   C. /sys mounted read-only so _on_ac_power() works inside the container
#
# Usage (in ~/.claude.json mcpServers):
#   "command": "/home/sam/dev/mempalace/mcp-wrapper.sh"

set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
PROJECT="mempalace"
NETWORK="${PROJECT}_default"
IMAGE="${PROJECT}-mempalace"
DB_URL="postgresql://mempalace:mempalace@postgres:5432/mempalace"

# ── Battery detection (mirrors localrag _shared/power.py) ───────────
on_ac_power() {
    local psys="/sys/class/power_supply"
    [ -d "$psys" ] || return 0          # no sysfs → assume AC (desktop)
    for d in "$psys"/*/; do
        [ -f "$d/type" ] || continue
        if [ "$(cat "$d/type" 2>/dev/null)" = "Mains" ]; then
            [ "$(cat "$d/online" 2>/dev/null)" = "1" ]
            return $?
        fi
    done
    return 0                             # no Mains adapter → assume AC
}

# ── Reap orphaned mempalace MCP containers ──────────────────────────
# Each MCP container is named mempalace-mcp-<owning-claude-PID>. When the
# owning Claude session dies, --rm normally removes it; if it hangs, the
# name still encodes the dead PID so we can force-remove it here. Also
# clears legacy `docker compose run` leftovers. Runs on AC *and* battery.
reap_orphans() {
    # 1. Containers tied to a now-dead Claude session
    for c in $(docker ps -a --filter "name=mempalace-mcp-" --format '{{.Names}}' 2>/dev/null); do
        pid="${c#mempalace-mcp-}"
        # claude runs as the same user → kill -0 is a reliable liveness probe
        if ! kill -0 "$pid" 2>/dev/null; then
            docker rm -f "$c" >/dev/null 2>&1 || true
        fi
    done
    # 2. Legacy compose-run leftovers (wrapper never creates these)
    for c in $(docker ps -a --filter "name=${PROJECT}-${PROJECT}-run-" --format '{{.Names}}' 2>/dev/null); do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
reap_orphans

# ── Ensure postgres is up ───────────────────────────────────────────
docker compose -f "$COMPOSE_FILE" up -d postgres >/dev/null 2>&1

# ── Ensure image exists (build once if missing) ─────────────────────
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" build mempalace >/dev/null 2>&1
fi

# ── Decide GPU mode ─────────────────────────────────────────────────
use_gpu=false
if on_ac_power; then
    # Only allocate GPU if no other mempalace MCP container already has it
    gpu_running=$(docker ps --filter "ancestor=${IMAGE}" --filter "status=running" -q 2>/dev/null | wc -l)
    if [ "$gpu_running" -eq 0 ]; then
        use_gpu=true
    fi
fi

# ── Build docker run command ────────────────────────────────────────
args=(
    docker run --rm -i
    --name "mempalace-mcp-$PPID"
    --network "$NETWORK"
    -e "DATABASE_URL=$DB_URL"
    -e "PYTHONPATH=/app"
    -v "${PROJECT}_model-cache:/root/.cache"
    -v "/home/sam/dev:/projects:ro"
    -v "$COMPOSE_DIR/mempalace:/app/mempalace:ro"
    -v "$COMPOSE_DIR/tests:/app/tests:ro"
    -v "/sys:/sys:ro"
    --entrypoint python
)

if $use_gpu; then
    args+=(--gpus all)
else
    args+=(-e "MEMPALACE_DEVICE=cpu")
fi

args+=("$IMAGE" -m mempalace.mcp_server "$@")

exec "${args[@]}"
