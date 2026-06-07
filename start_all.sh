#!/usr/bin/env bash
# start_all.sh — launch OpenMirror in a tmux session (one command on the box).
#
# Usage:
#   bash start_all.sh          # start (or attach if already running)
#   bash start_all.sh start    # same
#   bash start_all.sh stop     # kill the tmux session
#   bash start_all.sh status   # ports + tmux windows
#   bash start_all.sh attach   # attach to logs
#   bash start_all.sh restart  # stop then start
#
# On your laptop (separate terminal):
#   brev port-forward <instance> --port 3000:3000
#   open http://localhost:3000
#
set -euo pipefail

SESSION="${OPENMIRROR_SESSION:-openmirror}"
VENV="${VENV:-$HOME/venv}"
REPO="${REPO:-$HOME/weave-hack}"
BRAIN_MODEL="${OPENMIRROR_BRAIN_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BRAIN_PORT="${OPENMIRROR_BRAIN_PORT:-8001}"
BRAIN_GPU_UTIL="${OPENMIRROR_BRAIN_GPU_UTIL:-0.45}"

CMD="${1:-start}"

load_repo_env() {
  # Do NOT `source .env` — cron-like values (e.g. BATCH_SCHEDULE=0 3 * * *) break bash.
  local env_file="$REPO/.env"
  [ -f "$env_file" ] || return 0
  export ENV_FILE="$env_file"
  eval "$("$VENV/bin/python" - <<'PY'
import os, re, shlex
from pathlib import Path
p = Path(os.environ["ENV_FILE"])
for line in p.read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        continue
    key, _, val = s.partition("=")
    key = key.strip()
    val = val.strip()
    if val and val[0] not in "\"'":
        val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1]
    if key:
        print(f"export {key}={shlex.quote(val)}")
PY
)"
}

wait_for_port() {
  local port="$1" label="$2" max_secs="${3:-180}"
  local i=0
  echo "[$label] waiting for :$port (up to ${max_secs}s)..."
  while [ "$i" -lt "$max_secs" ]; do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[$label] :$port is up"
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  echo "[$label] timed out waiting for :$port" >&2
  return 1
}

port_up() {
  curl -sf "http://127.0.0.1:${1}/health" >/dev/null 2>&1
}

session_running() {
  tmux has-session -t "$SESSION" 2>/dev/null
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required. On the box: sudo apt-get install -y tmux" >&2
    exit 1
  fi
}

require_venv() {
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "venv not found at $VENV — run: bash setup_brev.sh" >&2
    exit 1
  fi
}

require_repo() {
  if [ ! -f "$REPO/controller_service.py" ]; then
    echo "repo not found at $REPO — run setup_brev.sh or set REPO=..." >&2
    exit 1
  fi
}

stop_session() {
  if session_running; then
    tmux kill-session -t "$SESSION"
    echo "stopped tmux session '$SESSION'"
  else
    echo "no session '$SESSION' running"
  fi
}

show_status() {
  echo "session: $SESSION"
  if session_running; then
    echo "tmux:    running ($(tmux list-windows -t "$SESSION" -F '#{window_name}' | tr '\n' ' '))"
  else
    echo "tmux:    not running"
  fi
  echo ""
  printf "  %-8s %-6s %s\n" "SERVICE" "PORT" "HEALTH"
  for row in \
    "brain:8001" \
    "track-a:8000" \
    "track-b:8100" \
    "track-d:8200"; do
    name="${row%%:*}"
    port="${row##*:}"
    if port_up "$port"; then
      printf "  %-8s %-6s %s\n" "$name" "$port" "up"
    else
      printf "  %-8s %-6s %s\n" "$name" "$port" "down"
    fi
  done
  if curl -sf "http://127.0.0.1:3000" >/dev/null 2>&1; then
    printf "  %-8s %-6s %s\n" "ui" "3000" "up"
  else
    printf "  %-8s %-6s %s\n" "ui" "3000" "down"
  fi
  echo ""
  echo "attach logs:  bash start_all.sh attach"
  echo "laptop:       brev port-forward <instance> --port 3000:3000"
}

start_session() {
  require_tmux
  require_venv
  require_repo
  ENV_FILE="$REPO/.env" load_repo_env

  if session_running; then
    echo "session '$SESSION' already running — attaching (use 'stop' or 'restart' to reset)"
    exec tmux attach -t "$SESSION"
  fi

  echo "=== preflight: python deps + REDIS_URL ==="
  SKIP_PIP=1 bash "$REPO/scripts/verify_box_deps.sh" || {
    echo "Preflight failed. Repair with:  bash scripts/verify_box_deps.sh" >&2
    exit 1
  }

  # Redis (required): use REDIS_URL from .env (cloud or local).
  local redis_url="${REDIS_URL:-redis://localhost:6379/0}"
  export REDIS_URL="$redis_url"
  if [[ "$redis_url" == redis://127.0.0.1* ]] || [[ "$redis_url" == redis://localhost* ]]; then
    if ! command -v redis-server >/dev/null 2>&1; then
      echo "redis-server is required for local REDIS_URL. Install: sudo apt-get install -y redis-server" >&2
      exit 1
    fi
    redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes >/dev/null 2>&1 || true
    if ! redis-cli ping >/dev/null 2>&1; then
      echo "Local Redis is not running. Start redis-server and retry." >&2
      exit 1
    fi
  fi

  # ntkmirror is a clone on PYTHONPATH (its .pth can be wiped by pip/uv installs);
  # exporting it here makes Track A's `import ntkmirror` self-heal on every start.
  local ntk_src="${NTK_SRC:-$HOME/ntkmirror_src}/src"
  local act="source \"$VENV/bin/activate\" && export PYTHONPATH=\"$ntk_src:\${PYTHONPATH:-}\""
  if [ -n "$redis_url" ]; then
    act="$act && export REDIS_URL=\"$redis_url\""
  fi
  local repo_cd="cd \"$REPO\""

  # Do NOT auto-install vllm here: recent vllm wheels target CUDA 12.9/13.0,
  # which need driver >= 575/580. On a CUDA 12.8 box (driver 570) install the
  # last vllm whose DEFAULT wheel is cu128 via setup_brev.sh / ensure_brain_deps.sh.
  tmux new-session -d -s "$SESSION" -n brain -x 200 -y 50
  tmux send-keys -t "$SESSION:brain" \
    "$act && $repo_cd && INSTALL_VLLM=0 bash scripts/ensure_brain_deps.sh 2>/dev/null || true; \
if command -v vllm >/dev/null 2>&1; then \
vllm serve $BRAIN_MODEL --port $BRAIN_PORT --max-model-len 8192 --gpu-memory-utilization $BRAIN_GPU_UTIL; \
else echo '[brain] vllm not installed. Run on box:'; \
echo '  bash scripts/ensure_brain_deps.sh'; fi" C-m

  local wait_cp="for i in \$(seq 1 60); do curl -sf http://127.0.0.1:8100/health >/dev/null && break; sleep 2; done"

  # Track A (NTK-Mirror engine) loads its own 7B via transformers; it does NOT
  # depend on the vLLM brain (:8001), so start it immediately.
  tmux new-window -t "$SESSION" -n track-a
  tmux send-keys -t "$SESSION:track-a" \
    "$act && $repo_cd && uvicorn controller_service:app --host 0.0.0.0 --port 8000" C-m

  tmux new-window -t "$SESSION" -n track-b
  tmux send-keys -t "$SESSION:track-b" \
    "$act && $repo_cd && uvicorn control_plane_service:app --host 0.0.0.0 --port 8100" C-m

  tmux new-window -t "$SESSION" -n track-d
  tmux send-keys -t "$SESSION:track-d" \
    "$act && $repo_cd && echo '[track-d] waiting for control plane :8100...' && $wait_cp && \
uvicorn agent_service:app --host 0.0.0.0 --port 8200" C-m

  # Next.js 14 needs Node 18+. Source nvm if present so the window uses the
  # nvm-managed Node (the box's system Node may be ancient, e.g. v12).
  tmux new-window -t "$SESSION" -n ui
  tmux send-keys -t "$SESSION:ui" \
    "export NVM_DIR=\"\$HOME/.nvm\"; [ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\"; \
cd \"$REPO/ui\" && cp -n .env.example .env.local 2>/dev/null || true && \
{ [ -x node_modules/.bin/next ] || npm install --no-audit --no-fund; } && npm run dev" C-m

  tmux select-window -t "$SESSION:brain"

  echo ""
  echo "started tmux session '$SESSION' with 5 windows: brain track-a track-b track-d ui"
  echo ""
  echo "  brain (vLLM) is OPTIONAL — only chat + Track D live reasoning need it."
  echo "  track-a + track-b are the governance demo and start on their own; track-d waits for :8100."
  echo "  state: Redis (see REDIS_URL in .env)"
  echo "  attach:       bash start_all.sh attach   (detach: Ctrl-b d)"
  echo "  status:       bash start_all.sh status"
  echo "  on laptop:    brev port-forward <instance> --port 3000:3000"
  echo "  open:         http://localhost:3000"
  echo ""
  echo "  memory: log chats with user_id (Agents panel) → consolidate:"
  echo "          python scripts/consolidate_memory.py --user alice"
  echo "          (or: python -m memory.consolidate --user alice)"
  echo "          (requires Redis — set REDIS_URL in .env)"
  echo ""
}

case "$CMD" in
  start)
    start_session
    ;;
  stop)
    require_tmux
    stop_session
    ;;
  restart)
    require_tmux
    stop_session
    start_session
    ;;
  status)
    show_status
    ;;
  attach)
    require_tmux
    if session_running; then
      exec tmux attach -t "$SESSION"
    else
      echo "session '$SESSION' not running — start with: bash start_all.sh" >&2
      exit 1
    fi
    ;;
  -h|--help|help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "unknown command: $CMD (try: start | stop | restart | status | attach)" >&2
    exit 1
    ;;
esac
