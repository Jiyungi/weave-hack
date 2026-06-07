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

  if session_running; then
    echo "session '$SESSION' already running — attaching (use 'stop' or 'restart' to reset)"
    exec tmux attach -t "$SESSION"
  fi

  local act="source \"$VENV/bin/activate\""
  local repo_cd="cd \"$REPO\""

  tmux new-session -d -s "$SESSION" -n brain -x 200 -y 50
  tmux send-keys -t "$SESSION:brain" \
    "$act && python -m pip install -q vllm 2>/dev/null || true && \
vllm serve $BRAIN_MODEL --port $BRAIN_PORT \
  --max-model-len 8192 --gpu-memory-utilization $BRAIN_GPU_UTIL" C-m

  local wait_brain="for i in \$(seq 1 150); do curl -sf http://127.0.0.1:${BRAIN_PORT}/health >/dev/null && break; sleep 2; done"
  local wait_cp="for i in \$(seq 1 60); do curl -sf http://127.0.0.1:8100/health >/dev/null && break; sleep 2; done"

  tmux new-window -t "$SESSION" -n track-a
  tmux send-keys -t "$SESSION:track-a" \
    "$act && $repo_cd && echo '[track-a] waiting for brain :${BRAIN_PORT}...' && $wait_brain && \
uvicorn controller_service:app --host 0.0.0.0 --port 8000" C-m

  tmux new-window -t "$SESSION" -n track-b
  tmux send-keys -t "$SESSION:track-b" \
    "$act && $repo_cd && uvicorn control_plane_service:app --host 0.0.0.0 --port 8100" C-m

  tmux new-window -t "$SESSION" -n track-d
  tmux send-keys -t "$SESSION:track-d" \
    "$act && $repo_cd && echo '[track-d] waiting for control plane :8100...' && $wait_cp && \
uvicorn agent_service:app --host 0.0.0.0 --port 8200" C-m

  tmux new-window -t "$SESSION" -n ui
  tmux send-keys -t "$SESSION:ui" \
    "cd \"$REPO/ui\" && cp -n .env.example .env.local 2>/dev/null || true && npm run dev" C-m

  tmux select-window -t "$SESSION:brain"

  echo ""
  echo "started tmux session '$SESSION' with 5 windows: brain track-a track-b track-d ui"
  echo ""
  echo "  brain loads first (~1–2 min). track-a waits for :$BRAIN_PORT; track-d waits for :8100."
  echo "  attach:       bash start_all.sh attach   (detach: Ctrl-b d)"
  echo "  status:       bash start_all.sh status"
  echo "  on laptop:    brev port-forward <instance> --port 3000:3000"
  echo "  open:         http://localhost:3000"
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
