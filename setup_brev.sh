#!/usr/bin/env bash
# setup_brev.sh — boot a fresh Brev box into a ready-to-run state.
#
# Usage on the box:
#   bash setup_brev.sh          # full bootstrap (~15–30 min first time)
#   bash setup_brev.sh ui       # Node 20 + npm install only (~2 min)
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Jiyungi/weave-hack.git}"
BRANCH="${BRANCH:-main}"
MODEL="${PEFT_CMP_MODEL:-Qwen/Qwen2.5-7B}"
VENV="${VENV:-$HOME/venv}"
NTK_SRC="${NTK_SRC:-$HOME/ntkmirror_src}"
REPO="${REPO:-$HOME/weave-hack}"

install_node_ui() {
  echo "=== Node 20 + UI deps (Next.js needs npm) ==="
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

  local need=0
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then need=1; fi
  if ! command -v npm >/dev/null 2>&1; then need=1; fi
  local node_major=""
  node_major="$(node -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/' || true)"
  if [ -z "$node_major" ] || [ "$node_major" -lt 18 ]; then need=1; fi

  if [ "$need" = "1" ]; then
    echo "  installing nvm + Node 20..."
    if [ ! -s "$NVM_DIR/nvm.sh" ]; then
      curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    fi
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
    nvm install 20
    nvm alias default 20
    nvm use 20
  else
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
    nvm use default 2>/dev/null || nvm use 20
  fi

  command -v npm >/dev/null 2>&1 || {
    echo "ERROR: npm still missing after nvm install" >&2
    exit 1
  }
  echo "  node: $(node -v) | npm: $(npm -v)"

  if [ ! -f ui/package.json ]; then
    echo "ERROR: ui/package.json not found (run from $REPO)" >&2
    exit 1
  fi
  (cd ui && npm install --no-audit --no-fund)
  if [ ! -f ui/.env.local ] && [ -f ui/.env.example ]; then
    cp ui/.env.example ui/.env.local
    echo "  created ui/.env.local"
  fi
  echo "=== UI deps OK ==="
}

if [ "${1:-}" = "ui" ]; then
  cd "$REPO"
  install_node_ui
  echo ""
  echo "  Next: bash start_all.sh restart"
  echo "  Mac:  brev port-forward narwhals --port 3000:3000"
  exit 0
fi

# --- full bootstrap ---
echo "=== [1/8] virtualenv ($VENV) ==="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "=== [2/8] PyTorch (CUDA-matched wheel) ==="
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
python -m pip install torch --index-url "$TORCH_CUDA_INDEX"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
      "| built for CUDA:", torch.version.cuda)
PY

echo "=== [3/8] ntkmirror (clone, not pip — upstream packaging is broken) ==="
if [ ! -d "$NTK_SRC/.git" ]; then
  git clone https://github.com/leochlon/ntkmirror.git "$NTK_SRC"
else
  (cd "$NTK_SRC" && git pull --ff-only || true)
fi
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$NTK_SRC/src" > "$SITE/ntkmirror_src.pth"
export PYTHONPATH="$NTK_SRC/src:${PYTHONPATH:-}"

echo "=== [4/8] repo ($BRANCH) ==="
if [ -d "$REPO/.git" ]; then
  cd "$REPO"
elif [ -d weave-hack/.git ]; then
  cd weave-hack
else
  git clone "$REPO_URL" "$REPO"
  cd "$REPO"
fi
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || true

echo "=== [5/8] python deps (requirements.txt — must run AFTER clone) ==="
python -m pip install -r requirements.txt

echo "=== sanity: GPU + ntkmirror import ==="
python - <<'PY'
import torch, ntkmirror
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("ntkmirror:", ntkmirror.__file__)
PY

echo "=== [6/8] pre-fetch base weights ($MODEL) ==="
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "$MODEL"
print("downloading", m, "...")
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("cached.")
PY

echo "=== [7/8] Track C UI (Next.js + CopilotKit) ==="
install_node_ui

echo "=== [redis] local redis-server (skip if using Redis Cloud only) ==="
if command -v redis-server >/dev/null 2>&1; then
  redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes >/dev/null 2>&1 || true
  redis-cli ping >/dev/null 2>&1 && echo "  redis: running" || echo "  redis: not running (OK if REDIS_URL is cloud)"
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get install -y redis-server >/dev/null 2>&1 && redis-server --daemonize yes >/dev/null 2>&1 || true
  echo "  redis: installed"
else
  echo "  redis: skipped (use Redis Cloud in .env)"
fi

echo "=== [8/8] vLLM brain (optional; INSTALL_VLLM=0 to skip) ==="
python -m pip install "transformers>=4.55.2,<5.0.0"
if [ "${INSTALL_VLLM:-1}" != "0" ]; then
  python -m pip install uv
  VIRTUAL_ENV="$VENV" uv pip install "vllm==0.11.0" --torch-backend=cu128
  python -m pip install "transformers>=4.55.2,<5.0.0"
fi

echo ""
echo "=== ready ==="
echo ""
echo "  bash start_all.sh"
echo "  Mac: brev port-forward narwhals --port 3000:3000  →  http://localhost:3000"
