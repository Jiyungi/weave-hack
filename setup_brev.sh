#!/usr/bin/env bash
# setup_brev.sh — boot a fresh Brev box into a ready-to-run state.
#
# The Brev A100 boxes are delete-only (no stop/start) and lose all data on
# delete. So this script makes a fresh box reproducible: install deps, pull the
# repo, and pre-fetch the 7B weights. Real controllers (~100 KB .pt files) live
# in git, so after a delete you reload them instantly — you never re-fit.
#
# Usage on the box (Jupyter terminal):
#   bash setup_brev.sh
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Jiyungi/weave-hack.git}"
BRANCH="${BRANCH:-unified}"
MODEL="${PEFT_CMP_MODEL:-Qwen/Qwen2.5-7B}"
VENV="${VENV:-$HOME/venv}"
NTK_SRC="${NTK_SRC:-$HOME/ntkmirror_src}"

# These boxes have an unwritable system site-packages and an ambiguous
# system python (`pip` and `python`/`python3` can target different
# interpreters), which makes `import torch` fail even after a "successful"
# install. A venv removes that ambiguity: `python` == `pip` and site-packages
# is writeable. Activate it in every shell you use:  source $VENV/bin/activate
echo "=== [1/8] virtualenv ($VENV) ==="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "=== [2/8] PyTorch (CUDA-matched wheel) ==="
# Match the box's CUDA *driver* (Brev boxes are typically on a 12.x driver). The
# default torch wheel is built for CUDA 13 and fails on a 12.x driver ("driver is
# too old"), silently falling back to CPU. Install a CUDA 12.8 build first so the
# requirements step finds torch already satisfied and doesn't pull cu130.
# If a box has an older driver (e.g. 12.4), change cu128 -> cu126 / cu124.
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
# Make `import ntkmirror` permanent for this venv via a .pth file.
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$NTK_SRC/src" > "$SITE/ntkmirror_src.pth"
export PYTHONPATH="$NTK_SRC/src:${PYTHONPATH:-}"

echo "=== [4/8] repo ($BRANCH) ==="
if [ ! -d weave-hack/.git ]; then
  git clone "$REPO_URL"
fi
cd weave-hack
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
# Next.js 14 needs Node 18+. The box's system Node is often ancient (v12),
# which crashes the `next` binary. Install Node 20 via nvm (no sudo needed) if
# node is missing or older than 18.
NODE_MAJOR="$(node -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
if [ -z "$NODE_MAJOR" ] || [ "$NODE_MAJOR" -lt 18 ]; then
  echo "  Node missing or too old (${NODE_MAJOR:-none}); installing Node 20 via nvm..."
  export NVM_DIR="$HOME/.nvm"
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  fi
  # shellcheck disable=SC1090
  . "$NVM_DIR/nvm.sh"
  nvm install 20 && nvm alias default 20 && nvm use 20
fi
echo "  node: $(node -v 2>/dev/null || echo none) | npm: $(npm -v 2>/dev/null || echo none)"
if [ -f ui/package.json ]; then
  (cd ui && npm install --no-audit --no-fund) || echo "  npm install failed — run manually: cd ui && npm install"
  if [ ! -f ui/.env.local ] && [ -f ui/.env.example ]; then
    cp ui/.env.example ui/.env.local
    echo "  created ui/.env.local from .env.example"
  fi
else
  echo "  ui/package.json not found — skip"
fi

echo "=== [redis] required — governance state + audit (sponsor) ==="
if command -v redis-server >/dev/null 2>&1; then
  redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes >/dev/null 2>&1 || true
  if redis-cli ping >/dev/null 2>&1; then
    echo "  redis: running"
  else
    echo "  ERROR: redis-server installed but not running" >&2
    exit 1
  fi
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get install -y redis-server >/dev/null 2>&1 || {
    echo "  ERROR: install redis-server: sudo apt-get install -y redis-server" >&2
    exit 1
  }
  redis-server --daemonize yes >/dev/null 2>&1 || true
  echo "  redis: installed and started"
else
  echo "  ERROR: redis-server required — install manually" >&2
  exit 1
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
echo "  One command:       bash start_all.sh"
echo "  (needs tmux:      sudo apt-get install -y tmux)"
echo ""
echo "  Or 5 manual tabs — see README 'Run it'"
echo "  On your laptop:    brev port-forward <instance> --port 3000:3000"
echo "  Open:              http://localhost:3000"
