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
BRANCH="${BRANCH:-controller-engine}"
MODEL="${PEFT_CMP_MODEL:-Qwen/Qwen2.5-7B}"
VENV="${VENV:-$HOME/venv}"
NTK_SRC="${NTK_SRC:-$HOME/ntkmirror_src}"

# These boxes have an unwritable system site-packages and an ambiguous
# system python (`pip` and `python`/`python3` can target different
# interpreters), which makes `import torch` fail even after a "successful"
# install. A venv removes that ambiguity: `python` == `pip` and site-packages
# is writeable. Activate it in every shell you use:  source $VENV/bin/activate
echo "=== [1/5] virtualenv ($VENV) ==="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "=== [2/5] python deps ==="
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  # running before the repo is cloned
  python -m pip install "torch>=2.3" "transformers>=4.44" accelerate datasets peft bitsandbytes \
              "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7"
fi

echo "=== [3/5] ntkmirror (clone, not pip — upstream packaging is broken) ==="
if [ ! -d "$NTK_SRC/.git" ]; then
  git clone https://github.com/leochlon/ntkmirror.git "$NTK_SRC"
else
  (cd "$NTK_SRC" && git pull --ff-only || true)
fi
# Make `import ntkmirror` permanent for this venv via a .pth file.
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$NTK_SRC/src" > "$SITE/ntkmirror_src.pth"
export PYTHONPATH="$NTK_SRC/src:${PYTHONPATH:-}"

echo "=== [4/5] repo ($BRANCH) ==="
if [ ! -d weave-hack/.git ]; then
  git clone "$REPO_URL"
fi
cd weave-hack
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || true

echo "=== sanity: GPU + ntkmirror import ==="
python - <<'PY'
import torch, ntkmirror
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("ntkmirror:", ntkmirror.__file__)
PY

echo "=== [5/5] pre-fetch base weights ($MODEL) ==="
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "$MODEL"
print("downloading", m, "...")
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("cached.")
PY

echo "=== ready. In each new shell first run:  source $VENV/bin/activate ==="
echo "  python smoke_compose_subtract.py                 # 0.5B operations smoke"
echo "  PEFT_CMP_MODEL=$MODEL python smoke_compose_subtract.py   # real 7B check"
echo "  uvicorn controller_service:app --host 0.0.0.0 --port 8000   # Track A service"
