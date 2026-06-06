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

echo "=== [1/4] python deps ==="
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  # running before the repo is cloned
  pip install "torch>=2.3" "transformers>=4.44" accelerate datasets peft bitsandbytes \
              "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7"
  pip install "git+https://github.com/leochlon/ntkmirror.git"
fi

echo "=== [2/4] repo ($BRANCH) ==="
if [ ! -d weave-hack/.git ]; then
  git clone "$REPO_URL"
fi
cd weave-hack
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || true

echo "=== [3/4] sanity: GPU + ntkmirror import ==="
python - <<'PY'
import torch, ntkmirror
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("ntkmirror:", ntkmirror.__file__)
PY

echo "=== [4/4] pre-fetch base weights ($MODEL) ==="
python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "$MODEL"
print("downloading", m, "...")
AutoTokenizer.from_pretrained(m)
AutoModelForCausalLM.from_pretrained(m)
print("cached.")
PY

echo "=== ready. Next: ==="
echo "  python smoke_compose_subtract.py                 # 0.5B operations smoke"
echo "  PEFT_CMP_MODEL=$MODEL python smoke_compose_subtract.py   # real 7B check"
echo "  uvicorn controller_service:app --host 0.0.0.0 --port 8000   # Track A service"
