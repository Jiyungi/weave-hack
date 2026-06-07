#!/usr/bin/env bash
# Ensure Python deps from requirements.txt and verify Track B can reach Redis.
#
# Usage (on the box):
#   bash scripts/verify_box_deps.sh
#   bash scripts/verify_box_deps.sh --skip-redis   # pip only, no REDIS_URL ping
#
set -euo pipefail

VENV="${VENV:-$HOME/venv}"
REPO="${REPO:-$HOME/weave-hack}"
SKIP_REDIS=0
SKIP_PIP="${SKIP_PIP:-0}"
for arg in "$@"; do
  case "$arg" in
    --skip-redis) SKIP_REDIS=1 ;;
    --skip-pip) SKIP_PIP=1 ;;
  esac
done

if [ ! -f "$VENV/bin/activate" ]; then
  echo "venv not found at $VENV — run: bash setup_brev.sh" >&2
  exit 1
fi
if [ ! -f "$REPO/requirements.txt" ]; then
  echo "repo not found at $REPO — run: bash setup_brev.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
cd "$REPO"

echo "=== verify: pip install -r requirements.txt ==="
if [ "$SKIP_PIP" = "1" ]; then
  echo "  (skipped — SKIP_PIP=1 / --skip-pip)"
else
  python -m pip install -r requirements.txt
fi

echo "=== verify: critical imports ==="
python - <<'PY'
import importlib.util
missing = [m for m in ("redis", "fastapi", "uvicorn", "openai", "transformers") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing packages after install: {', '.join(missing)}")
import transformers
major = int(transformers.__version__.split(".")[0])
if major >= 5:
    raise SystemExit(
        f"transformers {transformers.__version__} >=5 breaks vLLM 0.11 — "
        "run: bash scripts/ensure_brain_deps.sh"
    )
print("imports OK (transformers", transformers.__version__ + ")")
PY

if [ "$SKIP_REDIS" = "1" ]; then
  echo "=== verify: skipping REDIS_URL ping (--skip-redis) ==="
  exit 0
fi

if [ ! -f "$REPO/.env" ]; then
  echo "=== verify: no .env — copy .env.example and set REDIS_URL ===" >&2
  exit 1
fi

echo "=== verify: REDIS_URL reachable ==="
ENV_FILE="$REPO/.env" python - <<'PY'
import os, re, sys
from pathlib import Path

p = Path(os.environ["ENV_FILE"])
for line in p.read_text().splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        continue
    k, _, v = s.partition("=")
    v = re.split(r"\s+#", v.strip(), maxsplit=1)[0].strip().strip('"').strip("'")
    os.environ[k.strip()] = v

url = (os.environ.get("REDIS_URL") or "").strip()
if not url:
    sys.exit("REDIS_URL missing in .env")

scheme = url.split("://", 1)[0]
print(f"REDIS_URL scheme: {scheme}")

from control_plane.redis_client import get_redis

get_redis().ping()
print("Redis: OK")
PY

echo "=== verify: box deps OK ==="
