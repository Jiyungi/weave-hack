#!/usr/bin/env bash
# Pin transformers for vLLM 0.11.x and optionally install the brain server.
#
# vLLM 0.11 reads tokenizer.all_special_tokens_extended, which transformers 5.x
# removed — so pip/uv can silently install an incompatible combo.
#
# Usage (on the box):
#   bash scripts/ensure_brain_deps.sh          # pin transformers + install vLLM
#   INSTALL_VLLM=0 bash scripts/ensure_brain_deps.sh   # pin transformers only
#
set -euo pipefail

VENV="${VENV:-$HOME/venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "venv not found at $VENV — run: bash setup_brev.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

pin_transformers() {
  python -m pip install "transformers>=4.55.2,<5.0.0"
  python - <<'PY'
import transformers
major = int(transformers.__version__.split(".")[0])
if major >= 5:
    raise SystemExit(
        f"transformers {transformers.__version__} still >=5 after pin — fix manually"
    )
print(f"transformers OK: {transformers.__version__}")
PY
}

install_vllm() {
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
  fi
  VIRTUAL_ENV="$VENV" uv pip install "vllm==${VLLM_VERSION}" --torch-backend="$TORCH_BACKEND"
  pin_transformers
  if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm CLI not on PATH after install" >&2
    exit 1
  fi
  echo "vllm OK: $(vllm --version 2>/dev/null | head -1 || echo "$VLLM_VERSION")"
}

echo "=== brain deps: transformers <5 (vLLM 0.11 compat) ==="
pin_transformers

if [ "$INSTALL_VLLM" = "1" ]; then
  echo "=== brain deps: vLLM ${VLLM_VERSION} (${TORCH_BACKEND}) ==="
  install_vllm
else
  echo "=== skipping vLLM install (INSTALL_VLLM=0) ==="
fi

echo "=== brain deps ready ==="
