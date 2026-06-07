#!/usr/bin/env bash
# restore_controllers.sh — restore controller .pt files from tarball or directory.
#
# Usage:
#   cd ~/weave-hack && bash scripts/restore_controllers.sh ~/openmirror-backups/controllers-20250606.tar.gz
#   bash scripts/restore_controllers.sh /path/to/controllers/
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="${1:?usage: restore_controllers.sh <tarball.tar.gz-or-directory>}"

CTRL_DIR="controllers"
if [ -f "$REPO/.env" ]; then
  line="$(grep -E '^CONTROLLER(S)?_DIR=' "$REPO/.env" | tail -1 || true)"
  if [ -n "$line" ]; then
    CTRL_DIR="${line#*=}"
    CTRL_DIR="${CTRL_DIR%%#*}"
    CTRL_DIR="${CTRL_DIR//\"/}"
    CTRL_DIR="${CTRL_DIR//\'/}"
    CTRL_DIR="$(echo "$CTRL_DIR" | xargs)"
  fi
fi
CTRL_DIR="${CONTROLLER_DIR:-$CTRL_DIR}"
if [[ "$CTRL_DIR" != /* ]]; then
  CTRL_DIR="$REPO/$CTRL_DIR"
fi
mkdir -p "$CTRL_DIR"

if [ -f "$SRC" ]; then
  echo "extracting $SRC -> $(dirname "$CTRL_DIR")/"
  tar xzf "$SRC" -C "$(dirname "$CTRL_DIR")"
elif [ -d "$SRC" ]; then
  echo "copying $SRC/*.pt -> $CTRL_DIR/"
  cp "$SRC"/*.pt "$CTRL_DIR/"
else
  echo "not found: $SRC" >&2
  exit 1
fi

count="$(find "$CTRL_DIR" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')"
echo "restored: $count controller file(s) in $CTRL_DIR"
echo "verify:   curl -s http://127.0.0.1:8000/controllers | head"
echo "          (start Track A first if needed: bash start_all.sh)"
