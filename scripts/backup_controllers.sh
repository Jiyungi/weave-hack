#!/usr/bin/env bash
# backup_controllers.sh — tarball minted NTK controller .pt files for offline storage.
#
# Usage (on Brev box or anywhere with weave-hack):
#   cd ~/weave-hack && bash scripts/backup_controllers.sh
#   bash scripts/backup_controllers.sh /custom/output/dir
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_DIR="${1:-$HOME/openmirror-backups}"

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

if [ ! -d "$CTRL_DIR" ]; then
  echo "controller dir not found: $CTRL_DIR" >&2
  exit 1
fi

count="$(find "$CTRL_DIR" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')"
if [ "$count" = "0" ]; then
  echo "no *.pt files in $CTRL_DIR — nothing to backup" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
stamp="$(date +%Y%m%d-%H%M%S)"
base="$(basename "$CTRL_DIR")"
archive="$OUT_DIR/${base}-${stamp}.tar.gz"

tar czf "$archive" -C "$(dirname "$CTRL_DIR")" "$(basename "$CTRL_DIR")"
bytes="$(wc -c < "$archive" | tr -d ' ')"
echo "backed up $count controller(s) from $CTRL_DIR"
echo "archive: $archive ($bytes bytes)"
echo ""
echo "copy to laptop, e.g.:"
echo "  scp <brev-host>:$archive ~/openmirror-backups/"
