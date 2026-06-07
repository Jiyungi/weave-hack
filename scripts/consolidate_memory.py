#!/usr/bin/env python3
"""Consolidate a user's chat logs into user_style-{user_id} via Track A.

Usage:
  python scripts/consolidate_memory.py --user alice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_plane.store import CPError
from memory.consolidate import consolidate_user


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate chat logs → user_style adapter.")
    parser.add_argument("--user", required=True, help="user_id")
    parser.add_argument("--no-openai", action="store_true", help="Heuristic curation only.")
    args = parser.parse_args(argv)
    try:
        out = consolidate_user(args.user, use_openai=False if args.no_openai else None)
    except CPError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"consolidated {out['user_id']}: {out['curated_pairs']} pairs → {out['controller_id']}")
    print(f"  raw={out['raw_interactions']} discarded={out['discarded']} logs_deleted={out['logs_deleted']}")
    print(f"  loss {out.get('loss_first')} → {out.get('loss_last')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
