"""Mint controllers for the Tier 1–3 extra tools (run on the box).

For each tool in ``agents.tools_extra``, calls the control plane's one-shot
committee endpoint ``POST {CP_URL}/register`` with the tool's
``training_examples()`` so the NTK engine mints a ~100 KB controller (~36 s
each), registers the skill, and grants it to a principal (default
``exec-assistant``).

Requires the control plane (:8100) and NTK engine (:8000) to be running.

Usage (on the Brev box, venv active):
    python -m scripts.mint_tools                 # mint all extra tools
    python -m scripts.mint_tools --names currency unit_convert
    python -m scripts.mint_tools --principal exec-assistant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents import tools_extra  # noqa: E402

CP_URL = os.environ.get("CP_URL", "http://127.0.0.1:8100").rstrip("/")


def _post(path: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        f"{CP_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mint extra-tool controllers.")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Subset of tool names to mint (default: all extra tools).")
    parser.add_argument("--principal", default="exec-assistant",
                        help="Principal to grant the minted skills to.")
    args = parser.parse_args(argv)

    tools = tools_extra.extra_tools()
    if args.names:
        wanted = set(args.names)
        tools = [t for t in tools if t.name in wanted]
    if not tools:
        print("no matching tools to mint", file=sys.stderr)
        return 1

    print(f"Minting {len(tools)} tool controllers via {CP_URL}/register "
          f"(granting to {args.principal!r})...", flush=True)
    ok, fail = [], []
    for i, t in enumerate(tools, 1):
        payload = {
            "skill": t.name,
            "description": t.description,
            "examples": t.training_examples(),
            "grants": {args.principal: []},
        }
        t0 = time.time()
        try:
            _post("/register", payload)
            dt = time.time() - t0
            print(f"  [{i}/{len(tools)}] minted {t.name:<14} ({dt:5.1f}s)", flush=True)
            ok.append(t.name)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(tools)}] FAILED {t.name}: {e}", file=sys.stderr, flush=True)
            fail.append(t.name)

    print(f"\nDone. minted={len(ok)} failed={len(fail)}")
    if fail:
        print("failed:", ", ".join(fail))
        return 1
    print("granted to:", args.principal)
    print("Re-run a chat with that worker to use the new tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
