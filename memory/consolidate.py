"""Consolidate logged chats into a user_style adapter via Track B personalize."""
from __future__ import annotations

import os

from control_plane import store
from control_plane.trace import op

from .curation import curate_interactions, curate_with_openai


@op(name="memory.consolidate")
def consolidate_user(user_id: str, *, use_openai: bool | None = None) -> dict:
    """Collect → curate → mint ``user_style-{user_id}`` → delete raw logs."""
    user_id = user_id.strip()
    if not user_id:
        raise store.CPError("user_id required")
    interactions = store.get_interactions(user_id)
    if not interactions:
        raise store.CPError(f"no interactions logged for user {user_id!r}")

    if use_openai is None:
        use_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if use_openai:
        pairs, discarded = curate_with_openai(interactions, user_id)
    else:
        pairs, discarded = curate_interactions(interactions)

    if not pairs:
        raise store.CPError(f"could not curate any training pairs for {user_id!r}")

    result = store.personalize(user_id, pairs)
    deleted = store.clear_interactions(user_id)
    return {
        "user_id": user_id,
        "promoted": True,
        "raw_interactions": len(interactions),
        "curated_pairs": len(pairs),
        "discarded": discarded,
        "logs_deleted": deleted,
        "controller_id": result.get("controller_id"),
        "loss_first": result.get("loss_first"),
        "loss_last": result.get("loss_last"),
    }


if __name__ == "__main__":
    import argparse
    import sys

    from control_plane.store import CPError

    parser = argparse.ArgumentParser(description="Consolidate chat logs → user_style adapter.")
    parser.add_argument("--user", required=True, help="user_id")
    parser.add_argument("--no-openai", action="store_true", help="Heuristic curation only.")
    args = parser.parse_args()
    try:
        out = consolidate_user(args.user, use_openai=False if args.no_openai else None)
    except CPError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"consolidated {out['user_id']}: {out['curated_pairs']} pairs → {out['controller_id']}")
    print(f"  raw={out['raw_interactions']} discarded={out['discarded']} logs_deleted={out['logs_deleted']}")
    print(f"  loss {out.get('loss_first')} → {out.get('loss_last')}")
