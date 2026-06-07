"""Nightly consolidation job — "sleep": bake the day's chats into the weights.

For every Unit that has accumulated interactions in Redis, run the consolidation
pipeline (collect -> curate -> train -> eval-gate -> promote/reject) and log the
learning metrics to Weave/W&B: consolidation score (did we learn today),
forgetting score (did we keep earlier days), gate deviation, curation yield, and
the promote/reject decision.

Run (server should be stopped to free the GPU on a single-GPU laptop)::

    cd ml
    python -m scripts.consolidate                 # all units with interactions
    python -m scripts.consolidate --units alice    # specific units
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ML = Path(__file__).resolve().parents[1]
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from weaveself.consolidation import consolidate_unit  # noqa: E402
from weaveself.integration.redis_client import create_redis_client  # noqa: E402
from weaveself.training.nkt_trainer import train_adapter_nkt  # noqa: E402

REPO = _ML.parent


def _resolve(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (REPO / p))


def _discover_units(redis_client) -> list[str]:
    """Find units that have interactions logged (scan interactions:* keys)."""
    backend = redis_client._backend  # noqa: SLF001 — intentional for the CLI
    client = getattr(backend, "_client", None)
    if client is not None and hasattr(client, "keys"):
        keys = client.keys("interactions:*")
        return sorted(
            (k.decode() if isinstance(k, bytes) else k).split("interactions:", 1)[1]
            for k in keys
        )
    return []


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO / ".env")
    parser = argparse.ArgumentParser(description="Nightly consolidation job.")
    parser.add_argument("--units", nargs="*", default=None, help="Units to consolidate (default: all with interactions).")
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip())
    parser.add_argument("--device", default=os.environ.get("TORCH_DEVICE", "").strip() or None)
    parser.add_argument("--dtype", default=os.environ.get("MODEL_DTYPE", "bfloat16").strip() or "bfloat16")
    parser.add_argument("--adapters-dir", default=_resolve(os.environ.get("ADAPTERS_DIR", "./data/adapters").strip()))
    parser.add_argument("--no-weave", action="store_true", help="Skip Weave/W&B logging.")
    args = parser.parse_args(argv)

    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379").strip()
    redis_client = create_redis_client(url=redis_url, file_path=str(REPO / "data" / "redis_store.json"))

    units = args.units if args.units else _discover_units(redis_client)
    if not units:
        print("No units with interactions to consolidate.", file=sys.stderr)
        return 1

    adapters_dir = Path(args.adapters_dir)

    # Weave: observe the learning loop.
    weave_run = None
    wandb_mod = None
    if not args.no_weave:
        try:
            import weave
            import wandb

            entity = None
            if os.environ.get("WANDB_API_KEY"):
                wandb.login(key=os.environ["WANDB_API_KEY"])
                entity = wandb.Api().default_entity
            project = os.environ.get("WEAVE_PROJECT", "weaveself").strip() or "weaveself"
            weave.init(f"{entity}/{project}" if entity else project)
            wandb_mod = wandb
            weave_run = wandb.init(project=project, entity=entity, job_type="consolidation", reinit=True)
        except Exception as exc:
            print(f"[consolidate] Weave/W&B unavailable ({exc}); continuing without it.", flush=True)

    def make_engine_factory():
        created: list[object] = []

        def factory():
            from weaveself.serving.backend import HFBackend
            from weaveself.serving.engine import ServingEngine

            backend = HFBackend(device=args.device, torch_dtype=args.dtype)
            eng = ServingEngine(args.base_model, backend=backend, adapters_dir=str(adapters_dir))
            created.append(eng)
            return eng

        return factory, created

    for unit in units:
        print(f"\n[consolidate] === unit '{unit}' ===", flush=True)
        factory, created = make_engine_factory()
        result = consolidate_unit(
            unit,
            redis_client=redis_client,
            engine_factory=factory,
            train_fn=train_adapter_nkt,
            adapters_dir=adapters_dir,
        )
        # Free the resident model before the next unit's training (single-GPU).
        created.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        print(
            f"[consolidate] {unit}: promoted={result.promoted} "
            f"consolidation={result.consolidation_score:.3f} "
            f"forgetting={result.forgetting_score:.3f} "
            f"gate_dev={result.gate_deviation:.4f} "
            f"yield={result.curation_yield:.2f} | {result.decision_reason}",
            flush=True,
        )
        if wandb_mod is not None:
            wandb_mod.log({f"consolidation/{unit}/{k}": v for k, v in result.to_dict().items() if isinstance(v, (int, float))})
            try:
                import weave

                weave.publish(result.to_dict(), name=f"consolidation_{unit}_v{result.version}")
            except Exception:
                pass

    if weave_run is not None:
        try:
            weave_run.finish()
        except Exception:
            pass
        entity = weave_run.entity
        project = weave_run.project
        print(f"\n[consolidate] Weave: https://wandb.ai/{entity}/{project}/weave", flush=True)
    print("[consolidate] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
