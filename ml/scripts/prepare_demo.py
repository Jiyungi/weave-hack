"""Prepare a REAL WeaveSelf demo end-to-end (Track A + Track B).

This script runs the *real* pipeline against the real instruct Base_Model
(default ``Qwen/Qwen2.5-1.5B-Instruct``) on the GPU when one is available:

1. Build a tiny per-Unit dataset (train + held-out) for a few demo Units
   (``cooking``, ``fitness``, ``finance``) from small inline interactions.
2. Train a REAL NKT-Mirror adapter per Unit via
   :func:`weaveself.training.train_adapter`, writing ``adapter_<id>``
   ``.safetensors`` / ``.json`` pairs into ``ADAPTERS_DIR``.
3. Store every adapter blob + metadata into the live Redis (or the documented
   file fallback) via :func:`weaveself.integration.create_redis_client`, and
   build the routing index so ``route(unit_label)`` resolves the adapter.
4. Load the Base_Model exactly once into a resident
   :class:`~weaveself.serving.engine.ServingEngine` (real :class:`HFBackend` on
   GPU) and run the real Weave eval: held-out perplexity base-vs-adapter, the
   cross-unit confusion matrix, the NKT-Mirror-vs-LoRA size chart, and
   base-vs-adapter generation examples.
5. Write a schema-conformant ``eval_results.json`` to ``EVAL_RESULTS_PATH``.

Run it (from ``ml/``)::

    python -m scripts.prepare_demo
    # or
    python scripts/prepare_demo.py --device cuda --dtype bfloat16

Configuration comes from the repo-root ``.env`` (loaded with ``python-dotenv``);
explicit CLI flags win over the environment. The script prefers a real run; it
only falls back to the dependency-free :class:`StubBackend` if the real model
genuinely cannot load (and prints exactly why).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``weaveself`` importable when run as ``python scripts/prepare_demo.py``.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from weaveself.contracts.training_pair import write_training_pairs  # noqa: E402
from weaveself.eval.weave_eval import HeldOutSet, WeaveEval  # noqa: E402


# ---------------------------------------------------------------------------
# Inline demo data — small, hand-written, per-Unit interactions.
# ---------------------------------------------------------------------------

# Each Unit has a distinct "voice" so the eval is meaningful. Train rows fit the
# adapter; held-out rows (no overlap) are scored base-vs-adapter.
DEMO_DATA: dict[str, dict[str, list[tuple[str, str]]]] = {
    "cooking": {
        "train": [
            ("How do I make pasta better?", "Salt the water like the sea, finish the pasta in the sauce, and save a ladle of starchy water to emulsify."),
            ("What should I cook tonight?", "A one-pan roast chicken with lemon and thyme, crisp skin, and pan juices spooned over everything."),
            ("How do I sharpen flavors?", "A splash of acid at the end — lemon or vinegar — wakes up almost any dish."),
            ("What's a good weeknight meal?", "Sheet-pan salmon with olive oil, garlic, and seasonal vegetables roasted until caramelized."),
            ("How do I cook steak?", "Dry the surface, season generously, sear hot for a deep crust, then rest it before slicing across the grain."),
            ("Any breakfast ideas?", "Soft scrambled eggs over buttered sourdough with blistered cherry tomatoes and good coffee."),
            ("How do I make soup richer?", "Build a base of sweated aromatics, deglaze, and simmer with a parmesan rind for savory depth."),
            ("What dessert is easy?", "A simple panna cotta: warm cream, bloom gelatin, sweeten lightly, and chill until just set."),
        ],
        "heldout": [
            ("How do I improve my sauce?", "Reduce it slowly, mount it with cold butter off the heat, and season until it tastes bright and balanced."),
            ("What's a good side dish?", "Charred broccolini with garlic, chili, and a squeeze of lemon over the top."),
            ("How do I season properly?", "Season in layers as you cook and taste constantly so nothing arrives flat at the plate."),
        ],
    },
    "fitness": {
        "train": [
            ("How do I get stronger?", "Train the big lifts with progressive overload, rest fully between sets, and recover with sleep and protein."),
            ("What's a good workout split?", "Push, pull, and legs across the week lets each muscle group train hard and recover well."),
            ("How do I build endurance?", "Increase weekly mileage gradually, keep most runs easy, and add one harder interval session."),
            ("How many reps should I do?", "Stay in the five to eight range for strength, and eight to twelve to chase size."),
            ("How do I avoid injury?", "Warm up, own your form before loading, and never let ego pick the weight on the bar."),
            ("What should I eat to grow?", "Hit a slight calorie surplus with enough protein, around one gram per pound of bodyweight."),
            ("How important is rest?", "Critical — muscle is built during recovery, so protect your sleep and your rest days."),
            ("How do I stay consistent?", "Schedule training like an appointment and track every session so progress stays visible."),
        ],
        "heldout": [
            ("How do I improve my squat?", "Brace hard, sit between your hips, drive through mid-foot, and add weight only when depth stays clean."),
            ("What's a good warm-up?", "Five minutes easy cardio, then dynamic mobility and light ramp-up sets of the day's first lift."),
            ("How do I lose fat?", "Hold a modest deficit, keep protein high, and keep lifting so you preserve hard-earned muscle."),
        ],
    },
    "finance": {
        "train": [
            ("How should I start investing?", "Build an emergency fund first, then invest steadily in low-cost broad index funds."),
            ("How do I budget?", "Track income and expenses, automate savings first, and let what remains cover discretionary spending."),
            ("Should I pay off debt?", "Clear high-interest debt aggressively; it is a guaranteed return no market can promise."),
            ("How much should I save?", "Aim for at least fifteen percent of income, and increase the rate with every raise."),
            ("What about retirement?", "Use tax-advantaged accounts early so decades of compounding do the heavy lifting for you."),
            ("How do I handle risk?", "Diversify broadly, match risk to your time horizon, and rebalance on a calm schedule."),
            ("Is timing the market wise?", "Time in the market beats timing it; consistent contributions outperform anxious guessing."),
            ("How do I build wealth?", "Spend less than you earn, invest the difference automatically, and stay patient for years."),
        ],
        "heldout": [
            ("How do I plan for a big purchase?", "Set a target date, save into a separate high-yield account, and avoid touching long-term investments."),
            ("What's a good first step financially?", "Automate a small recurring transfer to savings so the habit forms before the amount grows."),
            ("How do I think about fees?", "Minimize them relentlessly; a one percent fee quietly compounds into a large lifetime cost."),
        ],
    },
}

DEMO_UNIT_TYPE = "category"


def _repo_root() -> Path:
    return _ML_ROOT.parent


def _load_env() -> None:
    """Load the repo-root ``.env`` (env vars already set take precedence)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _repo_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (_repo_root() / p)


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _write_datasets(units: list[str], data_dir: Path) -> dict[str, dict[str, Path]]:
    """Write per-Unit train/held-out JSONL files; return their paths."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, dict[str, Path]] = {}
    for unit in units:
        spec = DEMO_DATA[unit]
        train_path = data_dir / f"demo_{unit}.jsonl"
        heldout_path = data_dir / f"demo_{unit}_heldout.jsonl"
        write_training_pairs(
            train_path,
            [
                {"prompt": p, "completion": c, "unit_label": unit}
                for p, c in spec["train"]
            ],
        )
        write_training_pairs(
            heldout_path,
            [
                {"prompt": p, "completion": c, "unit_label": unit}
                for p, c in spec["heldout"]
            ],
        )
        paths[unit] = {"train": train_path, "heldout": heldout_path}
    return paths


def _train_adapters(
    units: list[str],
    dataset_paths: dict[str, dict[str, Path]],
    adapters_dir: Path,
    base_model_id: str,
) -> dict[str, dict[str, object]]:
    """Train a real adapter per Unit; return id/path/size keyed by unit."""
    from weaveself.training import train_adapter

    adapters_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, object]] = {}
    for unit in units:
        blob_path = train_adapter(
            str(dataset_paths[unit]["train"]),
            unit_label=unit,
            unit_type=DEMO_UNIT_TYPE,
            base_model=base_model_id,
            out_dir=str(adapters_dir),
            day_index=0,
        )
        blob = Path(blob_path)
        adapter_id = blob.stem.removeprefix("adapter_")
        size_bytes = blob.stat().st_size
        out[unit] = {
            "adapter_id": adapter_id,
            "blob_path": blob,
            "size_bytes": size_bytes,
        }
        print(
            f"[prepare_demo] trained adapter unit={unit} id={adapter_id} "
            f"size={size_bytes}B -> {blob}",
            flush=True,
        )
    return out


def _store_in_redis(
    units: list[str],
    adapters: dict[str, dict[str, object]],
    adapters_dir: Path,
    redis_url: str | None,
    redis_file_path: Path,
) -> tuple[object, str]:
    """Store each adapter blob+metadata into Redis; return (client, backend_kind)."""
    import json

    from weaveself.integration.redis_client import (
        RedisKvBackend,
        create_redis_client,
    )

    client = create_redis_client(url=redis_url, file_path=str(redis_file_path))
    backend_kind = (
        "live-redis"
        if isinstance(getattr(client, "_backend", None), RedisKvBackend)
        else "file-fallback"
    )
    for unit in units:
        adapter_id = str(adapters[unit]["adapter_id"])
        blob_path = adapters_dir / f"adapter_{adapter_id}.safetensors"
        meta_path = adapters_dir / f"adapter_{adapter_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        blob = blob_path.read_bytes()
        client.store_adapter(meta, blob)
        # Sanity round-trip: metadata and blob come back byte-identical.
        assert client.fetch_blob(adapter_id) == blob
        print(
            f"[prepare_demo] stored adapter unit={unit} id={adapter_id} "
            f"into Redis ({backend_kind})",
            flush=True,
        )
    return client, backend_kind


def _build_engine(base_model_id: str, adapters_dir: Path, device: str, dtype: str):
    """Build a resident engine on the REAL model backend (no mock fallback).

    Returns ``(engine, "hf", note)``. Raises if the real model cannot load — the
    demo is production and never silently falls back to a stub.
    """
    from weaveself.serving.backend import HFBackend
    from weaveself.serving.engine import ServingEngine

    backend = HFBackend(device=device, torch_dtype=dtype)
    engine = ServingEngine(
        base_model_id, backend=backend, adapters_dir=str(adapters_dir)
    )
    return engine, "hf", f"real model on {device} ({dtype})"


def _estimate_lora_size_bytes(engine, backend_kind: str) -> int:
    """A principled LoRA size estimate for the size-chart comparison.

    Computed from the loaded model's real config (hidden/intermediate/layers)
    for a rank-8 LoRA over the standard attention + MLP projections in fp16, so
    the NKT-Mirror-vs-LoRA chart contrasts the tiny gate adapter against a
    realistic LoRA footprint. Falls back to a representative constant when the
    real config is unavailable (stub mode).
    """
    rank = 8
    bytes_per_param = 2  # fp16
    try:
        cfg = engine.backend._model.config  # type: ignore[attr-defined]
        hidden = int(cfg.hidden_size)
        inter = int(cfg.intermediate_size)
        layers = int(cfg.num_hidden_layers)
        kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
        heads = int(cfg.num_attention_heads)
        head_dim = hidden // heads
        kv_dim = kv_heads * head_dim
        # LoRA adds A(in x r) + B(r x out) per targeted linear.
        def lora(in_f: int, out_f: int) -> int:
            return rank * (in_f + out_f)

        per_layer = (
            lora(hidden, hidden)        # q_proj
            + lora(hidden, kv_dim)      # k_proj
            + lora(hidden, kv_dim)      # v_proj
            + lora(hidden, hidden)      # o_proj
            + lora(hidden, inter)       # gate_proj
            + lora(hidden, inter)       # up_proj
            + lora(inter, hidden)       # down_proj
        )
        return per_layer * layers * bytes_per_param
    except Exception:
        # Representative rank-8 LoRA footprint for a ~1.5B model when config
        # is unavailable (stub fallback): documented estimate, not a measurement.
        return 18_000_000


def _run_eval(
    units: list[str],
    engine,
    backend_kind: str,
    dataset_paths: dict[str, dict[str, Path]],
    adapters: dict[str, dict[str, object]],
    eval_out_path: Path,
    max_new_tokens: int,
) -> None:
    """Run the real Weave eval and write ``eval_results.json``."""
    from weaveself.contracts.training_pair import read_training_pairs

    def score_fn(prompt: str, target: str, adapter_id: str | None) -> float:
        return float(engine.score(prompt, target, adapter_id).perplexity)

    weave = WeaveEval(score_fn=score_fn)

    # Build held-out sets; inject the train completions as the context-memory
    # baseline examples (zero extra weight cost) for the competitive comparison.
    heldouts: list[HeldOutSet] = []
    adapter_ids: dict[str, str] = {}
    for unit in units:
        rows = read_training_pairs(str(dataset_paths[unit]["heldout"]))
        train_rows = read_training_pairs(str(dataset_paths[unit]["train"]))
        context_examples = [
            f"{r.prompt} {r.completion}" for r in train_rows[:3]
        ]
        heldouts.append(
            HeldOutSet(
                unit_label=unit, rows=rows, context_examples=context_examples
            )
        )
        adapter_ids[unit] = str(adapters[unit]["adapter_id"])

    # Base-vs-adapter generation examples (one per Unit) to show the adapter
    # actually changes the model's output.
    examples = []
    for unit in units:
        held_rows = read_training_pairs(str(dataset_paths[unit]["heldout"]))
        sample = held_rows[0]
        adapter_id = adapter_ids[unit]
        base_gen = engine.generate(sample.prompt, None, max_new_tokens)
        adapter_gen = engine.generate(sample.prompt, adapter_id, max_new_tokens)
        examples.append(
            {
                "prompt": sample.prompt,
                "base": base_gen.text,
                "adapter": adapter_gen.text,
                "reference": sample.completion,
            }
        )
        changed = base_gen.text != adapter_gen.text
        print(
            f"[prepare_demo] example unit={unit} adapter_changed_output={changed}",
            flush=True,
        )
        print(f"    prompt : {sample.prompt}", flush=True)
        print(f"    base   : {base_gen.text!r}", flush=True)
        print(f"    adapter: {adapter_gen.text!r}", flush=True)

    nkt_size = max(int(a["size_bytes"]) for a in adapters.values())
    lora_size = _estimate_lora_size_bytes(engine, backend_kind)

    results = weave.run(
        heldouts,
        adapter_ids,
        nktmirror_size_bytes=nkt_size,
        lora_size_bytes=lora_size,
        examples=examples,
        out_path=str(eval_out_path),
    )
    print(
        f"[prepare_demo] eval complete: "
        f"perplexity base={results.perplexity.base:.4f} "
        f"adapter={results.perplexity.adapter:.4f} "
        f"context_memory={results.perplexity.context_memory:.4f}",
        flush=True,
    )
    print(
        f"[prepare_demo] size_bytes nktmirror={results.size_bytes.nktmirror} "
        f"lora={results.size_bytes.lora}",
        flush=True,
    )
    print(f"[prepare_demo] wrote {eval_out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Prepare a real WeaveSelf demo.")
    parser.add_argument(
        "--units",
        nargs="+",
        default=list(DEMO_DATA.keys()),
        choices=list(DEMO_DATA.keys()),
        help="Demo Units to prepare (default: all).",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("TORCH_DEVICE", "").strip() or _default_device(),
        help="Torch device for the real model (default: cuda if available).",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("MODEL_DTYPE", "bfloat16").strip() or "bfloat16",
        help="Model dtype: float32 | float16 | bfloat16 (default: bfloat16).",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip(),
        help="Instruct Base_Model id.",
    )
    parser.add_argument(
        "--adapters-dir",
        default=None,
        help="Adapter output dir (default: ADAPTERS_DIR from .env).",
    )
    parser.add_argument(
        "--eval-out",
        default=None,
        help="eval_results.json output path (default: EVAL_RESULTS_PATH from .env).",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "").strip() or None,
        help="Redis URL (default: REDIS_URL from .env; file fallback if unreachable).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Max new tokens for the base-vs-adapter examples (kept small).",
    )
    args = parser.parse_args(argv)

    units = list(args.units)
    adapters_dir = (
        _resolve_path(args.adapters_dir)
        if args.adapters_dir
        else _resolve_path(os.environ.get("ADAPTERS_DIR", "./data/adapters").strip())
    )
    eval_out = (
        _resolve_path(args.eval_out)
        if args.eval_out
        else _resolve_path(
            os.environ.get("EVAL_RESULTS_PATH", "./data/eval_results.json").strip()
        )
    )
    data_dir = _repo_root() / "data" / "demo"
    redis_file = _repo_root() / "data" / "redis_store.json"

    print(
        f"[prepare_demo] units={units} base_model={args.base_model} "
        f"device={args.device} dtype={args.dtype}",
        flush=True,
    )
    print(f"[prepare_demo] adapters_dir={adapters_dir}", flush=True)
    print(f"[prepare_demo] eval_out={eval_out}", flush=True)

    # 1. datasets
    dataset_paths = _write_datasets(units, data_dir)
    # 2. real adapters
    adapters = _train_adapters(units, dataset_paths, adapters_dir, args.base_model)
    # 3. store in Redis
    _store_in_redis(units, adapters, adapters_dir, args.redis_url, redis_file)
    # 4. resident engine (real model, single load)
    engine, backend_kind, note = _build_engine(
        args.base_model, adapters_dir, args.device, args.dtype
    )
    print(
        f"[prepare_demo] engine backend={backend_kind} "
        f"base_model_load_count={engine.base_model_load_count} ({note})",
        flush=True,
    )
    # 5. real eval + artifact
    _run_eval(
        units,
        engine,
        backend_kind,
        dataset_paths,
        adapters,
        eval_out,
        args.max_new_tokens,
    )
    print("[prepare_demo] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
