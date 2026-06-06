"""Controller lifecycle: train, compose/subtract, inspect, pair, generate.

All operations wrap the real ntkmirror API over the shared frozen model.
Controllers persist as ~100 KB .pt files in config.CONTROLLER_DIR.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

import torch
from ntkmirror import ForwardFineTuner, SignedLogMaskState
from ntkmirror.compose import compose_states, dense_gate_vector, gate_values, pair_report
from ntkmirror.data import Example

from . import config
from .model import device, get_model


class ControllerNotFound(KeyError):
    """Raised when a controller id has no .pt artifact on disk."""


def _new_tuner() -> ForwardFineTuner:
    model, tok = get_model()
    return ForwardFineTuner(model, tok, gates=config.GATES,
                            max_log_gate=config.MAX_LOG_GATE, layers="all")


def controller_path(controller_id: str) -> Path:
    p = config.CONTROLLER_DIR / f"{controller_id}.pt"
    if not p.exists():
        raise ControllerNotFound(controller_id)
    return p


def artifact_bytes(controller_id: str) -> int:
    return controller_path(controller_id).stat().st_size


def list_controllers() -> list[dict]:
    return [{"controller_id": p.stem, "artifact_bytes": p.stat().st_size}
            for p in sorted(config.CONTROLLER_DIR.glob("*.pt"))]


def train(task_id: str, examples: list[dict], *, steps: int = 240, lr: float = 5e-3,
          batch_size: int = 8, max_length: int = 256) -> dict:
    exs = [Example(str(e["prompt"]), str(e["completion"])) for e in examples]
    tuner = _new_tuner()
    t0 = time.perf_counter()
    stats = tuner.fit(exs, steps=steps, lr=lr, batch_size=batch_size,
                      max_length=max_length, verbose=False)
    controller_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
    tuner.save(config.CONTROLLER_DIR / f"{controller_id}.pt")
    return {
        "controller_id": controller_id,
        "n_gates": int(stats["selected_gates"]),
        "loss_first": stats["loss_first"],
        "loss_last": stats["loss_last"],
        "train_seconds": round(time.perf_counter() - t0, 2),
        "artifact_bytes": artifact_bytes(controller_id),
    }


def compose(controller_ids: list[str], weights: list[float], *, new_id: str | None = None) -> dict:
    states = [SignedLogMaskState.load(controller_path(c)) for c in controller_ids]
    composed = compose_states(states, weights=weights)
    new_id = new_id or f"compose-{uuid.uuid4().hex[:8]}"
    composed.save(config.CONTROLLER_DIR / f"{new_id}.pt")
    return {
        "controller_id": new_id,
        "n_gates": int(composed.n_gates),
        "artifact_bytes": artifact_bytes(new_id),
        "from": controller_ids,
        "weights": weights,
    }


def generator_for(controller_id: str | None) -> Callable[[str, int], str]:
    """Return gen(prompt, max_new_tokens) -> completion. Base model if id is None.
    Greedy/deterministic. A controller is loaded once and reused across calls."""
    model, tok = get_model()
    if controller_id is None:
        def gen(prompt: str, max_new_tokens: int) -> str:
            enc = tok(prompt, return_tensors="pt").to(device())
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen

    tuner = _new_tuner()
    tuner.load(controller_path(controller_id))

    def gen(prompt: str, max_new_tokens: int) -> str:
        full = tuner.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return full[len(prompt):] if full.startswith(prompt) else full
    return gen


def inspect_controller(controller_id: str, *, dense: bool = False) -> dict:
    state = SignedLogMaskState.load(controller_path(controller_id))
    gv = gate_values(state)
    payload = {
        "controller_id": controller_id,
        "n_gates": int(state.n_gates),
        "n_layers": int(state.n_layers),
        "hidden_size": int(state.hidden_size),
        "max_log_gate": float(state.max_log_gate),
        "model_name": state.model_name,
        "artifact_bytes": artifact_bytes(controller_id),
        "gates": [{"layer": l, "channel": c, "value": v} for (l, c), v in gv.items()],
    }
    if dense:
        payload["dense_vector"] = dense_gate_vector(state).tolist()
    return payload


def pair(a: str, b: str) -> dict:
    sa = SignedLogMaskState.load(controller_path(a))
    sb = SignedLogMaskState.load(controller_path(b))
    return {"a": a, "b": b, **pair_report(sa, sb)}
