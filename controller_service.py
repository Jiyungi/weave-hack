"""
controller_service.py — Track A (Controller Engine) as a real HTTP service.

This is NOT a mock. It wraps the actual ntkmirror library and a real, frozen
base model. Tracks B (control plane) and C (UI) call these endpoints so they
never import torch — but every number that comes back is produced by the real
model and real controllers.

Endpoints (the Track-A contract):
  POST /train      fit a controller on {prompt, completion} examples
  POST /compose    add/subtract controllers   (weight < 0 == revoke)
  POST /execute    generate with a controller attached (or base if null)
  POST /evaluate   score a controller on held-out prompts
  GET  /inspect/{id}   gate vector + artifact size (the Redis-index payload)
  POST /pair       gate cosine / jaccard overlap between two controllers
  GET  /controllers    list saved controllers
  GET  /health     model + device status

Run on the Brev box:
  uvicorn controller_service:app --host 0.0.0.0 --port 8000

Controllers persist as ~100 KB .pt files in CONTROLLER_DIR. Commit that dir to
git so they survive a box delete; on a fresh box they reload instantly (no
re-fit).
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ntkmirror import ForwardFineTuner, SignedLogMaskState, load_jsonl_examples
from ntkmirror.compose import compose_states, gate_values, dense_gate_vector, pair_report
from ntkmirror.data import Example


MODEL_NAME = os.environ.get("PEFT_CMP_MODEL", "Qwen/Qwen2.5-7B")
CONTROLLER_DIR = Path(os.environ.get("CONTROLLER_DIR", "./controllers"))
GATES = int(os.environ.get("CTRL_GATES", "5000"))
MAX_LOG_GATE = float(os.environ.get("CTRL_MAX_LOG_GATE", "0.05"))
CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NTK-Mirror Controller Engine", version="0.1")

# --- single real model, loaded once, shared across all controllers ---------
_model = None
_tok = None
_load_lock = Lock()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_model():
    """Lazily load the real base model+tokenizer (frozen) exactly once."""
    global _model, _tok
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if _device() == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(_device())
        model.eval()
        _model, _tok = model, tok


def _new_tuner() -> ForwardFineTuner:
    _ensure_model()
    return ForwardFineTuner(_model, _tok, gates=GATES, max_log_gate=MAX_LOG_GATE, layers="all")


def _path(controller_id: str) -> Path:
    p = CONTROLLER_DIR / f"{controller_id}.pt"
    if not p.exists():
        raise HTTPException(404, f"controller not found: {controller_id}")
    return p


def _artifact_bytes(controller_id: str) -> int:
    return _path(controller_id).stat().st_size


# ===========================================================================
# request/response schemas
# ===========================================================================
class TrainReq(BaseModel):
    task_id: str
    examples: list[dict] = Field(..., description="[{prompt, completion}, ...]")
    steps: int = 240
    lr: float = 5e-3
    batch_size: int = 8
    max_length: int = 256


class ComposeReq(BaseModel):
    controller_ids: list[str]
    weights: list[float]
    new_id: str | None = None


class ExecuteReq(BaseModel):
    controller_id: str | None = None          # null == base model, no controller
    prompt: str
    max_new_tokens: int = 32


class EvalItem(BaseModel):
    prompt: str
    needle: str | None = None                  # substring that must appear
    gold: str | None = None                    # or exact-match target


class EvaluateReq(BaseModel):
    controller_id: str | None = None
    items: list[EvalItem]
    max_new_tokens: int = 32


class PairReq(BaseModel):
    a: str
    b: str


# ===========================================================================
# endpoints
# ===========================================================================
@app.get("/health")
def health():
    return {
        "model": MODEL_NAME,
        "device": _device(),
        "model_loaded": _model is not None,
        "gates": GATES,
        "max_log_gate": MAX_LOG_GATE,
        "controller_dir": str(CONTROLLER_DIR.resolve()),
    }


@app.get("/controllers")
def list_controllers():
    out = []
    for p in sorted(CONTROLLER_DIR.glob("*.pt")):
        out.append({"controller_id": p.stem, "artifact_bytes": p.stat().st_size})
    return {"controllers": out}


@app.post("/train")
def train(req: TrainReq):
    if not req.examples:
        raise HTTPException(400, "no examples provided")
    examples = [Example(str(e["prompt"]), str(e["completion"])) for e in req.examples]
    tuner = _new_tuner()
    t0 = time.perf_counter()
    stats = tuner.fit(examples, steps=req.steps, lr=req.lr,
                      batch_size=req.batch_size, max_length=req.max_length, verbose=False)
    controller_id = f"{req.task_id}-{uuid.uuid4().hex[:8]}"
    tuner.save(CONTROLLER_DIR / f"{controller_id}.pt")
    return {
        "controller_id": controller_id,
        "n_gates": int(stats["selected_gates"]),
        "loss_first": stats["loss_first"],
        "loss_last": stats["loss_last"],
        "train_seconds": round(time.perf_counter() - t0, 2),
        "artifact_bytes": _artifact_bytes(controller_id),
    }


@app.post("/compose")
def compose(req: ComposeReq):
    if len(req.controller_ids) != len(req.weights):
        raise HTTPException(400, "controller_ids and weights must be same length")
    states = [SignedLogMaskState.load(_path(cid)) for cid in req.controller_ids]
    composed = compose_states(states, weights=req.weights)
    new_id = req.new_id or f"compose-{uuid.uuid4().hex[:8]}"
    composed.save(CONTROLLER_DIR / f"{new_id}.pt")
    return {
        "controller_id": new_id,
        "n_gates": int(composed.n_gates),
        "artifact_bytes": _artifact_bytes(new_id),
        "from": req.controller_ids,
        "weights": req.weights,
    }


@app.post("/execute")
def execute(req: ExecuteReq):
    _ensure_model()
    if req.controller_id is None:
        enc = _tok(req.prompt, return_tensors="pt").to(_device())
        with torch.no_grad():
            out = _model.generate(**enc, max_new_tokens=req.max_new_tokens,
                                  do_sample=False, pad_token_id=_tok.pad_token_id)
        gen = _tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"controller_id": None, "completion": gen}
    tuner = _new_tuner()
    tuner.load(_path(req.controller_id))
    full = tuner.generate(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=False)
    gen = full[len(req.prompt):] if full.startswith(req.prompt) else full
    return {"controller_id": req.controller_id, "completion": gen}


@app.post("/evaluate")
def evaluate(req: EvaluateReq):
    _ensure_model()
    tuner = None
    if req.controller_id is not None:
        tuner = _new_tuner()
        tuner.load(_path(req.controller_id))

    def gen_one(prompt: str) -> str:
        if tuner is None:
            enc = _tok(prompt, return_tensors="pt").to(_device())
            with torch.no_grad():
                out = _model.generate(**enc, max_new_tokens=req.max_new_tokens,
                                      do_sample=False, pad_token_id=_tok.pad_token_id)
            return _tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        full = tuner.generate(prompt, max_new_tokens=req.max_new_tokens, do_sample=False)
        return full[len(prompt):] if full.startswith(prompt) else full

    results, hits = [], 0
    for item in req.items:
        text = gen_one(item.prompt)
        if item.needle is not None:
            ok = item.needle in text
        elif item.gold is not None:
            ok = item.gold.strip() == text.strip()
        else:
            raise HTTPException(400, "each item needs a 'needle' or 'gold'")
        hits += int(ok)
        results.append({"prompt": item.prompt, "output": text, "correct": ok})
    return {
        "controller_id": req.controller_id,
        "accuracy": hits / len(req.items),
        "n": len(req.items),
        "items": results,
    }


@app.get("/inspect/{controller_id}")
def inspect(controller_id: str, dense: bool = False):
    state = SignedLogMaskState.load(_path(controller_id))
    gv = gate_values(state)  # {(layer, channel): signed_log_value}
    payload = {
        "controller_id": controller_id,
        "n_gates": int(state.n_gates),
        "n_layers": int(state.n_layers),
        "hidden_size": int(state.hidden_size),
        "max_log_gate": float(state.max_log_gate),
        "model_name": state.model_name,
        "artifact_bytes": _artifact_bytes(controller_id),
        "gates": [{"layer": l, "channel": c, "value": v} for (l, c), v in gv.items()],
    }
    if dense:
        # Flat [n_layers * hidden_size] vector for Redis vector search.
        payload["dense_vector"] = dense_gate_vector(state).tolist()
    return payload


@app.post("/pair")
def pair(req: PairReq):
    a = SignedLogMaskState.load(_path(req.a))
    b = SignedLogMaskState.load(_path(req.b))
    report = pair_report(a, b)
    return {"a": req.a, "b": req.b, **report}
