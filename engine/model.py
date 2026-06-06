"""Single frozen base model, loaded lazily and shared across all controllers."""
from __future__ import annotations

from threading import Lock

import torch

from . import config

_model = None
_tok = None
_lock = Lock()


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    """Load (frozen) base model + tokenizer exactly once; return (model, tok)."""
    global _model, _tok
    if _model is not None:
        return _model, _tok
    with _lock:
        if _model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.bfloat16 if device() == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME, torch_dtype=dtype
            ).to(device())
            model.eval()
            _model, _tok = model, tok
    return _model, _tok


def is_loaded() -> bool:
    return _model is not None
