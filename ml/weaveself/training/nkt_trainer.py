"""Real NKT-Mirror activation-gating trainer (Track A / Req 6, 9).

This is the genuine training loop (gradient descent), not a placeholder. It
learns a small set of **per-channel multiplicative gates** applied to the output
of selected decoder-layer MLP blocks of a FROZEN instruct Base_Model, so the
adapter *steers* the model toward a Unit's style/preferences (NKT-Mirror, not
LoRA). Only the gate vectors are trainable; no base weight is ever updated
(Req 6.3).

Gate layout
-----------
For each selected layer ``i`` a gate vector ``g_i`` of length ``hidden_size`` is
multiplied into the output of ``model.model.layers[i].mlp`` via a forward hook.
Gates are stored keyed ``model.layers.{i}.mlp.gate`` with shape
``(hidden_size,)`` — exactly the key/shape the serving
:class:`~weaveself.serving.backend.HFBackend` expects, so a trained adapter is
byte-compatible with serving.

Training
--------
Each Training_Pair is rendered with the tokenizer chat template
(``user: prompt`` -> ``assistant: completion``); the loss is the causal-LM NLL
on the **completion tokens only** (the prompt is masked with ``-100``), so the
gates learn to make the Unit's own answers more likely. Optimized with AdamW
over the gate vectors only. Gradient checkpointing + ``use_cache=False`` keep it
within a 6 GB GPU for a 1.5B model.

Requires the optional ``serving`` extra (``torch`` + ``transformers``); imports
are lazy so this module is import-safe without them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from weaveself.contracts import read_training_pairs, write_adapter_file
from weaveself.contracts.training_pair import TrainingPair


def _select_gate_layers(num_layers: int, n_gates: int) -> list[int]:
    """Evenly spread ``n_gates`` gate layers across the model depth."""
    n_gates = max(1, min(n_gates, num_layers))
    if n_gates == 1:
        return [num_layers // 2]
    step = (num_layers - 1) / (n_gates - 1)
    return sorted({int(round(k * step)) for k in range(n_gates)})


def train_adapter_nkt(
    dataset_path: str,
    unit_label: str,
    unit_type: str,
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    out_dir: str | None = None,
    day_index: int = 0,
    device: str | None = None,
    dtype: str = "bfloat16",
    n_gate_layers: int = 6,
    epochs: int = 4,
    lr: float = 2e-2,
    max_len: int = 192,
    weight_decay: float = 0.0,
    identity_reg: float = 0.1,
    init_gates: dict | None = None,
    anchor_gates: dict | None = None,
    verbose: bool = True,
) -> str:
    """Train real NKT-Mirror gates on ``dataset_path`` and write an Adapter_File.

    Continual learning (data-free): pass ``init_gates`` to **warm-start** from a
    previous adapter's gates (so the model continues from what it already knew),
    and ``anchor_gates`` to **regularize toward those gates** instead of toward
    identity — that preserves prior days in the weights without replaying old
    data. Both are dicts keyed ``model.layers.{i}.mlp.gate`` (the Adapter_File
    layout). When neither is given, training starts at identity and regularizes
    toward identity (single-shot behavior).

    Returns the ``.safetensors`` adapter path. Raises
    :class:`~weaveself.training.errors.DatasetNotReadable` /
    :class:`~weaveself.training.errors.InsufficientTrainingData` on bad input.
    """
    from weaveself.training.errors import (
        DatasetNotReadable,
        InsufficientTrainingData,
    )
    from weaveself.contracts.errors import MissingFieldError

    try:
        rows = read_training_pairs(dataset_path)
    except (DatasetNotReadable, InsufficientTrainingData):
        raise
    except (FileNotFoundError, MissingFieldError, ValueError, OSError) as exc:
        raise DatasetNotReadable(str(dataset_path), reason=str(exc)) from exc
    if not rows:
        raise InsufficientTrainingData(str(dataset_path))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.bfloat16)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch_dtype = torch.float32  # bf16 training on CPU is impractical

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Gates are applied via hooks (not module inputs), so make the input
        # embeddings require grad too, keeping the checkpointed graph connected.
        model.enable_input_require_grads()
    except Exception:
        pass
    model.config.use_cache = False

    inner = model.model  # Qwen2Model: has .layers
    num_layers = len(inner.layers)
    hidden = int(model.config.hidden_size)
    gate_layers = _select_gate_layers(num_layers, n_gate_layers)

    def _gate_key(i: int) -> str:
        return f"model.layers.{i}.mlp.gate"

    def _init_tensor(i: int) -> "torch.Tensor":
        # Warm-start from a prior adapter's gate when shapes match, else identity.
        if init_gates is not None:
            arr = init_gates.get(_gate_key(i))
            if arr is not None and np.asarray(arr).shape == (hidden,):
                return torch.tensor(np.asarray(arr), device=device, dtype=torch.float32)
        return torch.ones(hidden, device=device, dtype=torch.float32)

    # Learnable gate parameters (warm-started from prior gates when provided).
    gates: dict[int, torch.nn.Parameter] = {
        i: torch.nn.Parameter(_init_tensor(i)) for i in gate_layers
    }

    # Anchor tensors the regularizer pulls toward: prior gates (continual
    # learning) when provided, else identity 1.0 (single-shot "steer").
    anchors: dict[int, "torch.Tensor"] = {}
    for i in gate_layers:
        if anchor_gates is not None and anchor_gates.get(_gate_key(i)) is not None and np.asarray(anchor_gates[_gate_key(i)]).shape == (hidden,):
            anchors[i] = torch.tensor(np.asarray(anchor_gates[_gate_key(i)]), device=device, dtype=torch.float32)
        else:
            anchors[i] = torch.ones(hidden, device=device, dtype=torch.float32)

    handles = []

    def _make_hook(param: torch.nn.Parameter):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                return (out[0] * param.to(out[0].dtype),) + tuple(out[1:])
            return out * param.to(out.dtype)

        return hook

    for i in gate_layers:
        handles.append(inner.layers[i].mlp.register_forward_hook(_make_hook(gates[i])))

    # Build tokenized, completion-masked training tensors.
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row in rows:
        if tok.chat_template:
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": row.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = row.prompt
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
        comp_ids = tok(row.completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([prompt_ids, comp_ids])[:max_len]
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = -100  # mask the prompt
        examples.append((input_ids, labels))

    opt = torch.optim.AdamW(list(gates.values()), lr=lr, weight_decay=weight_decay)

    last_loss = math.nan
    for epoch in range(epochs):
        total = 0.0
        for input_ids, labels in examples:
            opt.zero_grad()
            ids = input_ids.unsqueeze(0).to(device)
            lbl = labels.unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=lbl)
            loss = out.loss
            # Anchor regularizer: pull gates toward the anchor (prior gates for
            # continual learning, else identity). Preserves past learning and
            # prevents overfitting the day's few rows.
            if identity_reg > 0.0:
                penalty = identity_reg * sum(
                    ((gates[i] - anchors[i]) ** 2).mean() for i in gate_layers
                )
                loss = loss + penalty
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        last_loss = total / max(1, len(examples))
        if verbose:
            print(f"[nkt] {unit_label} epoch {epoch + 1}/{epochs} loss={last_loss:.4f}", flush=True)

    for h in handles:
        h.remove()

    gate_tensors = {
        f"model.layers.{i}.mlp.gate": gates[i].detach().to(torch.float32).cpu().numpy()
        for i in gate_layers
    }

    # Free GPU memory promptly.
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    from datetime import datetime, timedelta, timezone
    import hashlib

    digest = hashlib.blake2b(
        f"{unit_type}:{unit_label}:{day_index}:{len(rows)}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    adapter_id = f"{unit_label}-d{day_index}-{digest[:8]}"
    trained_at = (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(day_index))).isoformat()

    if out_dir is None:
        out_dir = str(Path(dataset_path).resolve().parent / "adapters")

    metadata = {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "unit_type": unit_type,
        "unit_label": unit_label,
        "train_rows": len(rows),
        "trained_at": trained_at,
        "day_index": int(day_index),
        "size_bytes": 0,
    }
    blob_path, _ = write_adapter_file(out_dir, metadata, gate_tensors)
    if verbose:
        print(f"[nkt] wrote {blob_path} (final loss={last_loss:.4f})", flush=True)
    return str(blob_path)
