"""NKT-Mirror ``train_adapter`` training loop (Track A / Requirement 9).

This module fits a tiny set of per-channel **activation gates** (NKT-Mirror,
*not* LoRA) on top of a frozen instruct Base_Model and serializes them as an
Adapter_File pair via :func:`weaveself.contracts.write_adapter_file`.

Design constraints honored here (design.md "Train_Adapter", Requirements 6.2,
6.3, 6.4, 9.1, 9.2):

* **Activation gating, not LoRA.** The trainable parameters are ~5,000
  per-channel gate multipliers centered near ``1.0``. The base model weights are
  never read or written here — only gate tensors are produced (Req 6.3).
* **Frozen base.** No base-weight tensors are emitted or mutated; the adapter is
  purely the gate set applied on the resident base at serve time (Req 6.3).
* **Bounded size.** ~5,000 float32 gates serialize to well under the 200,000-byte
  ceiling, and the written metadata ``size_bytes`` reflects the real serialized
  size (Req 6.4) because :func:`write_adapter_file` stamps it from disk.
* **Deterministic.** For a fixed dataset + arguments the produced gates,
  ``adapter_id``, and metadata are identical across runs, so metadata/property
  tests are stable. Randomness is seeded from the dataset content only.

torch / transformers are **optional** (the heavy ``serving`` extra). The default
training path is pure numpy and needs no GPU or model download, so this function
runs in CI and tests. If torch happens to be installed an optional refinement
pass may sharpen the gates, but the numpy path is the default and is what tests
exercise.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from weaveself.contracts import read_training_pairs, write_adapter_file
from weaveself.contracts.errors import MissingFieldError
from weaveself.contracts.training_pair import TrainingPair
from weaveself.training.errors import DatasetNotReadable, InsufficientTrainingData

__all__ = ["train_adapter", "GATE_LAYERS", "GATE_CHANNELS", "TOTAL_GATE_PARAMS"]

# ~5,000 trainable activation-gating parameters laid out as a small stack of
# per-channel gate vectors. 5 layers x 1000 channels == 5,000 params exactly.
GATE_LAYERS: int = 5
GATE_CHANNELS: int = 1000
TOTAL_GATE_PARAMS: int = GATE_LAYERS * GATE_CHANNELS

# Gates are multiplicative and centered at 1.0; deviations are bounded so a
# trained adapter only *steers* activations (style/preferences) rather than
# overwriting them.
_GATE_CENTER = 1.0
_GATE_SPREAD = 0.1

# Deterministic ISO-8601 base instant for ``trained_at`` (offset by day_index)
# so metadata is stable for a fixed dataset.
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _row_text(row: TrainingPair) -> str:
    """Stable text encoding of a single training row."""
    return f"{row.prompt}\x00{row.completion}\x00{row.unit_label}"


def _dataset_digest(rows: list[TrainingPair], unit_label: str, unit_type: str) -> str:
    """Deterministic content hash over the consumed rows and unit identity."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(unit_type.encode("utf-8"))
    hasher.update(b"\x1f")
    hasher.update(unit_label.encode("utf-8"))
    hasher.update(b"\x1e")
    for row in rows:
        hasher.update(_row_text(row).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _fit_gates(rows: list[TrainingPair], seed_material: str) -> np.ndarray:
    """Deterministically fit the flat ~5K gate vector from the training rows.

    This is a lightweight gate-fitting routine over per-channel statistics: each
    row contributes a deterministic pseudo-activation vector (seeded from the
    row's own text so order/content fully determines the result), and the gates
    are the row-averaged statistics squashed into a bounded multiplier around
    1.0. No base-model weights are touched.
    """
    accum = np.zeros(TOTAL_GATE_PARAMS, dtype=np.float64)

    # A global component seeded from unit identity so two units trained on
    # similarly-shaped data still differ.
    base_seed = int.from_bytes(
        hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest(),
        "big",
    )
    base_rng = np.random.default_rng(base_seed)
    accum += 0.25 * base_rng.standard_normal(TOTAL_GATE_PARAMS)

    for row in rows:
        row_seed = int.from_bytes(
            hashlib.blake2b(_row_text(row).encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        row_rng = np.random.default_rng(row_seed)
        accum += row_rng.standard_normal(TOTAL_GATE_PARAMS)

    mean = accum / float(len(rows) + 1)  # +1 for the base component
    gates = _GATE_CENTER + _GATE_SPREAD * np.tanh(mean)
    return gates.astype(np.float32)


def _gate_tensors(gates_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Reshape the flat gate vector into named per-layer gate tensors."""
    reshaped = gates_flat.reshape(GATE_LAYERS, GATE_CHANNELS)
    return {
        f"model.layers.{i}.mlp.gate": np.ascontiguousarray(reshaped[i])
        for i in range(GATE_LAYERS)
    }


def train_adapter(
    dataset_path: str,
    unit_label: str,
    unit_type: str,
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    out_dir: str | None = None,
    day_index: int = 0,
    method: str | None = None,
) -> str:
    """Train an NKT-Mirror gate set on the frozen instruct base and write an Adapter_File.

    ``method`` selects the trainer:

    * ``"nkt"`` (or env ``WEAVESELF_TRAIN=nkt``/``real``) runs the REAL
      gradient-descent NKT-Mirror trainer (:func:`weaveself.training.nkt_trainer.train_adapter_nkt`)
      on the frozen instruct model — this is the production path that actually
      personalizes.
    * ``"fast"`` (the default, used by CI/unit tests) runs the lightweight
      dependency-free gate fitter below: deterministic, numpy-only, no model
      download. It is a TEST FIXTURE, not a real personalization trainer, and
      must not be used for the demo.

    The written metadata ``train_rows`` equals the number of consumed rows
    (Req 9.1) and ``unit_label`` / ``unit_type`` equal the supplied arguments
    (Req 9.1, 9.2). ``size_bytes`` is stamped from the real serialized size and
    is bounded well under 200,000 bytes (Req 6.4).

    Input error handling: an unreadable/malformed ``dataset_path`` raises
    :class:`DatasetNotReadable`; a zero-row dataset raises
    :class:`InsufficientTrainingData`; no Adapter_File is written in either case
    (Req 9.4, 9.5).
    """
    import os as _os

    resolved = (method or _os.environ.get("WEAVESELF_TRAIN", "fast")).strip().lower()
    if resolved in ("nkt", "real", "nktmirror"):
        from weaveself.training.nkt_trainer import train_adapter_nkt

        return train_adapter_nkt(
            dataset_path,
            unit_label,
            unit_type,
            base_model=base_model,
            out_dir=out_dir,
            day_index=day_index,
        )

    # ----- fast/test fixture path (numpy, no model) -----
    # Req 9.4: a missing/unreadable/malformed dataset path must raise a typed
    # error naming the path before anything is written to out_dir.
    try:
        rows = read_training_pairs(dataset_path)
    except (DatasetNotReadable, InsufficientTrainingData):
        raise
    except FileNotFoundError as exc:
        raise DatasetNotReadable(str(dataset_path), reason=str(exc)) from exc
    except MissingFieldError as exc:
        # A row missing a required Training_Pair field => the file is not a
        # valid training-pair file.
        raise DatasetNotReadable(str(dataset_path), reason=str(exc)) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        # Garbage / unparseable file contents.
        raise DatasetNotReadable(str(dataset_path), reason=str(exc)) from exc
    except OSError as exc:
        # Unreadable for other filesystem reasons (permissions, is-a-directory).
        raise DatasetNotReadable(str(dataset_path), reason=str(exc)) from exc

    # Req 9.5: a readable but empty dataset is insufficient; write nothing.
    if not rows:
        raise InsufficientTrainingData(str(dataset_path))

    train_rows = len(rows)
    digest = _dataset_digest(rows, unit_label, unit_type)

    gates_flat = _fit_gates(rows, seed_material=f"{digest}:{day_index}")
    gate_tensors = _gate_tensors(gates_flat)

    # Deterministic adapter id + timestamp for a fixed dataset.
    adapter_id = digest[:16]
    trained_at = (_EPOCH + timedelta(days=int(day_index))).isoformat()

    if out_dir is None:
        out_dir = str(Path(dataset_path).resolve().parent / "adapters")

    metadata = {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "unit_type": unit_type,
        "unit_label": unit_label,
        "train_rows": train_rows,
        "trained_at": trained_at,
        "day_index": int(day_index),
        # Stamped to the real serialized size by write_adapter_file.
        "size_bytes": 0,
    }

    blob_path, _meta_path = write_adapter_file(out_dir, metadata, gate_tensors)
    return str(blob_path)
