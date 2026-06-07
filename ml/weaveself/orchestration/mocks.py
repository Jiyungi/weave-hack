"""Mock_Dependencies for Track B (Requirement 22.2, 16.1).

Until the Integration_Milestone, Track B runs the batch graph against mocks of
the interfaces it consumes but does not own:

* :class:`MockTrainAdapter` — stands in for Track A's ``train_adapter``; it
  writes a real, schema-conformant Adapter_File pair (so ``store`` and ``eval``
  exercise the real contracts) without doing any model training.
* :class:`MockInference` — stands in for the Track A Inference_API ``/score``;
  it returns deterministic perplexities where a Unit's own adapter scores that
  Unit's text lowest, yielding a sensible diagonal confusion matrix.
* :class:`MockRedisClient` — stands in for the Track C Redis_Client_API blob /
  metadata / interaction store.

These let Track B emit a real ``eval_results.json`` with a confusion matrix with
no dependency on Tracks A or C (Req 16).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from weaveself.contracts.adapter_file import (
    AdapterMetadata,
    read_adapter_file,
    write_adapter_file,
)
from weaveself.contracts.training_pair import read_training_pairs


def _stable_unit_float(seed: str) -> float:
    """Deterministic float in [0, 1) from a string (no global RNG state)."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class MockTrainAdapter:
    """A ``train_adapter(dataset_path, unit_label, unit_type) -> adapter_path``
    mock that writes a real Adapter_File pair.

    Tracks how many times it was invoked so tests can assert that live chat
    never triggers training (Property 19).
    """

    def __init__(self, out_dir: str | Path, *, day_index: int = 0) -> None:
        self.out_dir = Path(out_dir)
        self.day_index = day_index
        self.call_count = 0
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, dataset_path: str, unit_label: str, unit_type: str) -> str:
        self.call_count += 1
        self.calls.append((dataset_path, unit_label, unit_type))

        rows = read_training_pairs(dataset_path)
        if not rows:
            # Mirror Track A's InsufficientTrainingData contract shape.
            raise ValueError(f"insufficient training data: {dataset_path}")

        adapter_id = f"{unit_label}-d{self.day_index}"
        # Tiny deterministic gate tensor seeded by the unit so different units
        # get visibly different adapters (well under the 200 KB bound).
        seed = int(_stable_unit_float(adapter_id) * (1 << 32))
        rng = np.random.default_rng(seed)
        gates = {"layer.0.gate": rng.standard_normal(64).astype(np.float32)}

        meta = AdapterMetadata(
            adapter_id=adapter_id,
            base_model="mock-instruct",
            unit_type=unit_type,  # type: ignore[arg-type]
            unit_label=unit_label,
            train_rows=len(rows),
            trained_at="2024-01-01T00:00:00Z",
            day_index=self.day_index,
            size_bytes=0,
        )
        blob_path, _meta_path = write_adapter_file(self.out_dir, meta, gates)
        return str(blob_path)


class MockInference:
    """A deterministic ``/score`` mock.

    ``score(prompt, target, adapter_id)`` returns a perplexity that is lower
    when the adapter's Unit token appears in the scored text, so a Unit's own
    adapter wins on that Unit's held-out set (diagonal confusion matrix) and the
    adapter beats the base (personalization passes).
    """

    def __init__(self, base_perplexity: float = 20.0) -> None:
        self.base_perplexity = base_perplexity

    @staticmethod
    def _unit_from_adapter(adapter_id: str) -> str:
        # MockTrainAdapter encodes ids as "<unit_label>-d<day>".
        return adapter_id.rsplit("-d", 1)[0]

    def score(self, prompt: str, target: str, adapter_id: str | None) -> float:
        text = f"{prompt} {target}"
        jitter = _stable_unit_float(text) * 0.5  # small deterministic spread
        if adapter_id is None:
            return self.base_perplexity + jitter
        unit = self._unit_from_adapter(adapter_id)
        matches = unit in text
        # Matching adapter lowers perplexity well below base; a mismatching
        # adapter helps only slightly, staying above the matching adapter.
        delta = 8.0 if matches else 1.0
        return max(0.1, self.base_perplexity - delta + jitter)

    # Convenience alias matching the ScoreFn signature directly.
    def __call__(self, prompt: str, target: str, adapter_id: str | None) -> float:
        return self.score(prompt, target, adapter_id)


class MockRedisClient:
    """In-memory stand-in for the Track C Redis_Client_API."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.meta: dict[str, dict] = {}
        self.interactions: dict[str, list[dict]] = defaultdict(list)

    def store_adapter(
        self, meta: Mapping[str, object] | AdapterMetadata, blob: bytes
    ) -> None:
        if isinstance(meta, AdapterMetadata):
            meta_dict = meta.model_dump()
        else:
            meta_dict = dict(meta)
        adapter_id = str(meta_dict["adapter_id"])
        self.meta[adapter_id] = meta_dict
        self.blobs[adapter_id] = bytes(blob)

    def fetch_meta(self, adapter_id: str) -> dict:
        return self.meta[adapter_id]

    def fetch_blob(self, adapter_id: str) -> bytes:
        return self.blobs[adapter_id]

    def append_interaction(self, unit_label: str, interaction: dict) -> None:
        self.interactions[unit_label].append(interaction)


def store_adapter_from_path(redis_client: MockRedisClient, adapter_path: str) -> str:
    """Read an Adapter_File pair from disk and store it through the client.

    Returns the stored ``adapter_id``. Used by the batch ``store`` node.
    """
    path = Path(adapter_path)
    directory = path.parent
    # Filenames are adapter_<id>.safetensors.
    adapter_id = path.stem.removeprefix("adapter_")
    meta, _gates = read_adapter_file(directory, adapter_id)
    blob = path.read_bytes()
    redis_client.store_adapter(meta, blob)
    return adapter_id
