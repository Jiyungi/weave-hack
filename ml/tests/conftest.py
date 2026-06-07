"""Shared pytest fixtures for the WeaveSelf Python test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_metadata() -> dict:
    """A complete, valid eight-field Adapter_File metadata dict."""
    return {
        "adapter_id": "abc123",
        "base_model": "Qwen2.5-1.5B-Instruct",
        "unit_type": "user",
        "unit_label": "alice",
        "train_rows": 42,
        "trained_at": "2024-01-15T03:00:00Z",
        "day_index": 7,
        "size_bytes": 0,
    }


@pytest.fixture
def sample_gates() -> dict:
    """A small set of gate tensors resembling NKT-Mirror per-channel gates."""
    rng = np.random.default_rng(0)
    return {
        "layer.0.gate": rng.standard_normal(128).astype(np.float32),
        "layer.1.gate": rng.standard_normal(128).astype(np.float32),
    }


@pytest.fixture
def sample_training_pair() -> dict:
    """A complete, valid three-field Training_Pair row."""
    return {
        "prompt": "What's a good weekend project?",
        "completion": "Try building a small CLI tool in Rust.",
        "unit_label": "alice",
    }
