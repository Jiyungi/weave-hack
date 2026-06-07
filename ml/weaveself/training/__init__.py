"""Track A — NKT-Mirror train_adapter."""

from weaveself.training.errors import (
    DatasetNotReadable,
    InsufficientTrainingData,
    TrainAdapterError,
)
from weaveself.training.trainer import (
    GATE_CHANNELS,
    GATE_LAYERS,
    TOTAL_GATE_PARAMS,
    train_adapter,
)

__all__ = [
    "train_adapter",
    "GATE_LAYERS",
    "GATE_CHANNELS",
    "TOTAL_GATE_PARAMS",
    "TrainAdapterError",
    "DatasetNotReadable",
    "InsufficientTrainingData",
]
