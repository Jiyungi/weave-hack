"""Track 0 — shared data/interface contracts.

Re-exports the Adapter_File contract surface (Requirement 1), the
Training_Pair / Held_Out_Set schema contract (Requirement 4), the
Inference_API request/response schema contract (Requirement 2), and the
Eval_Results artifact schema contract (Requirement 5).
"""

from weaveself.contracts.errors import MissingFieldError
from weaveself.contracts.adapter_file import (
    AdapterMetadata,
    ADAPTER_METADATA_FIELDS,
    validate_metadata,
    write_adapter_file,
    read_adapter_file,
    adapter_blob_filename,
    adapter_meta_filename,
)
from weaveself.contracts.training_pair import (
    TrainingPair,
    TRAINING_PAIR_FIELDS,
    validate_training_pair,
    read_training_pairs,
    write_training_pairs,
)
from weaveself.contracts.inference_api import (
    GenerateRequest,
    GenerateResponse,
    ScoreRequest,
    ScoreResponse,
    AdaptersResponse,
    TrainRequest,
    TrainResponse,
)
from weaveself.contracts.eval_results import (
    Perplexity,
    ConfusionMatrix,
    SizeBytes,
    EvalExample,
    EvalResults,
    EVAL_RESULTS_FIELDS,
    InvalidConfusionMatrixError,
    validate_eval_results,
    read_eval_results,
    write_eval_results,
)

__all__ = [
    "MissingFieldError",
    "AdapterMetadata",
    "ADAPTER_METADATA_FIELDS",
    "validate_metadata",
    "write_adapter_file",
    "read_adapter_file",
    "adapter_blob_filename",
    "adapter_meta_filename",
    "TrainingPair",
    "TRAINING_PAIR_FIELDS",
    "validate_training_pair",
    "read_training_pairs",
    "write_training_pairs",
    "GenerateRequest",
    "GenerateResponse",
    "ScoreRequest",
    "ScoreResponse",
    "AdaptersResponse",
    "TrainRequest",
    "TrainResponse",
    "Perplexity",
    "ConfusionMatrix",
    "SizeBytes",
    "EvalExample",
    "EvalResults",
    "EVAL_RESULTS_FIELDS",
    "InvalidConfusionMatrixError",
    "validate_eval_results",
    "read_eval_results",
    "write_eval_results",
]
