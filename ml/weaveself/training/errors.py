"""Typed Train_Adapter input errors (Track A / Requirement 9.4, 9.5).

These errors back the training-input contract in design.md "Training errors
(Track A)":

* :class:`DatasetNotReadable` — the supplied ``dataset_path`` does not resolve
  to a readable training-pair file (missing file, unreadable file, or a
  malformed / unparseable / schema-invalid file). The message NAMES the
  offending ``dataset_path`` (Req 9.4).
* :class:`InsufficientTrainingData` — the dataset is readable but contains zero
  training rows (Req 9.5).

In both cases :func:`weaveself.training.train_adapter` raises before any
Adapter_File is written, so a failed train leaves ``out_dir`` untouched.
"""

from __future__ import annotations


class TrainAdapterError(ValueError):
    """Base class for ``train_adapter`` input errors.

    Subclasses ``ValueError`` so existing callers that only caught the previous
    plain ``ValueError`` zero-row guard continue to work.
    """


class DatasetNotReadable(TrainAdapterError):
    """Raised when ``dataset_path`` is not a readable training-pair file.

    The message names the offending ``dataset_path`` so the caller can report
    exactly which path failed (Requirement 9.4). An optional ``reason`` carries
    the underlying cause (e.g. the original ``FileNotFoundError`` or JSON parse
    error text).
    """

    def __init__(self, dataset_path: str, reason: str | None = None) -> None:
        self.dataset_path = dataset_path
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"dataset path is not readable: '{dataset_path}'{detail}"
        )


class InsufficientTrainingData(TrainAdapterError):
    """Raised when a readable dataset contains zero training rows (Req 9.5).

    No Adapter_File is produced in this case. The message names the offending
    ``dataset_path``; an optional ``reason`` carries extra context.
    """

    def __init__(self, dataset_path: str, reason: str | None = None) -> None:
        self.dataset_path = dataset_path
        self.reason = reason
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"insufficient training data: dataset '{dataset_path}' has zero rows"
            f"{detail}"
        )
