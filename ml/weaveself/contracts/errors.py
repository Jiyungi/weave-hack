"""Typed contract-validation errors shared across tracks."""

from __future__ import annotations


class MissingFieldError(ValueError):
    """Raised when a required contract field is absent.

    The error message names the missing field so consuming components can
    report exactly which field was missing (Requirements 1.4, 4.4).
    """

    def __init__(self, field_name: str, context: str | None = None) -> None:
        self.field_name = field_name
        self.context = context
        location = f" in {context}" if context else ""
        super().__init__(f"Missing required field '{field_name}'{location}")
