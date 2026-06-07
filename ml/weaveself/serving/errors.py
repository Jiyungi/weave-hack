"""Typed serving errors for Track A (Serving_Engine).

These errors back the Requirement 7.5 contract: a request for an adapter that
cannot be loaded SHALL yield an error response that names the missing
``adapter_id``.
"""

from __future__ import annotations


class AdapterNotLoadable(LookupError):
    """Raised when a requested ``adapter_id`` cannot be loaded.

    The error message names the offending ``adapter_id`` so the Inference_API
    can surface exactly which adapter was missing (Requirement 7.5). Load is
    validated before any forward-pass mutation, so a failed load never leaves
    gates partially applied.
    """

    def __init__(self, adapter_id: str, reason: str | None = None) -> None:
        self.adapter_id = adapter_id
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(f"Adapter '{adapter_id}' is not loadable{detail}")
