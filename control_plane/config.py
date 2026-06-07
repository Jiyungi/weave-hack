"""Environment-driven configuration for the control plane (Track B)."""
from __future__ import annotations

import os
from pathlib import Path

# Where Track A (the controller engine) is reachable.
TRACK_A_URL = os.environ.get("TRACK_A_URL", "http://localhost:8000")

# Required — governance state, memory interaction logs, and audit stream (Redis sponsor).
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

AUDIT_FILE = Path(os.environ.get("CP_AUDIT_FILE", "./control_plane_audit.jsonl"))
AUDIT_STREAM = os.environ.get("CP_AUDIT_STREAM", "cp:audit")

# Short generations: the synthetic tool calls are emitted in the first few tokens.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("CP_MAX_NEW_TOKENS", "16"))
STYLE_MAX_NEW_TOKENS = int(os.environ.get("CP_STYLE_MAX_NEW_TOKENS", "64"))
