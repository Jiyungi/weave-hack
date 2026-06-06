"""Environment-driven configuration for the controller engine."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_NAME = os.environ.get("PEFT_CMP_MODEL", "Qwen/Qwen2.5-7B")
CONTROLLER_DIR = Path(os.environ.get("CONTROLLER_DIR", "./controllers"))
GATES = int(os.environ.get("CTRL_GATES", "5000"))
MAX_LOG_GATE = float(os.environ.get("CTRL_MAX_LOG_GATE", "0.05"))

CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
