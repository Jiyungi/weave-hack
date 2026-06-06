"""Environment-driven configuration for the controller engine."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_NAME = os.environ.get("PEFT_CMP_MODEL", "Qwen/Qwen2.5-7B")
CONTROLLER_DIR = Path(os.environ.get("CONTROLLER_DIR", "./controllers"))
# Defaults validated by smoke_compose_subtract.py on Qwen2.5-7B: at gates=5000 /
# max_log_gate=0.05 the controllers under-fit (solo skills failed). gates=10000 /
# max_log_gate=0.1 saturated solo skills (loss -> 0.000) and gave clean
# compose/subtract. ~5% per-channel scaling is too weak to steer a 7B.
GATES = int(os.environ.get("CTRL_GATES", "10000"))
MAX_LOG_GATE = float(os.environ.get("CTRL_MAX_LOG_GATE", "0.1"))

CONTROLLER_DIR.mkdir(parents=True, exist_ok=True)
