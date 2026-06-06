"""Pydantic request models for the HTTP layer."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class EvalItem(BaseModel):
    prompt: str
    needle: str | None = None
    gold: str | None = None

    @model_validator(mode="after")
    def _needs_target(self):
        if self.needle is None and self.gold is None:
            raise ValueError("each item needs a 'needle' or 'gold'")
        return self


class TrainReq(BaseModel):
    task_id: str
    examples: list[dict] = Field(..., description="[{prompt, completion}, ...]")
    # Smoke-validated on Qwen2.5-7B: 240/5e-3 under-fits, 600/8e-3 saturates.
    steps: int = 600
    lr: float = 8e-3
    batch_size: int = 8
    max_length: int = 256


class ComposeReq(BaseModel):
    controller_ids: list[str]
    weights: list[float]
    new_id: str | None = None

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.controller_ids) != len(self.weights):
            raise ValueError("controller_ids and weights must be the same length")
        return self


class ExecuteReq(BaseModel):
    controller_id: str | None = None
    prompt: str
    max_new_tokens: int = 32


class EvaluateReq(BaseModel):
    controller_id: str | None = None
    items: list[EvalItem]
    max_new_tokens: int = 32


class PairReq(BaseModel):
    a: str
    b: str


class DiagnoseReq(BaseModel):
    skill: str
    items: list[EvalItem]
    threshold: float = 0.1
    max_new_tokens: int = 32


class ForgettingReq(BaseModel):
    controller_id: str
    items: list[EvalItem]
    max_new_tokens: int = 32


class JailbreakReq(BaseModel):
    controller_id: str
    needle: str
    prompts: list[str]
    baseline_controller_id: str | None = None
    max_new_tokens: int = 48
