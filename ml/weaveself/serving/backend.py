"""Model backend abstraction for the Serving_Engine (Track A).

The :class:`ServingEngine` does not talk to a model framework directly. Instead
it drives a :class:`ModelBackend`: an interface that abstracts loading the
Base_Model exactly once and running forward passes (generate / score) with an
optional set of per-channel gate tensors applied for a single request.

Two implementations are provided:

* :class:`HFBackend` — loads a real HuggingFace instruct model (default
  ``Qwen2.5-1.5B-Instruct``) via ``transformers``/``torch``. It is **import-safe
  and lazy**: importing this module never imports ``torch`` or downloads
  weights, and ``HFBackend`` only loads weights when :meth:`load_base_model` is
  actually invoked. The heavy dependencies live in the optional ``serving``
  extra (see ``pyproject.toml``).
* :class:`StubBackend` — a tiny deterministic in-memory fake requiring no
  ``torch``/``transformers``. Unit tests and CI use it so engine logic and
  100+ property iterations are verifiable without a GPU or multi-GB downloads.

Every backend tracks a ``load_count`` so callers (and Property "base loaded
once", Req 7.1) can assert the Base_Model is loaded exactly once per process.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from numpy.typing import NDArray

# A set of NKT-Mirror gate tensors keyed by parameter name.
GateTensors = dict[str, np.ndarray]


@dataclass
class Generation:
    """Result of a generate call: decoded text and the token count produced."""

    text: str
    tokens: int


@dataclass
class ScoreResult:
    """Result of a score call: teacher-forced NLL and perplexity of a target."""

    nll: float
    perplexity: float


class ModelBackend(ABC):
    """Abstracts Base_Model load + forward/generate/score.

    Implementations MUST load the Base_Model lazily inside
    :meth:`load_base_model` (never at import time) and MUST increment
    :attr:`load_count` each time the base model is loaded so the single-load
    invariant (Req 7.1) is observable.
    """

    @property
    @abstractmethod
    def load_count(self) -> int:
        """Number of times the Base_Model has been loaded by this backend."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the Base_Model is currently resident in memory."""

    @property
    @abstractmethod
    def base_model_id(self) -> str | None:
        """The id of the resident Base_Model, or ``None`` if not loaded."""

    @property
    @abstractmethod
    def active_gate_signature(self) -> str | None:
        """Signature of the gate set currently applied, or ``None`` when none.

        This is the observable backing the per-request gate lifecycle: it is
        non-``None`` only *while* a request is mid-flight with gates applied and
        MUST return to ``None`` after every request completes — including
        requests that raise — so the next request runs against the pure
        Base_Model (Req 6.3, 7.2, 7.3; protects Property 4).
        """

    @abstractmethod
    def load_base_model(self, base_model_id: str) -> None:
        """Load the Base_Model identified by ``base_model_id`` into memory."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        gates: Mapping[str, np.ndarray] | None,
        max_new_tokens: int,
    ) -> Generation:
        """Generate text for ``prompt`` with optional ``gates`` applied."""

    @abstractmethod
    def score(
        self,
        prompt: str,
        target: str,
        gates: Mapping[str, np.ndarray] | None,
    ) -> ScoreResult:
        """Teacher-forced NLL/perplexity of ``target`` given ``prompt``."""


class StubBackend(ModelBackend):
    """A deterministic, dependency-free fake Base_Model for tests and CI.

    No ``torch``/``transformers`` required. Outputs are a deterministic function
    of the prompt, the resident base-model id, and any applied gate tensors, so
    that:

    * base-vs-adapter outputs differ whenever gates are applied (supports the
      differential-serving properties), and
    * results are reproducible across the 100+ iterations of a property test.
    """

    def __init__(self) -> None:
        self._load_count = 0
        self._base_model_id: str | None = None
        # Signature of the gate set currently applied for an in-flight request.
        # ``None`` between requests — see the per-request lifecycle in
        # :meth:`_apply_gates`.
        self._active_gate_sig: str | None = None

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def is_loaded(self) -> bool:
        return self._base_model_id is not None

    @property
    def base_model_id(self) -> str | None:
        return self._base_model_id

    @property
    def active_gate_signature(self) -> str | None:
        return self._active_gate_sig

    def load_base_model(self, base_model_id: str) -> None:
        self._base_model_id = base_model_id
        self._load_count += 1

    # -- per-request gate lifecycle -----------------------------------------

    @contextmanager
    def _apply_gates(self, gates: Mapping[str, np.ndarray] | None) -> Iterator[None]:
        """Apply ``gates`` for a single request, ALWAYS clearing them after.

        The real backend registers forward hooks here; the stub models the same
        lifecycle by recording the active gate signature for the duration of the
        ``with`` block. Clearing happens in a ``finally`` block so a request that
        raises mid-flight cannot leave gates applied for the next request
        (Req 6.3, 7.2; protects the null-adapter==base property). A ``None`` /
        empty gate set leaves the pure Base_Model in place (Req 7.3).
        """
        self._active_gate_sig = self._gate_signature(gates) if gates else None
        try:
            yield
        finally:
            self._active_gate_sig = None

    # -- internal deterministic helpers -------------------------------------

    @staticmethod
    def _gate_signature(gates: Mapping[str, np.ndarray] | None) -> str:
        """A short stable signature of a gate-tensor set (empty when ``None``)."""
        if not gates:
            return "base"
        hasher = hashlib.sha256()
        for name in sorted(gates):
            arr = np.ascontiguousarray(np.asarray(gates[name]))
            hasher.update(name.encode("utf-8"))
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(arr.tobytes())
        return hasher.hexdigest()[:12]

    def _require_loaded(self) -> str:
        if self._base_model_id is None:
            raise RuntimeError("Base_Model is not loaded")
        return self._base_model_id

    def generate(
        self,
        prompt: str,
        gates: Mapping[str, np.ndarray] | None,
        max_new_tokens: int,
    ) -> Generation:
        base = self._require_loaded()
        with self._apply_gates(gates):
            # Output derives from the gates applied for THIS request, so adapter
            # output differs from base (Property 5) and the signature is read
            # from the active per-request state.
            sig = self._active_gate_sig or "base"
            text = f"[{base}|{sig}] {prompt}"[
                : max(0, len(prompt) + len(sig) + len(base) + 8)
            ]
            if max_new_tokens >= 0:
                tokens = min(max_new_tokens, len(text.split()))
            else:
                tokens = len(text.split())
            return Generation(text=text, tokens=tokens)

    def score(
        self,
        prompt: str,
        target: str,
        gates: Mapping[str, np.ndarray] | None,
    ) -> ScoreResult:
        self._require_loaded()
        with self._apply_gates(gates):
            sig = self._active_gate_sig or "base"
            token_count = max(1, len(target.split()))
            # Deterministic non-negative mean NLL per token derived from inputs;
            # gate-sensitive. Keeping the per-token mean as the primitive makes
            # the perplexity == exp(nll / token_count) identity exact (Property 8).
            seed = int(
                hashlib.sha256(
                    f"{prompt}\x00{target}\x00{sig}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            mean_nll = (seed % 1000) / 1000.0  # in [0, 1), non-negative
            nll = mean_nll * token_count
            perplexity = float(np.exp(nll / token_count))
            return ScoreResult(nll=float(nll), perplexity=perplexity)


class HFBackend(ModelBackend):
    """Real HuggingFace instruct-model backend (lazy, import-safe).

    Importing this module does NOT import ``torch``/``transformers`` and does NOT
    download weights. The heavy imports and the model/tokenizer load happen only
    when :meth:`load_base_model` is invoked. Install the optional dependencies
    with ``pip install -e .[serving]``.

    Per-request gate application is performed with forward hooks registered for
    the duration of a single :meth:`generate`/:meth:`score` call and always
    removed in a ``finally`` block (Req 6.3, 7.2, 7.3); :meth:`score` computes a
    teacher-forced NLL and a consistent perplexity (Req 8.2). The forward-pass
    paths require real weights and are therefore excluded from coverage.
    """

    DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self, device: str | None = None, torch_dtype: str | None = None
    ) -> None:
        self._load_count = 0
        self._base_model_id: str | None = None
        self._device = device
        # Optional dtype hint (e.g. "float32"/"float16"/"bfloat16"); applied at
        # load time. ``None`` lets transformers pick its default. Stored as a
        # plain string so importing this module never imports ``torch``.
        self._torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        self._active_gate_sig: str | None = None

    @staticmethod
    def _resolve_dtype(torch_dtype: str | None):  # pragma: no cover - requires torch
        """Map a dtype name to a real ``torch.dtype`` (``None`` -> default)."""
        if not torch_dtype:
            return None
        import torch

        name = str(torch_dtype).strip().lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if name in ("auto", "default", "none"):
            return None
        if name not in mapping:
            raise ValueError(f"unsupported MODEL_DTYPE: {torch_dtype!r}")
        return mapping[name]

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def base_model_id(self) -> str | None:
        return self._base_model_id

    @property
    def active_gate_signature(self) -> str | None:
        return self._active_gate_sig

    def load_base_model(self, base_model_id: str) -> None:
        # Lazy, guarded imports — never executed at module import time.
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "HFBackend requires the optional 'serving' dependencies. "
                "Install them with: pip install -e '.[serving]'"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        dtype = self._resolve_dtype(self._torch_dtype)
        if dtype is not None:  # pragma: no cover - hardware/dtype dependent
            self._model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=dtype
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(base_model_id)
        self._model.eval()
        if self._device:  # pragma: no cover - hardware dependent
            self._model.to(self._device)
        self._base_model_id = base_model_id
        self._load_count += 1

    def _require_loaded(self):  # pragma: no cover - requires real weights
        if self._model is None:
            raise RuntimeError("Base_Model is not loaded")
        return self._model

    def _resolve_module(self, gate_name: str):  # pragma: no cover - requires weights
        """Resolve the resident module a gate tensor scales.

        Gate tensors are keyed by ``<module path>.gate``; the corresponding
        module is the dotted path with the trailing ``.gate`` removed. Returns
        ``None`` when the path does not resolve so an unknown gate is skipped
        rather than crashing the request.
        """
        target = gate_name[: -len(".gate")] if gate_name.endswith(".gate") else gate_name
        try:
            return self._model.get_submodule(target)
        except AttributeError:
            return None

    @contextmanager
    def _apply_gates(
        self, gates: Mapping[str, np.ndarray] | None
    ) -> Iterator[None]:  # pragma: no cover - requires real weights
        """Apply ``gates`` via forward hooks for ONE request, then always clear.

        For each gate tensor a ``register_forward_hook`` performs per-channel
        activation scaling on the resident model's matching layer (Req 6.3,
        7.2). Every hook handle is removed in the ``finally`` block so the next
        request runs against the pure Base_Model even if generation raises
        (Req 7.3; protects the null-adapter==base property). When ``gates`` is
        ``None``/empty no hooks are registered and the pure base runs.
        """
        import torch

        handles = []
        self._active_gate_sig = StubBackend._gate_signature(gates) if gates else None
        try:
            if gates:
                for name, gate in gates.items():
                    module = self._resolve_module(name)
                    if module is None:
                        continue
                    scale = torch.as_tensor(np.asarray(gate), dtype=torch.float32)
                    if self._device:
                        scale = scale.to(self._device)

                    def _hook(_mod, _inp, out, _scale=scale):
                        # Per-channel activation scaling on the layer output.
                        # The resident model's activation width (Qwen2.5-1.5B
                        # MLP output == hidden_size) may differ from the gate
                        # vector's channel count, and its dtype/device follow
                        # the loaded model. Align the gate vector to the live
                        # activation before scaling so a real adapter steers
                        # activations without a shape/dtype mismatch: gate
                        # channels beyond the activation width are dropped and
                        # any remaining activation channels stay neutral (1.0).
                        activation = out[0] if isinstance(out, tuple) else out
                        width = activation.shape[-1]
                        s = _scale.to(dtype=activation.dtype, device=activation.device)
                        if s.shape[-1] != width:
                            if s.shape[-1] > width:
                                s = s[:width]
                            else:
                                pad = torch.ones(
                                    width - s.shape[-1],
                                    dtype=s.dtype,
                                    device=s.device,
                                )
                                s = torch.cat([s, pad], dim=0)
                        if isinstance(out, tuple):
                            return (activation * s, *out[1:])
                        return activation * s

                    handles.append(module.register_forward_hook(_hook))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active_gate_sig = None

    def generate(
        self,
        prompt: str,
        gates: Mapping[str, np.ndarray] | None,
        max_new_tokens: int,
    ) -> Generation:  # pragma: no cover - requires real weights
        import torch

        self._require_loaded()
        # Qwen2.5 is an *instruct* model: wrap the prompt in the chat template
        # (<|im_start|>user ... <|im_start|>assistant) so it answers as an
        # assistant instead of autocompleting raw text. Fall back to the bare
        # prompt only if the tokenizer has no chat template.
        if getattr(self._tokenizer, "chat_template", None):
            text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt")
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        with self._apply_gates(gates):
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
        new_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text=text, tokens=int(new_ids.shape[-1]))

    def score(
        self,
        prompt: str,
        target: str,
        gates: Mapping[str, np.ndarray] | None,
    ) -> ScoreResult:  # pragma: no cover - requires real weights
        import torch

        self._require_loaded()
        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        target_ids = self._tokenizer(target, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        if self._device:
            input_ids = input_ids.to(self._device)
        # Teacher forcing: only the target tokens contribute to the loss; the
        # prompt positions are masked out with the ignore index.
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[-1]] = -100
        with self._apply_gates(gates):
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, labels=labels)
        token_count = max(1, int(target_ids.shape[-1]))
        # HF returns the mean NLL per (unmasked) target token; recover the total
        # NLL and a perplexity that satisfies perplexity == exp(nll / tokens),
        # both non-negative (Req 8.2, Property 8).
        mean_nll = float(outputs.loss)
        nll = mean_nll * token_count
        perplexity = float(np.exp(nll / token_count))
        return ScoreResult(nll=nll, perplexity=perplexity)
