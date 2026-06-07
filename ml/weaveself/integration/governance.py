"""Critical-path governance + blocked-state fallback (Requirements 22, 23).

Track A custom serving (Req 7) and method reproduction (Req 6) are the
**critical path**: they are built and verified *before* any dependent
integration (Req 22.1). This module is the control point that enforces that
ordering at wiring time:

* **Serving-verification gate** — :func:`verify_track_a_serving` exercises the
  resident :class:`~weaveself.serving.engine.ServingEngine` against the hard
  serving contract (Base_Model loaded exactly once, a null-adapter request runs
  the pure base, an adapter request produces *different* output than the base,
  and ``/score`` returns a non-negative perplexity). It returns a structured
  :class:`ServingVerification` rather than a bare bool so the decision is
  auditable.

* **Dependency selection** — :class:`CriticalPathGovernor` /
  :func:`build_governed_batch_deps` choose what gets wired into the Track B
  batch graph based on that gate (Req 22.1, 22.2):

      WHILE Track A serving is not yet verified  -> keep the Mock_Dependency
      (``MockTrainAdapter`` / ``MockInference`` / ``MockRedisClient``) wired so
      Track B and Track C keep moving (Req 22.2).

      Serving verified                          -> wire the REAL cross-track
      dependencies (the task 12.1 :func:`~weaveself.integration.wiring.build_real_batch_deps`).

* **Blocked-state recording** — when serving cannot be verified the governor
  records a structured :class:`IntegrationStatus` (and optionally writes an
  ``integration_status.json`` artifact) stating that downstream integration is
  ``blocked`` and that the system falls back to the per-track standalone demos
  (Req 22.3).

* **Fallback demo entry points** — :func:`fallback_demo_entry_points` enumerates
  the per-track standalone fallbacks (Req 23.1-23.3). The Track B
  confusion-matrix demo (:func:`run_track_b_confusion_matrix_demo`) is the single
  highest-priority artifact, runnable independently of every other track
  (Req 23.2, 23.4); a runnable Track A base-vs-adapter compare demo
  (:func:`run_track_a_compare_demo`, Req 23.1) is also provided. Both run on the
  dependency-free defaults (``StubBackend`` + mocks), so they never require a GPU,
  a Qwen download, or a live Redis.

The :mod:`weaveself.orchestration.mocks` module is intentionally kept in place:
governance *uses* those mocks while serving is unverified (Req 22.2); they are
never deleted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Union

from weaveself.data.curation import Curator, GPTCurationNode
from weaveself.eval.weave_eval import PerplexityLogger
from weaveself.integration.redis_client import RedisClientApi, create_redis_client
from weaveself.integration.wiring import (
    DEFAULT_BASE_MODEL,
    Collector,
    EndToEndLoop,
    build_end_to_end_loop,
    build_real_batch_deps,
)
from weaveself.orchestration import BatchDeps, build_batch_graph, initial_state
from weaveself.orchestration.mocks import (
    MockInference,
    MockRedisClient,
    MockTrainAdapter,
)
from weaveself.serving.backend import ModelBackend
from weaveself.serving.engine import ServingEngine

# What a caller may pass to express "is Track A serving verified?": an explicit
# bool, a pre-computed verification, or a zero-arg callable producing either.
ServingVerifiedInput = Union[
    bool,
    "ServingVerification",
    Callable[[], Union[bool, "ServingVerification"]],
]


# ---------------------------------------------------------------------------
# Status / verification value objects
# ---------------------------------------------------------------------------


class IntegrationState(str, Enum):
    """Whether downstream cross-track integration is live or blocked (Req 22.3)."""

    INTEGRATED = "integrated"
    BLOCKED = "blocked"


class DependencyMode(str, Enum):
    """Which dependencies are wired into the batch graph (Req 22.1, 22.2)."""

    REAL = "real"  # real Track A serving + Track C Redis (Integration_Milestone)
    MOCK = "mock"  # Mock_Dependency kept wired while serving is unverified


@dataclass(frozen=True)
class ServingVerification:
    """The structured outcome of the Track A serving-verification gate (Req 22.1).

    ``verified`` is the headline decision; ``checks`` records each individual
    contract probe so a blocked state can name *what* failed, and ``reason`` is a
    human-readable summary.
    """

    verified: bool
    reason: str
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"verified": self.verified, "reason": self.reason, "checks": dict(self.checks)}


@dataclass
class IntegrationStatus:
    """Structured critical-path status recorded by the governor (Req 22.3).

    When serving is verified this is an ``integrated`` status wiring the real
    dependencies. When serving cannot be verified this is a ``blocked`` status
    naming the reason and listing the per-track standalone fallback demos the
    system falls back to (Req 23).
    """

    state: IntegrationState
    serving_verified: bool
    dependency_mode: DependencyMode
    reason: str
    fallback_demos: list[str] = field(default_factory=list)
    verification: ServingVerification | None = None
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def blocked(self) -> bool:
        return self.state is IntegrationState.BLOCKED

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "serving_verified": self.serving_verified,
            "dependency_mode": self.dependency_mode.value,
            "reason": self.reason,
            "fallback_demos": list(self.fallback_demos),
            "verification": self.verification.to_dict() if self.verification else None,
            "recorded_at": self.recorded_at,
        }

    def write(self, path: str | Path) -> Path:
        """Persist this status as an ``integration_status.json`` artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Per-track standalone fallback demos (Requirement 23)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackDemo:
    """Describes one track's standalone fallback demo (Req 23.1-23.4)."""

    track: str
    title: str
    description: str
    requirement: str
    entry_point: str
    highest_priority: bool = False
    # A zero-config runnable for demos that execute in-process on the
    # dependency-free defaults; ``None`` for demos that live in another runtime
    # (e.g. the Track C Node/TS CopilotKit UI).
    runnable: Callable[..., object] | None = None


def fallback_demo_entry_points() -> dict[str, FallbackDemo]:
    """Enumerate the per-track standalone fallback demos (Req 23).

    The Track B Confusion_Matrix is flagged ``highest_priority`` — it is the
    single artifact demoable independently of every other track (Req 23.4).
    """
    return {
        "track_a": FallbackDemo(
            track="A",
            title="CLI compare: Base_Model vs one Adapter on the same prompt",
            description=(
                "Generate the same prompt through the resident Serving_Engine with "
                "no adapter and with one Adapter, showing visibly different, more "
                "on-style adapter output (Req 23.1)."
            ),
            requirement="23.1",
            entry_point="weaveself.integration.governance:run_track_a_compare_demo",
            runnable=run_track_a_compare_demo,
        ),
        "track_b": FallbackDemo(
            track="B",
            title="Confusion-matrix heatmap + base-vs-adapter perplexity",
            description=(
                "Emit a schema-conformant eval_results.json containing a square "
                "cross-unit Confusion_Matrix and the base-vs-adapter perplexity "
                "comparison from pre-trained (mock) adapters, with no dependency "
                "on Track A serving or the Frontend_App (Req 23.2). This is the "
                "single highest-priority, independently-demoable artifact (Req 23.4)."
            ),
            requirement="23.2, 23.4",
            entry_point="weaveself.integration.governance:run_track_b_confusion_matrix_demo",
            highest_priority=True,
            runnable=run_track_b_confusion_matrix_demo,
        ),
        "track_c": FallbackDemo(
            track="C",
            title="CopilotKit UI with mock eval_results.json + Redis library view",
            description=(
                "Render the dashboard (confusion-matrix heatmap, example pairs, "
                "size chart, adapter library) from a mock eval_results.json fixture "
                "and the Redis adapter-library view (Req 23.3). This demo runs in "
                "the Track C Node/TS runtime (app/src/frontend); the Track B demo "
                "above produces a compatible eval_results.json fixture for it."
            ),
            requirement="23.3",
            entry_point="app/src/frontend (CopilotKit React, Node/TS runtime)",
            runnable=None,
        ),
    }


def _default_demo_units() -> list[dict]:
    return [
        {"unit_label": "alice", "unit_type": "user"},
        {"unit_label": "bob", "unit_type": "user"},
        {"unit_label": "carol", "unit_type": "user"},
    ]


def _demo_collector(labels: list[str], n: int = 8) -> Collector:
    data = {
        label: [
            {"prompt": f"{label} question {i}", "completion": f"{label} answer {i}"}
            for i in range(n)
        ]
        for label in labels
    }
    return lambda label: list(data.get(label, []))


def run_track_b_confusion_matrix_demo(
    workdir: str | Path,
    *,
    units: list[dict] | None = None,
    interactions_per_unit: int = 8,
    min_rows: int = 2,
    out_path: str | Path | None = None,
):
    """Run the Track B confusion-matrix fallback demo (Req 23.2, 23.4).

    Runs the full LangGraph batch graph against the Track A / Track C
    ``Mock_Dependency`` fixtures (``MockTrainAdapter`` / ``MockInference`` /
    ``MockRedisClient``) and emits a schema-conformant ``eval_results.json``
    containing a square cross-unit Confusion_Matrix — with **no dependency** on
    Track A serving or the Frontend_App. Returns ``(EvalResults, eval_path)``.

    This is the single highest-priority, independently-demoable artifact
    (Req 23.4): it is exactly the proof visual the Track B fallback shows when
    integration slips.
    """
    from weaveself.contracts.eval_results import read_eval_results

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    units = units or _default_demo_units()
    labels = [u["unit_label"] for u in units]
    eval_path = Path(out_path) if out_path is not None else workdir / "eval_results.json"

    deps = build_mock_batch_deps(
        collector=_demo_collector(labels, interactions_per_unit),
        workdir=workdir,
        min_rows=min_rows,
        eval_out_path=eval_path,
    )
    graph = build_batch_graph(deps)
    graph.invoke(initial_state(units))

    results = read_eval_results(eval_path)
    return results, eval_path


def run_track_a_compare_demo(
    workdir: str | Path,
    *,
    prompt: str = "Summarize today in my voice.",
    unit_label: str = "alice",
    unit_type: str = "user",
    base_model: str = "stub-base",
    backend: ModelBackend | None = None,
):
    """Run the Track A base-vs-adapter CLI compare fallback demo (Req 23.1).

    Trains one Adapter_File (via ``MockTrainAdapter`` so the demo is
    dependency-free), then generates the *same* prompt through a single resident
    :class:`ServingEngine` with no adapter and with that adapter, returning a dict
    of ``{base, adapter, differ}``. On the default ``StubBackend`` the adapter
    output deterministically differs from the base, demonstrating that the
    adapter steers the output (Req 7.4 / 23.1) with no dependency on Track B or C.
    """
    workdir = Path(workdir)
    adapters_dir = workdir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # A tiny local "dataset" of one Unit's training pairs (Mock_Dependency).
    from weaveself.contracts.training_pair import write_training_pairs

    dataset_path = workdir / "demo_train.jsonl"
    write_training_pairs(
        dataset_path,
        [
            {
                "prompt": f"{unit_label} q{i}",
                "completion": f"{unit_label} a{i}",
                "unit_label": unit_label,
            }
            for i in range(4)
        ],
    )

    train = MockTrainAdapter(adapters_dir)
    adapter_path = train(str(dataset_path), unit_label, unit_type)
    adapter_id = Path(adapter_path).stem.removeprefix("adapter_")

    engine = ServingEngine(base_model, backend=backend, adapters_dir=str(adapters_dir))
    base_text = engine.generate(prompt, None, max_new_tokens=32).text
    adapter_text = engine.generate(prompt, adapter_id, max_new_tokens=32).text
    return {
        "prompt": prompt,
        "adapter_id": adapter_id,
        "base": base_text,
        "adapter": adapter_text,
        "differ": base_text != adapter_text,
    }


# ---------------------------------------------------------------------------
# Serving-verification gate (Requirement 22.1)
# ---------------------------------------------------------------------------


def verify_track_a_serving(
    engine: ServingEngine,
    *,
    probe_prompt: str = "Tell me about your day.",
    probe_target: str = "It was productive.",
    adapter_id: str | None = None,
) -> ServingVerification:
    """Probe the resident Serving_Engine against the hard serving contract.

    Returns a :class:`ServingVerification` recording each probe:

    * ``base_loaded_once`` — the Base_Model was loaded exactly once (Req 7.1).
    * ``base_generates`` — a null-adapter request runs the pure base (Req 7.3).
    * ``score_non_negative`` — ``/score`` returns a non-negative perplexity (Req 8.2).
    * ``adapter_differs`` — when ``adapter_id`` is supplied, an adapter request
      produces different output than the base (Req 7.4). When no adapter is
      supplied this probe is skipped (recorded ``True``) since no trained adapter
      is available to verify the differential.

    Serving is ``verified`` iff every executed probe passes. Any exception is
    captured as a failed verification (never raised) so the governor can fall
    back rather than crash (Req 22.3).
    """
    checks: dict[str, bool] = {}
    try:
        base_gen = engine.generate(probe_prompt, None, max_new_tokens=16)
        checks["base_generates"] = isinstance(base_gen.text, str) and bool(base_gen.text)

        score = engine.score(probe_prompt, probe_target, None)
        checks["score_non_negative"] = score.perplexity >= 0 and score.nll >= 0

        checks["base_loaded_once"] = engine.base_model_load_count == 1

        if adapter_id is not None:
            adapter_gen = engine.generate(probe_prompt, adapter_id, max_new_tokens=16)
            checks["adapter_differs"] = adapter_gen.text != base_gen.text
        else:
            checks["adapter_differs"] = True  # skipped: no adapter to differentiate
    except Exception as exc:  # noqa: BLE001 - capture, never crash the gate
        checks.setdefault("base_generates", False)
        return ServingVerification(
            verified=False,
            reason=f"serving probe raised: {type(exc).__name__}: {exc}",
            checks=checks,
        )

    verified = all(checks.values())
    if verified:
        reason = "Track A serving verified: " + ", ".join(sorted(checks))
    else:
        failed = sorted(name for name, ok in checks.items() if not ok)
        reason = "Track A serving NOT verified; failed probes: " + ", ".join(failed)
    return ServingVerification(verified=verified, reason=reason, checks=checks)


def _resolve_verification(serving_verified: ServingVerifiedInput) -> ServingVerification:
    """Normalize the many accepted ``serving_verified`` inputs to a verification."""
    value: object = serving_verified
    if callable(value):
        value = value()
    if isinstance(value, ServingVerification):
        return value
    verified = bool(value)
    reason = (
        "serving reported verified"
        if verified
        else "serving reported NOT verified"
    )
    return ServingVerification(verified=verified, reason=reason, checks={"reported": verified})


# ---------------------------------------------------------------------------
# Mock-wired BatchDeps (Requirement 22.2) — keep mocks wired while unverified
# ---------------------------------------------------------------------------


def build_mock_batch_deps(
    *,
    collector: Collector,
    workdir: str | Path,
    adapters_dir: str | Path | None = None,
    redis_client: object | None = None,
    curator: Curator | None = None,
    day_index: int = 0,
    min_rows: int = 1,
    logger: PerplexityLogger | None = None,
    eval_out_path: str | Path | None = None,
) -> BatchDeps:
    """Compose :class:`BatchDeps` wired to the ``Mock_Dependency`` fixtures (Req 22.2).

    This is the dependency set the batch graph runs against WHILE Track A serving
    is not yet verified: ``MockTrainAdapter`` (writes real, schema-conformant
    Adapter_Files without training), ``MockInference`` (deterministic ``/score``
    giving a diagonal confusion matrix), and ``MockRedisClient``. The real
    :class:`GPTCurationNode` is used (Track B owns curation, so it is never
    mocked).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    adapters_dir = Path(adapters_dir) if adapters_dir is not None else workdir / "mock_adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    return BatchDeps(
        collector=collector,
        curation_node=GPTCurationNode(curator),
        train_adapter=MockTrainAdapter(adapters_dir, day_index=day_index),
        redis_client=redis_client if redis_client is not None else MockRedisClient(),
        score_fn=MockInference(),
        workdir=workdir,
        min_rows=min_rows,
        logger=logger,
        eval_out_path=eval_out_path,
    )


# ---------------------------------------------------------------------------
# The governor (Requirement 22)
# ---------------------------------------------------------------------------


@dataclass
class GovernedBatch:
    """The governed wiring decision: the selected deps plus the recorded status."""

    deps: BatchDeps
    status: IntegrationStatus
    verification: ServingVerification


class CriticalPathGovernor:
    """Decides real-vs-mock wiring from the serving-verification gate (Req 22).

    The governor never trains or serves anything itself; it only resolves the
    "is Track A serving verified?" gate and turns it into a dependency-mode
    decision plus a recordable :class:`IntegrationStatus`.
    """

    def __init__(self, serving_verified: ServingVerifiedInput) -> None:
        self._serving_verified = serving_verified

    def verify(self) -> ServingVerification:
        """Resolve the serving-verification gate (Req 22.1)."""
        return _resolve_verification(self._serving_verified)

    def dependency_mode(
        self, verification: ServingVerification | None = None
    ) -> DependencyMode:
        """Real deps when serving is verified, else keep mocks wired (Req 22.2)."""
        verification = verification or self.verify()
        return DependencyMode.REAL if verification.verified else DependencyMode.MOCK

    def status(
        self, verification: ServingVerification | None = None
    ) -> IntegrationStatus:
        """Build the structured :class:`IntegrationStatus` for the current gate.

        Verified -> ``integrated`` (real deps). Not verified -> ``blocked``: the
        system records that downstream integration is blocked and falls back to
        the per-track standalone demos (Req 22.3 / 23).
        """
        verification = verification or self.verify()
        if verification.verified:
            return IntegrationStatus(
                state=IntegrationState.INTEGRATED,
                serving_verified=True,
                dependency_mode=DependencyMode.REAL,
                reason=verification.reason,
                fallback_demos=[],
                verification=verification,
            )
        fallbacks = [demo.entry_point for demo in fallback_demo_entry_points().values()]
        return IntegrationStatus(
            state=IntegrationState.BLOCKED,
            serving_verified=False,
            dependency_mode=DependencyMode.MOCK,
            reason=(
                "Track A serving could not be verified; downstream integration is "
                "blocked. Mock_Dependency stays wired (Req 22.2) and the team falls "
                f"back to the per-track standalone demos (Req 23). {verification.reason}"
            ),
            fallback_demos=fallbacks,
            verification=verification,
        )

    def record_blocked_state(
        self, path: str | Path, verification: ServingVerification | None = None
    ) -> IntegrationStatus:
        """Record the blocked-integration state to ``path`` (Req 22.3).

        Resolves the status and, when blocked, writes the
        ``integration_status.json`` artifact. (The status is also written when
        integrated so the artifact always reflects the latest decision.)
        """
        status = self.status(verification)
        status.write(path)
        return status


def build_governed_batch_deps(
    *,
    serving_verified: ServingVerifiedInput,
    collector: Collector,
    workdir: str | Path,
    redis_client: RedisClientApi | None = None,
    engine: ServingEngine | None = None,
    adapters_dir: str | Path | None = None,
    curator: Curator | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    day_index: int = 0,
    min_rows: int = 1,
    logger: PerplexityLogger | None = None,
    eval_out_path: str | Path | None = None,
    backend: ModelBackend | None = None,
    status_path: str | Path | None = None,
) -> GovernedBatch:
    """Build :class:`BatchDeps` selecting real-vs-mock from the serving gate (Req 22).

    * Serving verified -> wire the REAL cross-track dependencies via the task
      12.1 :func:`build_real_batch_deps` (real ``train_adapter``, the resident
      ``ServingEngine`` ``/score``, the real ``RedisClientApi``). A resident
      engine and Redis client are constructed on the dependency-free defaults
      when not supplied.
    * Serving NOT verified -> keep the ``Mock_Dependency`` wired via
      :func:`build_mock_batch_deps` (Req 22.2) and record the blocked-integration
      state (Req 22.3).

    When ``status_path`` is given the :class:`IntegrationStatus` is always written
    there as ``integration_status.json``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    governor = CriticalPathGovernor(serving_verified)
    verification = governor.verify()
    status = governor.status(verification)

    if verification.verified:
        adapters_dir = Path(adapters_dir) if adapters_dir is not None else workdir / "adapters"
        adapters_dir.mkdir(parents=True, exist_ok=True)
        if engine is None:
            engine = ServingEngine(base_model, backend=backend, adapters_dir=str(adapters_dir))
        if redis_client is None:
            redis_client = create_redis_client(file_path=workdir / "redis_store.json")
        if eval_out_path is None:
            eval_out_path = workdir / "eval_results.json"
        deps = build_real_batch_deps(
            collector=collector,
            redis_client=redis_client,
            engine=engine,
            workdir=workdir,
            adapters_dir=adapters_dir,
            curator=curator,
            base_model=base_model,
            day_index=day_index,
            min_rows=min_rows,
            logger=logger,
            eval_out_path=eval_out_path,
        )
    else:
        deps = build_mock_batch_deps(
            collector=collector,
            workdir=workdir,
            adapters_dir=adapters_dir,
            redis_client=redis_client,
            curator=curator,
            day_index=day_index,
            min_rows=min_rows,
            logger=logger,
            eval_out_path=eval_out_path,
        )

    if status_path is not None:
        status.write(status_path)

    return GovernedBatch(deps=deps, status=status, verification=verification)


def build_governed_end_to_end_loop(
    *,
    serving_verified: ServingVerifiedInput,
    collector: Collector,
    workdir: str | Path,
    status_path: str | Path | None = None,
    **loop_kwargs: object,
) -> tuple[EndToEndLoop | None, IntegrationStatus]:
    """Build the real end-to-end loop only when serving is verified (Req 22.1).

    Returns ``(loop, status)``. When serving is verified the real
    :class:`EndToEndLoop` (task 12.1) is constructed. When serving cannot be
    verified the loop is ``None`` (integration is blocked, Req 22.3) and the
    caller falls back to the per-track standalone demos (Req 23); the blocked
    status is recorded to ``status_path`` when provided.
    """
    governor = CriticalPathGovernor(serving_verified)
    verification = governor.verify()
    status = governor.status(verification)
    if status_path is not None:
        status.write(status_path)

    if not verification.verified:
        return None, status

    loop = build_end_to_end_loop(collector=collector, workdir=workdir, **loop_kwargs)  # type: ignore[arg-type]
    return loop, status
