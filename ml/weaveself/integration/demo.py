"""Demo pre-baking + Unit-selection routing to proof visuals (Req 21.3, 21.4).

Training is strictly batch/overnight (Req 13.4); the live demo therefore cannot
wait for an adapter to train. This module **pre-bakes** the Adapter_Files for a
set of demo Units across the demo ``day_index`` values so the demo is
*time-compressed* (Req 21.3): the overnight batch graph is run ahead of time,
once per demo day, and every produced Adapter_File is stored through the
Redis_Client_API so it can be served by ``adapter_id`` (consistent with the
Integration_Milestone wiring from task 12.1).

It then wires the **Unit-selection -> route -> generate -> proof-visuals** flow
(Req 21.4). Selecting a Unit:

1. routes to the correct Adapter via the Redis_Client_API
   (:meth:`RedisClientApi.route`),
2. generates a response through the Inference_API (the single resident
   :class:`ServingEngine`, Base_Model loaded once, gates swapped per request),
   and
3. surfaces the corresponding ``eval_results.json`` proof-visual payload for the
   active demo day.

Time-compression detail: a demo Unit accumulates interactions over the demo
days, so each ``day_index`` produces a *distinct* Adapter_File (distinct
``adapter_id``, distinct ``metadata.day_index``) and its own
``eval_results_day_<d>.json``. All days' adapters live in one shared adapters
directory and one shared Redis keyspace; the route index is pointed at a single
**active day** (:meth:`DemoEnvironment.activate_day`) so ``route(unit_label)``
resolves to that day's adapter instead of an arbitrary tie-break across days.

Environment note: like the rest of the integration layer this defaults to the
dependency-free :class:`~weaveself.serving.backend.StubBackend` and a
file/in-memory Redis fallback, so pre-baking runs without a GPU, a Qwen
download, or a live Redis server. Inject an ``HFBackend`` / live Redis to run
against the real Base_Model and Redis_Layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from weaveself.contracts.eval_results import EvalResults, read_eval_results
from weaveself.integration.redis_client import RedisClientApi, create_redis_client
from weaveself.integration.wiring import (
    DEFAULT_BASE_MODEL,
    build_real_batch_deps,
    materialize_adapter_from_redis,
)
from weaveself.orchestration import UnitSpec, build_batch_graph, initial_state
from weaveself.serving.backend import ModelBackend
from weaveself.serving.engine import ServingEngine

# A day-aware collector returns a Unit's accumulated interactions *as of* a
# given demo day: ``collector(unit_label, day_index) -> [interaction, ...]``.
DayCollector = Callable[[str, int], list[dict]]


# The default demo cast. Categories + a user, matching the design's demo Units.
DEMO_UNITS: tuple[UnitSpec, ...] = (
    {"unit_label": "cooking", "unit_type": "category"},
    {"unit_label": "fitness", "unit_type": "category"},
    {"unit_label": "alice", "unit_type": "user"},
)

# The demo day indices: a short time-compressed sequence of "overnight" runs.
DEMO_DAY_INDICES: tuple[int, ...] = (0, 1, 2)


def make_demo_collector(
    *,
    base_rows: int = 4,
    rows_per_day: int = 2,
) -> DayCollector:
    """Build a deterministic, *cumulative* day-aware collector for the demo.

    A Unit's interaction history grows with the demo day:
    ``base_rows + rows_per_day * day_index`` rows on day ``day_index`` (and the
    earlier rows are a prefix of the later days', modelling accumulation). The
    growing row count makes each day's curated train split distinct, which makes
    each day's Adapter_File distinct (distinct ``adapter_id``) — exactly what a
    time-compressed multi-day demo needs.
    """

    def collector(unit_label: str, day_index: int) -> list[dict]:
        n = base_rows + rows_per_day * max(0, int(day_index))
        return [
            {
                "prompt": f"{unit_label} interaction {i}",
                "completion": f"{unit_label} response {i}",
            }
            for i in range(n)
        ]

    return collector


@dataclass
class DemoSelection:
    """The outcome of selecting a Unit in the demo (Req 21.4).

    Carries the routed ``adapter_id`` (resolved through the Redis_Client_API),
    the generated response (from the Inference_API), and the corresponding
    ``eval_results.json`` proof-visual payload for the active demo day.
    """

    unit_label: str
    day_index: int
    adapter_id: str
    prompt: str
    text: str
    tokens: int
    eval_results: EvalResults


@dataclass
class DemoEnvironment:
    """A pre-baked, ready-to-serve demo (Req 21.3, 21.4).

    Holds the single resident :class:`ServingEngine` (Base_Model loaded once),
    the shared Redis_Client_API with every demo day's adapters stored, the
    per-(day, unit) ``adapter_id`` catalog, and the per-day
    ``eval_results.json`` paths. :meth:`select_unit` performs the
    route -> generate -> proof-visuals flow against the active demo day.
    """

    units: list[UnitSpec]
    day_indices: list[int]
    engine: ServingEngine
    redis_client: RedisClientApi
    adapters_dir: Path
    serve_dir: Path
    # day_index -> {unit_label -> adapter_id}
    catalog: dict[int, dict[str, str]]
    # day_index -> eval_results.json path
    eval_paths: dict[int, Path]
    active_day: int
    failures: list[dict] = field(default_factory=list)

    # -- demo-day control --------------------------------------------------

    def activate_day(self, day_index: int) -> None:
        """Point the Redis route index at ``day_index``'s adapters (Req 21.4).

        After this, :meth:`RedisClientApi.route` resolves each Unit to that
        day's Adapter_File. Blobs/metadata for all days remain addressable by
        ``adapter_id`` regardless of which day is active.
        """
        if day_index not in self.catalog:
            raise KeyError(f"no pre-baked adapters for day_index {day_index}")
        self.redis_client.reindex_route_targets(
            list(self.catalog[day_index].values())
        )
        self.active_day = day_index

    def adapter_id_for(self, unit_label: str, day_index: int | None = None) -> str:
        """Return the pre-baked ``adapter_id`` for a Unit on a demo day."""
        day = self.active_day if day_index is None else day_index
        try:
            return self.catalog[day][unit_label]
        except KeyError as exc:
            raise KeyError(
                f"no pre-baked adapter for unit '{unit_label}' on day {day}"
            ) from exc

    def proof_visuals(self, day_index: int | None = None) -> EvalResults:
        """Read the ``eval_results.json`` proof-visual payload for a demo day."""
        day = self.active_day if day_index is None else day_index
        path = self.eval_paths.get(day)
        if path is None:
            raise KeyError(f"no eval_results.json for day_index {day}")
        return read_eval_results(path)

    # -- Unit selection: route -> generate -> proof visuals (Req 21.4) -----

    def select_unit(
        self,
        unit_label: str,
        prompt: str,
        *,
        day_index: int | None = None,
        max_new_tokens: int = 32,
        from_redis_bytes: bool = True,
    ) -> DemoSelection:
        """Select a Unit and run the full demo flow (Req 21.4).

        1. Resolve the ``adapter_id`` for ``unit_label`` via the Redis route
           index (the routing goes through the Redis_Client_API).
        2. Optionally round-trip the Adapter_File bytes back through the
           Redis_Client_API into ``serve_dir`` (proving the served bytes match
           what the Redis_Layer holds).
        3. Generate a response through the Inference_API (the resident engine
           swaps in the adapter's gate tensors for this single request).
        4. Attach the corresponding ``eval_results.json`` proof-visual payload.
        """
        day = self.active_day if day_index is None else day_index
        if day != self.active_day:
            self.activate_day(day)

        adapter_id = self.redis_client.route(unit_label)

        if from_redis_bytes:
            materialize_adapter_from_redis(
                self.redis_client, adapter_id, self.serve_dir
            )

        generation = self.engine.generate(prompt, adapter_id, max_new_tokens)
        return DemoSelection(
            unit_label=unit_label,
            day_index=day,
            adapter_id=adapter_id,
            prompt=prompt,
            text=generation.text,
            tokens=generation.tokens,
            eval_results=self.proof_visuals(day),
        )


def prebake_demo_adapters(
    units: Sequence[UnitSpec] = DEMO_UNITS,
    day_indices: Sequence[int] = DEMO_DAY_INDICES,
    *,
    collector: DayCollector | None = None,
    workdir: str | Path,
    redis_client: RedisClientApi | None = None,
    redis_url: str | None = None,
    redis_file_path: str | Path | None = None,
    backend: ModelBackend | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    min_rows: int = 2,
    active_day: int | None = None,
) -> DemoEnvironment:
    """Pre-bake Adapter_Files across the demo ``day_index`` values (Req 21.3).

    For each demo day this runs the *real* batch graph (collect -> curate ->
    train -> eval -> store) with the real Track A trainer and the resident
    Inference_API engine, producing a real Adapter_File per Unit and storing each
    through the Redis_Client_API (so it is servable by ``adapter_id``) plus a
    per-day ``eval_results.json`` proof artifact. The returned
    :class:`DemoEnvironment` is then ready to serve the Unit-selection flow
    (Req 21.4) with no further training.

    Args:
        units: the demo Units (defaults to :data:`DEMO_UNITS`).
        day_indices: the demo days to pre-bake (defaults to
            :data:`DEMO_DAY_INDICES`).
        collector: a day-aware interaction collector; defaults to
            :func:`make_demo_collector` (a deterministic, cumulative source).
        workdir: scratch directory for per-day splits, adapters, serving
            materials, and the eval artifacts.
        redis_client: an explicit Redis_Client_API; when ``None`` one is built
            via :func:`create_redis_client` (live Redis when available, else a
            file/in-memory fallback honoring the same layout).
        backend: the serving :class:`ModelBackend`; defaults to the
            dependency-free ``StubBackend`` (inject ``HFBackend`` for real Qwen).
        min_rows: inclusion threshold for the Data_Pipeline split (Req 11.4).
        active_day: which demo day the route index starts pointed at; defaults to
            the latest pre-baked day (the demo's "today").

    Returns:
        A pre-baked :class:`DemoEnvironment`.
    """
    units = list(units)
    day_list = sorted(int(d) for d in day_indices)
    if not day_list:
        raise ValueError("day_indices must be non-empty")
    collector = collector or make_demo_collector()

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # One shared adapters directory holds every day's Adapter_File so the single
    # resident engine can load any of them.
    adapters_dir = workdir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    serve_dir = workdir / "serve_adapters"
    serve_dir.mkdir(parents=True, exist_ok=True)

    if redis_client is None:
        redis_client = create_redis_client(
            url=redis_url,
            file_path=redis_file_path or (workdir / "redis_store.json"),
        )

    # Single resident Serving_Engine: Base_Model loaded exactly once (Req 7.1),
    # shared across every pre-bake day and the serving flow.
    engine = ServingEngine(base_model, backend=backend, adapters_dir=str(adapters_dir))

    catalog: dict[int, dict[str, str]] = {}
    eval_paths: dict[int, Path] = {}
    failures: list[dict] = []

    for day in day_list:
        day_workdir = workdir / f"day_{day}"
        eval_out = workdir / f"eval_results_day_{day}.json"

        # Bind the day-aware collector to this day so the batch graph sees the
        # Unit's interaction history as of this demo day.
        def _day_collector(unit_label: str, _day: int = day) -> list[dict]:
            return list(collector(unit_label, _day))

        deps = build_real_batch_deps(
            collector=_day_collector,
            redis_client=redis_client,
            engine=engine,
            workdir=day_workdir,
            adapters_dir=adapters_dir,
            base_model=base_model,
            day_index=day,
            min_rows=min_rows,
            eval_out_path=eval_out,
        )
        graph = build_batch_graph(deps)
        state = graph.invoke(initial_state(units))

        day_map: dict[str, str] = {}
        for label, adapter_path in state.get("adapters", {}).items():
            adapter_id = Path(adapter_path).stem.removeprefix("adapter_")
            day_map[label] = adapter_id
        catalog[day] = day_map
        for failure in state.get("failures", []):
            failures.append({**failure, "day_index": day})
        if eval_out.exists():
            eval_paths[day] = eval_out

    resolved_active = day_list[-1] if active_day is None else int(active_day)

    env = DemoEnvironment(
        units=units,
        day_indices=day_list,
        engine=engine,
        redis_client=redis_client,
        adapters_dir=adapters_dir,
        serve_dir=serve_dir,
        catalog=catalog,
        eval_paths=eval_paths,
        active_day=resolved_active,
        failures=failures,
    )
    # Point the route index at the active demo day so Unit selection routes to
    # that day's adapter (Req 21.4).
    env.activate_day(resolved_active)
    return env
