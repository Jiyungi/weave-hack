# Spec Mode Prompt — WeaveSelf (legacy hackathon spec)

> **Superseded for OpenMirror:** use [`README.md`](README.md) and
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the unified product (Tracks A–D,
> separate style + tool adapters, `memory/` consolidation). This prompt is kept for
> the original WeaveSelf three-track hackathon spec.

Paste the block below into Spec Mode. It assumes `PROJECT_CONTEXT_README.md` (in this
folder) is available as context — read it first.

---

## PROMPT TO SPEC MODE

We are building **WeaveSelf** at a 2-day hackathon (WeaveHacks 4). Full background,
architecture, decisions, constraints, and risks are in `PROJECT_CONTEXT_README.md` — read it
before producing the spec.

**What it is:** a local, overnight "weight-memory" personalization engine. A frozen
open-source **instruct** model is specialized per user/category by tiny (~100 KB) **NKT-Mirror
activation-gating adapters**, retrained in nightly batches, served locally, and **proven** by
an objective evaluation (held-out perplexity + a cross-user identification confusion matrix,
plus beating a context-memory baseline at zero context cost). Orchestrated locally by
LangGraph; GPT is used ONLY as a data-curation node. Sponsor tools (all load-bearing):
**Redis** (adapter library + routing + interaction queue), **Weave/W&B** (eval + observability
+ proof charts), **CopilotKit** (React/AG-UI frontend). Base = Qwen2.5-7B-Instruct.

### Produce a spec that splits the work into EXACTLY 3 INDEPENDENT, PARALLEL tracks

Each track must be buildable **independently** against a shared contract so 3 engineers never
block each other. Define the shared contracts/interfaces FIRST (data schemas, the adapter
file format, the Redis keys, the serving API endpoints, the eval input/output formats), then
write the three tracks against those contracts. Each track must include: goal, tasks,
inputs/outputs, the interface it owns vs. consumes, a standalone test it can run with mocked
dependencies, and acceptance criteria.

Respect these hard constraints from the README (do not violate):
- NKT-Mirror is **activation gating, NOT LoRA** → serving is **custom** (load frozen base once,
  swap 100 KB gate tensors per request). vLLM/LoRAX multi-LoRA will NOT work.
- Training is **batch/overnight**, never live. Demo is **time-compressed with pre-baked adapters.**
- Build on the **INSTRUCT** model only.
- Personalize **style/preferences**, not arbitrary facts ("steer, not teach").
- Two-runtime reality: CopilotKit runtime = Node/TS, ML = Python, bridged via **AG-UI +
  LangGraph-Python SDK**.
- Data-curation quality matters more than pipeline complexity. GPT = curation node only.

### The 3 tracks (use this split)

**TRACK A — Model & Serving (Python, the riskiest, do-first internally).**
Owns: reproducing the NKT-Mirror training loop on the instruct model; the **custom adapter
serving** (frozen base loaded once + per-request 100 KB gate-tensor swap) exposed as a simple
local **inference API** (e.g. FastAPI `/generate` taking `{prompt, adapter_id}`); the adapter
**file format** + a `train_adapter(dataset) -> adapter_file` function. Defines the contracts
the others depend on (adapter format, inference API schema). Standalone test: train one
adapter on a tiny dataset, serve base vs adapter, confirm different outputs. Reuse the
`nkt-mirror-test` repo.

**TRACK B — Data, Orchestration & Eval (Python).**
Owns: data pipeline (load Stack Exchange / support data → per-user-or-category train/held-out
splits → clean training pairs); the **GPT curation node**; the **LangGraph** nightly-batch
graph (collect→curate→train→eval→store) calling Track A's `train_adapter`; the **Weave eval**
producing held-out perplexity, the **cross-user confusion matrix**, the context-memory
baseline comparison, and the NKT-vs-LoRA size chart; and the optional **fact-capacity test**.
Consumes: Track A's `train_adapter` + inference API (mock both until ready). Standalone test:
run the full graph on mock adapters and emit a confusion matrix.

**TRACK C — Frontend, Redis & Integration (TS/Node + light Python).**
Owns: the **CopilotKit** React app (chat + dashboard showing adapter library, confusion-matrix
heatmap, base-vs-adapter examples, size chart); the **AG-UI ↔ LangGraph-Python** wiring; the
**Redis** layer (adapter blob store, vector routing user/query→adapter_id, interaction queue)
with a clean client API both other tracks use; and the end-to-end demo glue + run-of-show.
Consumes: Track A inference API + Track B eval artifacts (mock with fixtures until ready).
Standalone test: UI renders with mock data; Redis store/fetch/route round-trips a dummy 100 KB
blob.

### Sequencing & integration
- Define shared contracts in a short "Track 0 / interfaces" section first.
- Each track works against mocks; integration milestone = Track A serving + Track B eval +
  Track C UI/Redis wired on real adapters.
- Call out the critical path (Track A serving / R1, R2) and what each track demos alone if
  integration slips.

Output: the interface/contract section, then the 3 tracks with the detail above, then a short
integration & demo plan. Keep it concrete and buildable in ~1.5 days by 3 people.

---

## APPENDIX — Proposed shared contracts (give these to Spec Mode as the starting point)

Spec Mode should treat these as the v0 interfaces. All three tracks code against them and
mock what they don't own. Lock these in the first 30 minutes; change only by team agreement.

### 1. Adapter file format (owned by Track A)
- One adapter = a single file `adapter_<id>.safetensors` (~100 KB) holding the NKT-Mirror
  gate tensors, plus a sidecar `adapter_<id>.json` metadata:
  ```json
  {
    "adapter_id": "stackexchange_cooking_v3",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "unit_type": "category",          // "category" | "user"
    "unit_label": "cooking",
    "train_rows": 812,
    "trained_at": "2026-06-06T21:00:00Z",
    "day_index": 3,                    // for the time-compressed demo
    "size_bytes": 102400
  }
  ```

### 2. Inference API (owned by Track A, consumed by B & C)
- `POST /generate` → `{ "prompt": str, "adapter_id": str | null, "max_new_tokens": int }`
  returns `{ "text": str, "tokens": int, "latency_ms": int }`. `adapter_id: null` = base model.
- `POST /score` → `{ "prompt": str, "target": str, "adapter_id": str | null }`
  returns `{ "perplexity": float, "nll": float }`  (this powers the eval — keep it cheap).
- `GET /adapters` → list of loaded adapter_ids.
- `train_adapter(dataset_path, unit_label, unit_type) -> adapter_path` (Python function, also
  callable as `POST /train` for the batch job).

### 3. Redis layout (owned by Track C, consumed by A & B)
- `adapter:blob:<adapter_id>` → the safetensors bytes (or a path if blobs kept on disk).
- `adapter:meta:<adapter_id>` → JSON metadata (above).
- `adapter:index` → vector index of `unit_label` embeddings for routing.
- `route(query_or_user) -> adapter_id` helper (vector search, top-1).
- `interactions:<unit_label>` → list/stream of the day's raw interactions (the batch input).

### 4. Dataset / training-pair schema (owned by Track B)
- Curated training row: `{ "prompt": str, "completion": str, "unit_label": str }` (JSONL).
- Held-out eval row: same shape, in a separate file; train/held-out MUST NOT overlap.

### 5. Eval artifact schema (owned by Track B, consumed by C for the dashboard)
- `eval_results.json`:
  ```json
  {
    "perplexity": { "base": 12.4, "adapter": 8.1, "context_memory": 8.3 },
    "confusion_matrix": { "labels": ["cooking","diy","money"], "matrix": [[..],[..],[..]] },
    "size_bytes": { "nktmirror": 102400, "lora": 161480704 },
    "examples": [ { "prompt": "...", "base": "...", "adapter": "...", "reference": "..." } ]
  }
  ```

## APPENDIX — Milestone timeline (~1.5 days, 3 people)

- **Hour 0–0.5 (all):** lock the contracts above. Agree on base model + 3–4 units (categories).
- **Hour 0.5–4:** A reproduces NKT-Mirror on instruct + trains ONE real adapter (kills R1/R2
  early). B builds data pipeline + curation + produces train/held-out JSONL. C scaffolds
  CopilotKit app + Redis client, both on mock fixtures.
- **Hour 4–10:** A finishes serving API + `train_adapter`. B builds LangGraph batch graph +
  Weave eval against A's API (mock until A ready) and emits `eval_results.json`. C wires
  AG-UI↔LangGraph and builds the dashboard against the eval-artifact schema.
- **Integration milestone (~hour 10–14):** swap mocks for real — A's API + B's adapters/eval +
  C's UI/Redis end to end. Pre-bake adapters for "days" 1→3.
- **Hour 14+:** polish the confusion-matrix heatmap, rehearse the run-of-show, record a backup
  video.

## APPENDIX — Fallback (what each track demos alone if integration slips)
- **Track A alone:** CLI showing base vs one adapter on the same prompt → visibly different,
  more on-style output. Proves the method works.
- **Track B alone:** the **confusion-matrix heatmap + perplexity chart** in Weave from
  pre-trained adapters. This is the core proof and can stand without the UI.
- **Track C alone:** CopilotKit UI with mock `eval_results.json` + Redis adapter library
  view. Proves the product story.
- **Golden rule:** Track B's confusion matrix is the single most important artifact — if only
  one thing works, make it that.
