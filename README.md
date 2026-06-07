# WeaveSelf

**Local, overnight "weight-memory" personalization.** Instead of remembering you by
re-injecting a growing transcript into the context window every message (what ChatGPT/Claude
memory do), WeaveSelf *consolidates* your day's conversations into a tiny (~40–100 KB) adapter
that lives in the model's weights — then **deletes the raw chat logs**. Your personality becomes a
small file you own, costs zero extra context tokens at inference, can't "drift," and runs on a
local open model.

> The core idea: **context is not a memory problem, it's a weight problem.** We bake *how you like
> things* (style, preferences, voice) into the weights overnight — like sleep — rather than storing
> and re-reading text forever.

---

## How it works (the loop)

```
You chat  ──►  Inference_API (frozen base model + your gate adapter, on GPU)
   │ every turn is logged
   ▼
Redis:  interactions:<unit>      ← the day's raw chats (temporary)

── once a day, the "consolidation" job runs ──
collect (Redis)                         the new day's chats only
   └► curate (OpenAI → clean pairs; local fallback if offline)
        └► train (NKT-Mirror gates, warm-started from yesterday's adapter,
                  anchored to it so earlier days aren't forgotten)
             └► EVAL-GATE (did today's held-out improve? did it drift too far?)
                  └► promote the new adapter ONLY if it's better
                       └► DELETE the day's raw chats   ← memory now lives in the weights
                            └► log everything to Weave (what it learned, drift, decision)
```

Next day, the same loop **warm-starts from yesterday's adapter** — so day 2's model is day 1's
adapter plus one more night of learning, and the conversations that produced it are gone.

### What makes it different from context-window memory
- **Zero per-turn token cost** — the personality is in the weights, not re-read each message.
- **No context rot / drift** — weights don't degrade as history grows.
- **Private & portable** — a small adapter file you own; runs on a local open model.
- **Scales** — a tiny adapter per user/category is feasible at scale (unlike stuffing context or
  154 MB LoRAs).
- **You can measure the learning** — because we actually train, Weave shows what was learned each
  night and refuses to ship a worse model. Context-memory systems can't do this (they don't train).

Honest scope: this personalizes **style/preferences**, not arbitrary **facts**. For facts, pair it
with RAG. It is not trying to be a better general assistant — it's a privacy-first, local,
zero-context-cost personalization layer.

---

## The method: NKT-Mirror (activation gating, not LoRA)

An adapter is a small set of **per-channel multiplicative gates** applied to the output of a few
decoder-layer MLP blocks of a **frozen** instruct model. Only the gate vectors are trained
(~thousands of numbers, ~40 KB); the base model is never modified. Gates "steer" features the model
already computes — which is why it's great for style and weak for new facts.

Continual learning without storing data: each night the trainer **warm-starts from the previous
adapter's gates** and **regularizes toward them** (an anchor), so prior days survive in the weights
without replaying old text. "Forgetting" is monitored as **gate drift** (how far the new gates moved
from yesterday's) — measured without keeping any past chats.

---

## Architecture & sponsor tools

| Component | What it does | Sponsor tool |
| --- | --- | --- |
| **Inference_API** (`ml/`, FastAPI) | Loads the frozen base model **once** on the GPU; serves chat (`/generate`), scoring (`/score`), and adapter listing. Applies a unit's gate adapter per request. | — (Qwen2.5 open model) |
| **Chat + Dashboard** (`app/`, React) | Chat with the model; dashboard shows learning metrics. Chat runs through a real CopilotKit runtime that proxies to the Inference_API. | **CopilotKit** |
| **Redis** (Docker) | The interaction log (`interactions:<unit>`), the adapter store (`adapter:blob/meta:<id>`), the vector router (`adapter:index`), and the current-adapter pointer. Raw chats are deleted after consolidation. | **Redis** |
| **Consolidation job** (`ml/scripts/consolidate.py`) | The nightly collect → curate → train → eval-gate → promote → delete-logs loop, orchestrated as a pipeline. | **LangGraph** (orchestration) |
| **Curation** | Turns raw chats into clean training pairs; OpenAI when reachable, local heuristic fallback otherwise. | **OpenAI** |
| **Observability** | Logs each consolidation run: consolidation score, gate drift, gate deviation, curation yield, and the promote/reject decision. Eval traces + published artifacts. | **Weave / W&B** |

---

## Repo layout

```
ml/                                  Python: model, training, consolidation, eval
  weaveself/
    serving/        Inference_API (FastAPI) + serving engine + model backends
    training/       NKT-Mirror trainer (nkt_trainer.py = real gradient training)
    consolidation/  the nightly consolidate loop (consolidate.py)
    data/           curation node (OpenAI + local fallback) + data pipeline
    eval/           Weave eval + WeaveLogger
    integration/    Redis client (interaction log, adapter store, router)
    contracts/      shared data schemas (adapter file, training pair, eval results)
  scripts/          consolidate.py (nightly job), run_weave_eval.py
  tests/            Python test suite
app/                                 Node/TS: CopilotKit React app + runtime
  src/components/   ChatView, DashboardView (React)
  src/server/       CopilotKit runtime + service adapter (proxies to /generate, logs to Redis)
  src/redis/        Redis client (TS) used by the runtime
  src/frontend/     framework-independent chat + dashboard logic
data/                                generated adapters + eval artifacts (gitignored)
.env                                 configuration (gitignored; see .env.example)
```

---

## Running it locally

### Prerequisites
- Python 3.12, Node 20+, Docker.
- A GPU is recommended for the real model. Configured for **Qwen2.5-1.5B-Instruct** (fits ~6 GB in
  bfloat16); set `BASE_MODEL_ID` to a 7B variant if you have more VRAM.
- `cp .env.example .env` and fill in what you need (`OPENAI_API_KEY`, `WANDB_API_KEY` for Weave).

### 1. Start Redis
```bash
docker run -d --name weaveself-redis -p 6379:6379 redis:7-alpine
```

### 2. Install deps
```bash
# Python
cd ml && pip install -e ".[dev,api,serving,eval]"
# Node
cd ../app && npm install
```

### 3. Start the model server (loads the base model once, on the GPU)
```bash
cd ml
python -m weaveself.serving        # serves http://127.0.0.1:8000
```
Requires `WEAVESELF_BACKEND=hf` in `.env` (the server refuses to run a stub in production).

### 4. Start the chat UI
```bash
cd app
npm run dev                        # http://localhost:3000
```
Pick a Unit, chat with the model. Every turn is logged to Redis under `interactions:<unit>`.
With no adapters yet, the dropdown is just "Base model".

### 5. Consolidate (the "nightly" job)
After a day of chatting (stop the model server first on a single-GPU machine to free VRAM):
```bash
cd ml
python -m scripts.consolidate      # all units with interactions
```
This curates the day's chats, trains/updates each unit's adapter (warm-started from the previous
one), promotes it only if it improved, **deletes the raw chats**, and logs the run to Weave. Restart
the server and the new adapter is served automatically.

### 6. (Optional) Stand-alone eval to Weave
```bash
cd ml
python -m scripts.run_weave_eval   # scores adapters, logs perplexity + confusion matrix to Weave
```

---

## Current state (honest)

**Working & verified:**
- Real Qwen2.5-1.5B serving on GPU; chat through the real CopilotKit UI.
- Chat turns logged to Redis; raw logs deleted after consolidation.
- Real NKT-Mirror gradient training; adapters measurably beat the base on held-out perplexity.
- Data-free continual learning: day N warm-starts from day N-1's adapter; verified across two days.
- Weave logging of consolidation/drift/yield/decision (produces live run + trace URLs).

**By design, empty until you use it:**
- No adapters exist until you chat and run a consolidation pass.
- OpenAI curation is used when reachable; it falls back to a local curator when offline (so a
  network blip never drops your data).

**Known limitations / next steps:**
- Personalizes style/preferences, not arbitrary facts (pair with RAG for facts).
- The advanced forgetting guard (per-channel NTK/Fisher-weighted anchoring) is a future upgrade;
  today it uses warm-start + uniform anchor + gate-drift monitoring.
- Tested on 1.5B locally; scale to 7B on a larger GPU for sharper results.

---

## Configuration (`.env`)

Key variables (see `.env.example` for the full list):

| Variable | Meaning |
| --- | --- |
| `BASE_MODEL_ID` | Instruct base model (default `Qwen/Qwen2.5-1.5B-Instruct`) |
| `WEAVESELF_BACKEND` | `hf` for the real model (required by the server) |
| `TORCH_DEVICE` / `MODEL_DTYPE` | `cuda` / `bfloat16` |
| `REDIS_URL` | Redis connection (default `redis://127.0.0.1:6379`) |
| `ADAPTERS_DIR` | Where adapter files are written/served |
| `INFERENCE_API_URL` | Where the UI/runtime reach the API |
| `OPENAI_API_KEY` / `CURATION_MODEL` | OpenAI curation (optional; local fallback otherwise) |
| `WANDB_API_KEY` / `WEAVE_PROJECT` | Weave/W&B observability |

`.env` is gitignored — secrets are never committed.
