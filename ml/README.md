# WeaveSelf — Python ML package

This package hosts the Python (Track A / Track B) side of WeaveSelf: serving, training,
data pipeline, orchestration, eval, and the shared Track 0 data contracts.

## Layout

```
weaveself/
  contracts/      # Track 0 shared data models (adapter file format, schemas)
  serving/        # Track A serving engine + inference API (+ server entrypoint)
  training/       # Track A NKT-Mirror train_adapter
  data/           # Track B data pipeline + curation
  orchestration/  # Track B LangGraph nightly-batch graph
  eval/           # Track B Weave eval
scripts/          # runnable entrypoints (prepare_demo)
tests/            # Python test suite
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The contracts/engine/tests install is light (no torch). To run the **real**
model serving stack you also need the heavy serving extra:

```bash
pip install -e ".[dev,api,serving]"
```

This installs `fastapi`, `uvicorn`, `torch`, and `transformers`. The default
`StubBackend` keeps the unit tests runnable without a GPU; the real
`HFBackend` is only loaded when you explicitly run the server / demo below.

---

## Running the REAL stack

Configuration is read from the repo-root `.env` (loaded with `python-dotenv`);
explicit environment variables and CLI flags take precedence. The relevant
Track A settings:

```dotenv
BASE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
WEAVESELF_BACKEND=hf          # hf = real model; stub = dependency-free fake
TORCH_DEVICE=cuda             # cuda | cuda:0 | cpu | mps
MODEL_DTYPE=bfloat16          # bf16 keeps the 1.5B model inside ~6 GB VRAM
ADAPTERS_DIR=./data/adapters
INFERENCE_API_HOST=127.0.0.1
INFERENCE_API_PORT=8000
REDIS_URL=redis://127.0.0.1:6379
EVAL_RESULTS_PATH=./data/eval_results.json
```

### 1. Start Redis

A live Redis lets the demo store adapters in a keyspace shared with the Track C
frontend. With Docker:

```bash
docker run -d --name weaveself-redis -p 6379:6379 redis:7
# verify
docker exec weaveself-redis redis-cli ping   # -> PONG
```

Point `REDIS_URL` at it (e.g. `redis://127.0.0.1:6379`). If Redis is
unreachable, the Python client transparently falls back to a JSON file
(`data/redis_store.json`) that honors the *same* key layout — useful offline,
and documented rather than silent.

### 2. Prepare the demo (real adapters + eval_results.json)

Trains a real NKT-Mirror adapter per demo Unit (`cooking`, `fitness`,
`finance`), stores each into Redis, loads the real Base_Model once, runs the
Weave eval (held-out perplexity base-vs-adapter, cross-unit confusion matrix,
NKT-Mirror-vs-LoRA size chart, base-vs-adapter generation samples), and writes
`eval_results.json`:

```bash
cd ml
python -m scripts.prepare_demo
# equivalently: python scripts/prepare_demo.py

# useful flags:
python scripts/prepare_demo.py --device cuda --dtype bfloat16 --max-new-tokens 24
python scripts/prepare_demo.py --units cooking fitness         # subset
python scripts/prepare_demo.py --stub                          # no GPU / no download
```

Outputs:

- `data/adapters/adapter_<id>.safetensors` + `.json` — the real Adapter_File pairs.
- adapters + metadata stored in Redis (`adapter:blob:*`, `adapter:meta:*`, `adapter:index`).
- `data/eval_results.json` — the schema-conformant eval artifact the dashboard renders.

On a 6 GB GPU the 1.5B model loads in bfloat16 and the eval runs forward-only
(scoring + short generations), so it completes quickly. If the real model
cannot load (no VRAM / no network for the first download), the script prints
exactly why and falls back to the `StubBackend` so the pipeline still produces
an artifact.

### 3. Run the Inference_API server

```bash
cd ml
python -m weaveself.serving          # or: python -m weaveself.serving.server
# or the console script:
weaveself-serve
```

This loads the Base_Model **exactly once** into a single resident
`ServingEngine`, enables CORS for the local UI origins, and serves on
`INFERENCE_API_HOST:INFERENCE_API_PORT` (default `127.0.0.1:8000`). Endpoints:

- `POST /generate` — `{prompt, adapter_id?, max_new_tokens}` → text under an adapter (null adapter = pure base).
- `POST /score`    — `{prompt, target, adapter_id?}` → teacher-forced `nll` + `perplexity`.
- `GET  /adapters` — list of loadable `adapter_id` values found in `ADAPTERS_DIR`.
- `POST /train`    — `{dataset_path, unit_label, unit_type}` → trains an adapter.

Quick check once it is up:

```bash
curl http://127.0.0.1:8000/adapters
curl -X POST http://127.0.0.1:8000/generate \
  -H "content-type: application/json" \
  -d '{"prompt":"How do I improve my sauce?","adapter_id":null,"max_new_tokens":24}'
```

Pass a real `adapter_id` (from `/adapters`) to see the adapter steer the output
relative to the base.
