# OpenMirror — Capability Governance for Tool-Using Agents

**Grant, revoke, and compose what an agent can actually *do* — at the model level, cheaply and un-jailbreakably — using NTK-Mirror "skill arithmetic."**

Today an agent's permissions live in its prompt ("you may not use tool X"). That's a soft guard: the capability is still in the model, so it can be jailbroken. This project removes the capability from the model itself, treating each skill as a tiny composable controller you can **add** (grant) and **subtract** (revoke) — then backs it with a runtime guard as a hard boundary.

---

## The two ideas

**Operations (the novelty engine).** [NTK-Mirror](https://github.com/leochlon/ntkmirror) expresses a learned skill as a ~200 KB forward-pass controller (signed log-gates). The unique part is that these controllers behave like vectors:

- `compose([A, B], [1, 1])` → **grant**: the model does both skills.
- `compose([A+B, B], [1, -1])` → **revoke**: keeps A, removes B, **losslessly**.
- scaling a weight dials a skill up/down.

LoRA adapters don't add/subtract cleanly and are megabytes; prompt guards are jailbreakable. This arithmetic is the thing only NTK-Mirror gives you.

**Domain (what makes it matter).** Agent **tool-calling capability governance**. A "skill" is a tool-using behavior (call `weather()`, call `calendar()`, …). In agentic systems you constantly need to grant, revoke, and compose what an agent is *capable* of — per role, per session — not just what a prompt asks it to avoid.

The `weather`/`calendar` skills here are deliberately simple **stand-ins for real tools**, chosen so every success is attributable (the base model emits neither unaided).

---

## Architecture

```
request
  → [Track B] control plane
        authorization policy (principal → allowed skills)
        compose ONLY the authorized skills  ─────────────►  [Track A] controller engine
        runtime tool-call guard (hard boundary)                 train / compose / subtract
        audit every action (Redis stream or file)               + risk evals
  ← governed completion

Defense in depth:
  layer 1 (model-level): the session controller can only emit granted skills
  layer 2 (runtime):     the guard blocks any unauthorized call even if emitted
```

- **Track A — controller engine** (`engine/`, `controller_service.py`): the NTK-Mirror operations as an HTTP service over a single frozen 7B. Endpoints: `/train`, `/compose`, `/execute`, `/evaluate`, `/inspect`, `/pair`, plus the risk evals `/diagnose`, `/forgetting`, `/jailbreak`.
- **Track B — control plane** (`control_plane/`, `control_plane_service.py`): authorization, per-session capability composition, runtime guard, audit, revocation. Talks to Track A over HTTP so the tracks stay decoupled.

---

## What's proven

All on **Qwen2.5-7B** (real model, no mocks), reproducible from a clean checkout:

| claim | evidence | script |
|---|---|---|
| grant (compose) | `A+B` does both skills, 1.00 / 1.00 | `smoke_compose_subtract.py`, `verify_service.py` |
| revoke (subtract) | `(A+B)−B` = 1.00 / 0.00; reversibility cosine 1.0000 | `smoke_compose_subtract.py` |
| erase vs reduce | frozen base scores 0.00 → revocation truly *erases* | `verify_risks.py` |
| revocation holds | granted fires 1.00 across held-out surface, revoked 0.00 | `verify_risks.py` |
| no forgetting | weather controller leaves arithmetic at 1.00 → 1.00 (Δ 0.00) | `verify_risks.py` |
| governance end-to-end | policy → compose → runtime block → revoke | `verify_control_plane.py` |

Operational properties that make it practical: **~36 s** to mint a skill controller on a 7B, **~200 KB** per controller (git-storable), and compose/subtract are **free** (gate arithmetic, no training).

> Composition is clip-bounded, not capacity-bounded: composing controllers sums their signed log-gates, so the composed controller needs enough `max_log_gate` headroom to represent the sum. Both the engine and the smoke set that headroom to the worst-case `|sum|`, which is lossless.

---

## Run it

Tested on a Brev A100 80GB box. Use **one command** (`start_all.sh`) or **5 manual tabs**, plus **one port-forward** on your laptop.

### Quick start (one command on the box)

```bash
cd ~/weave-hack
bash start_all.sh          # starts everything in tmux
bash start_all.sh attach   # view logs (detach: Ctrl-b d)
bash start_all.sh status   # check which ports are up
bash start_all.sh stop     # tear down
```

Then on your laptop: `brev port-forward <instance> --port 3000:3000` → **http://localhost:3000**

Requires `tmux` (`sudo apt-get install -y tmux` on a fresh box). Track A waits for the brain; Track D waits for the control plane.

### What runs where

| Port | Service | What it does |
|------|---------|--------------|
| **3000** | Track C — `ui/` (Next.js + CopilotKit) | **Open this in your browser.** Dashboard + chat. |
| 8001 | Brain — vLLM (14B) | Reasoning for chat + agents. |
| 8000 | Track A — `controller_service` | Governed 7B model (train / compose / act). |
| 8100 | Track B — `control_plane_service` | Policies, sessions, audit, `/register`. |
| 8200 | Track D — `agent_service` | Multi-agent orchestrator + real tools. |

Track C proxies to B and D internally — you only forward **port 3000** from the box to your laptop.

---

### Step 0 — One-time setup (fresh box only)

```bash
cd ~/weave-hack
bash setup_brev.sh
```

This installs Python deps, clones NTK-Mirror, caches Qwen2.5-7B weights, and runs `npm install` in `ui/`.

---

### Step 1 — Manual: open 5 terminals on the Brev box

Skip this if you used `bash start_all.sh` above.

In **every** Python terminal, activate the venv first:

```bash
source ~/venv/bin/activate
cd ~/weave-hack
```

Then start each service in its own tab (**order matters** — brain and Track A load the GPU models):

**Terminal 1 — Brain** (start first; ~1–2 min to load 14B)

```bash
source ~/venv/bin/activate
# First time only — install the vLLM build that matches the box's CUDA.
# PyPI's default wheel targets CUDA 13 and fails on a CUDA 12.8 box
# (driver < 580 -> ImportError: libcudart.so.13). Pin the cu128 build:
VIRTUAL_ENV=~/venv uv pip install "vllm==0.22.1" --torch-backend=cu128
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 \
  --max-model-len 8192 --gpu-memory-utilization 0.45
```

Wait until you see `Uvicorn running`. The brain is **optional** — only the chat
sidebar and Track D live reasoning use it; the governance demo (A + B + UI)
runs without it.

**Terminal 2 — Track A** (governed 7B; ~17 GB VRAM)

```bash
source ~/venv/bin/activate
cd ~/weave-hack
uvicorn controller_service:app --host 0.0.0.0 --port 8000
```

**Terminal 3 — Track B** (control plane)

```bash
source ~/venv/bin/activate
cd ~/weave-hack
# optional: export WANDB_API_KEY=...   # enables Weave traces
uvicorn control_plane_service:app --host 0.0.0.0 --port 8100
```

**Terminal 4 — Track D** (agents)

```bash
source ~/venv/bin/activate
cd ~/weave-hack
uvicorn agent_service:app --host 0.0.0.0 --port 8200
```

**Terminal 5 — Track C** (UI — Node, not Python)

```bash
cd ~/weave-hack/ui
cp -n .env.example .env.local   # first time only
npm install                      # first time only
npm run dev
```

Wait for `Ready on http://0.0.0.0:3000`.

---

### Step 2 — Port-forward from your laptop

On your **Mac** (not on the box):

```bash
brev port-forward igdun8pzb --port 3000:3000
```

Leave that running. Open **http://localhost:3000** in your browser.

---

### Step 3 — Use the demo

1. In the UI, click **Seed demo** (~72s — trains `weather` + `calendar`).
2. **Open session** as `support-bot`, request both skills, enable defense-in-depth → see `authorized=[weather] denied=[calendar]`.
3. **Act console** → weather prompt → permitted; calendar prompt → blocked.
4. **Agents** panel → run orchestrator with a combined task.
5. Open the **Copilot** sidebar (chat icon) → e.g. *"seed the demo"* or *"run orchestrator on weather in Berlin"*.

---

### Smoke tests (no UI; optional)

With Track A + B up only:

```bash
source ~/venv/bin/activate
cd ~/weave-hack
python verify_service.py
python verify_control_plane.py
```

Math-only smoke (no services):

```bash
python smoke_compose_subtract.py
```

---

## Configuration

| variable | default | what it does |
|---|---|---|
| `PEFT_CMP_MODEL` | `Qwen/Qwen2.5-7B` | base model (shared, frozen) |
| `CTRL_GATES` | `10000` | gate budget per controller (7B-validated) |
| `CTRL_MAX_LOG_GATE` | `0.1` | per-channel log-scale ceiling (7B-validated) |
| `CONTROLLER_DIR` | `./controllers` | where ~200 KB `.pt` controllers persist |
| `TRACK_A_URL` | `http://localhost:8000` | where Track B reaches Track A |
| `REDIS_URL` | _(unset)_ | optional: durable shared governance state + audit stream; falls back to in-memory |
| `WEAVE_PROJECT` | `OpenMirror` | W&B Weave project for traces (set `WEAVE_DISABLE=1` to force off) |
| `CP_URL` | `http://localhost:8100` | where Track D agents reach Track B |
| `OPENMIRROR_BRAIN_BASE_URL` | `http://localhost:8001/v1` | brain endpoint (OpenAI-compatible); local vLLM by default |
| `OPENMIRROR_BRAIN_MODEL` | `Qwen/Qwen2.5-14B-Instruct` | brain model name |
| `OPENMIRROR_BRAIN_API_KEY` | `sk-no-key` | ignored by vLLM; set to a real key for OpenAI |

The `gates=10000 / max_log_gate=0.1 / steps=600 / lr=8e-3` defaults are smoke-validated on 7B; the weaker `5000 / 0.05 / 240` under-fits (≈5 % per-channel scaling is too weak to steer a 7B).

---

## Layout

```
engine/                 Track A: config, model, controllers, evals, schemas, api
controller_service.py   Track A entrypoint (uvicorn target)
control_plane/          Track B: config, track_a client, runtime guard, audit,
                        durable state (memory/Redis), store, registry, schemas,
                        api, optional Weave tracing
ui/                   Track C: Next.js + CopilotKit control surface (port 3000)
control_plane/static/dashboard.html  legacy HTML dashboard (retired; see ui/)
control_plane_service.py  Track B entrypoint (uvicorn target)
agents/                 Track D: brain client, real tools registry, governed
                        ReAct loop, multi-agent orchestrator, control-plane client
agent_service.py        Track D entrypoint (uvicorn target)
smoke_compose_subtract.py Operations smoke (compose/subtract/reversibility)
verify_service.py         Track A HTTP verification
verify_risks.py           Risk evidence (diagnose/forgetting/jailbreak)
verify_control_plane.py   Track B end-to-end governance demo
setup_brev.sh             One-shot box bootstrap
```

---

## Scope / honesty notes

- Skills are narrow synthetic tool-call formats, kept simple so results are attributable. The risk-2 ("un-revokable") test demonstrates **capability-level** revocation across held-out instances, not robustness to arbitrary prompt-injection phrasing — that stronger claim needs a skill trained on diverse phrasings and is future work.
- Governance state defaults to in-memory; set `REDIS_URL` for durable/shared state across processes. Controllers persist on disk in `CONTROLLER_DIR`.
