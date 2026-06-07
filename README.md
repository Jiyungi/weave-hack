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

Everything runs on a single GPU box (an A100/H100; tested on Brev). On a **fresh** box:

```bash
bash setup_brev.sh
```

That creates a venv, installs deps, clones NTK-Mirror to `~/ntkmirror_src` (its upstream pip packaging is broken, so it's used from a clone), installs a CUDA-12.8 torch build, and pre-fetches the 7B weights. In every shell after, activate the venv first:

```bash
source ~/venv/bin/activate
```

### Validate the operations (no service needed)

```bash
python smoke_compose_subtract.py                                  # 0.5B, fast
PEFT_CMP_MODEL=Qwen/Qwen2.5-7B SMOKE_STEPS=600 SMOKE_MAX_LOG_GATE=0.1 \
  SMOKE_GATES=10000 SMOKE_LR=8e-3 python smoke_compose_subtract.py  # real 7B
```

### Bring up the services (each in its own shell, venv activated)

```bash
# Track A — controller engine
uvicorn controller_service:app --host 0.0.0.0 --port 8000

# Track B — control plane (governance API; landing page at / points to Track C)
uvicorn control_plane_service:app --host 0.0.0.0 --port 8100

# Track C — CopilotKit control surface (Next.js UI)
cd ui && cp .env.example .env.local && npm install
npm run dev   # http://localhost:3000

# Track D — agent orchestrator (real tool-using agents governed by Track B)
uvicorn agent_service:app --host 0.0.0.0 --port 8200

# Brain (shared by Track C CopilotKit chat + Track D agents; local vLLM default):
pip install vllm openai
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8001 \
    --max-model-len 8192 --gpu-memory-utilization 0.45
```

Track C is a **Next.js + CopilotKit** app in `ui/` (port **3000**): governance
panels (seed, session, revoke, act, audit) plus a **CopilotSidebar** chat wired
to local vLLM. Every action is also a typed CopilotKit action (`seed_demo`,
`open_session`, `run_orchestrator`, `register_tool`, …). Same-origin proxies
hit Track B (:8100) and Track D (:8200) — only port 3000 needs port-forwarding.

Track D is the agent layer (port 8200): a planner orchestrator decomposes a
task and delegates to worker agents whose tool capabilities are governed at the
weight level by Track B. The reasoning brain is pluggable (env-swappable to any
OpenAI-compatible endpoint, including real OpenAI). See `agents/` for the
brain, tools, governed ReAct loop, and orchestrator.

### Run the verifications (third shell)

```bash
python verify_service.py        # Track A: grant/revoke over HTTP
python verify_risks.py          # Track A: erase-vs-reduce, forgetting, jailbreak
python verify_control_plane.py  # Track B: policy, runtime guard, revoke + audit
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
- Governance state in Track B is in-memory (single-process demo). Controllers persist on disk; restart re-reads them.
