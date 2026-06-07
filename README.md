# OpenMirror

**Local agent platform — overnight weight-memory for *how you talk*, governed composable skills for *what you can do*, on models you host.**

Closed APIs send your prompts to someone else's server. Context-window "memory" re-reads a growing transcript every turn — it rots, costs tokens, and drifts. Prompt guards ("don't use tool X") are jailbreakable: the capability is still in the model.

OpenMirror treats **both** personalization and permissions as the same object: **~200 KB NTK-Mirror controllers** on a frozen Qwen2.5-7B. Compose to add, subtract to revoke, mint in ~36 s. Raw chat logs consolidate into weights overnight, then get **deleted**. Your data stays on your box.

Built on [NTK-Mirror](https://github.com/leochlon/ntkmirror) skill arithmetic — LoRA can't subtract cleanly; prompts can't revoke cleanly. This can.

---

## Adapter model (read this — we are explicit)

OpenMirror creates **separate adapters**, not one adapter that learns tools and personality together.

| Adapter type | ID pattern | Trained on | Scope | Updates |
|--------------|------------|------------|-------|---------|
| **Personalization** | `user_style-{user_id}` | Styled chat pairs (tone, format, verbosity — **HOW**) | **Broad** — biases every reply | Overnight consolidation → `POST /personalize` |
| **Tool** | `weather`, `python`, `arxiv_search`, … | Tool-call pairs (`weather("Paris")`) — **WHAT you may emit** | **Narrow** — fires on matching prompts only | Seed, register MCP, or self-improvement → `POST /skills` |

**We do not** train a single adapter on mixed tool + style examples. That would break the product:

- You couldn't **revoke `weather`** without retraining or wiping the user's style.
- Tool mint and memory consolidation run on **different cadences** (~36 s on demand vs nightly batch).
- **Governance** only applies to tools (runtime guard checks tool names). Style rides in the composed controller but is not a "callable skill."

**At session time**, Track B **composes** the separate stored adapters into one session controller:

```
session_controller = compose([ user_style[u_123], weather, calculator, python ])
                       ────────────────────────   ─────────────────────────────
                              1 adapter                    N adapters (per policy)
```

That composition is **runtime arithmetic** (~free), not a new trained file. The underlying adapters stay separate on disk (~200 KB each). Revoke subtracts one tool controller; personalization stays. Consolidation replaces only `user_style-u_123`.

Validated by `smoke_style_plus_tool.py`: style + tool compose without interference; subtract style → tool remains; subtract tool → style remains.

---

## Two axes, one engine

**Two adapter types. Composed at session time. Never merged into one training run.**

```
stored on disk:
  user_style-alice.pt     ← personalization (1 per user)
  weather.pt              ← tool adapters (1 per skill)
  python.pt
  arxiv_search.pt

at session open:
  session = compose([ user_style-alice, weather, python ])   ← ephemeral bundle
```

| Axis | Adapter type | What it learns | Revocable alone? |
|------|--------------|----------------|------------------|
| **Memory** | Personalization (`user_style-{user_id}`) | HOW you talk | Yes — subtract style, tools unchanged |
| **Capabilities** | Tool (`weather`, `python`, …) | WHAT calls you may emit | Yes — subtract tool, style unchanged |

Same NTK-Mirror object (~200 KB controller). Same `compose()` / `subtract()` math. **Different training data, different lifecycle, different governance role.**

---

## Architecture

![OpenMirror architecture](docs/architecture.png)

| Track | Port | Role |
|-------|------|------|
| **C — UI** | 3000 | Chat, agents, memory dashboard, policies, approvals, audit |
| **D — Agents** | 8200 | Orchestrator + governed workers; brain proposes, control plane decides |
| **B — Control plane** | 8100 | Policy, sessions, `/personalize`, runtime guard, Redis |
| **A — Engine** | 8000 | NTK train / compose / act on frozen Qwen2.5-7B |
| **Brain** | 8001 | Qwen2.5-14B via vLLM — reasoning, swappable, ungoverned |
| **Memory job** | — | Consolidation: collect → curate → mint style → eval-gate → delete logs *(integrating from `main`)* |

**Defense in depth:** (1) model-level — session controller only emits granted skills; (2) runtime — guard blocks unauthorized calls even if the model leaks.

**Weave:** traces across memory consolidation, agents, control plane, and Track A.

---

## What we prove

Real Qwen2.5-7B, no mocks:

| Claim | Script |
|-------|--------|
| Grant / revoke / reversibility | `smoke_compose_subtract.py`, `verify_service.py` |
| Style + tool compose without interference | `smoke_style_plus_tool.py` |
| Revocation erases (doesn't just reduce) | `verify_risks.py` |
| Policy → compose → block → revoke E2E | `verify_control_plane.py` |

**~36 s** to mint a controller · **~200 KB** per skill · compose/revoke is **free** (gate arithmetic).

---

## Live demo

1. **Memory** — Chat as a user → run consolidation → style shifts; raw logs deleted. Memory is in the weights.
2. **Governance** — Orchestrator delegates → revoke `weather` → governed model stops emitting it.
3. **Both** — Session with `user_id` + tools → your style **and** only allowed capabilities. Self-acquire `python` (sensitive → human approve → mint → run locally).

---

## Run it

```bash
cd ~/weave-hack && bash setup_brev.sh    # once
bash start_all.sh                        # brain, A, B, D, UI
bash start_all.sh attach                 # logs
```

Laptop: `brev port-forward <instance> --port 3000:3000` → **http://localhost:3000**

```bash
python verify_service.py && python verify_control_plane.py   # smoke, no UI
```

---

## Config (essentials)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENMIRROR_BRAIN_BASE_URL` | `http://localhost:8001/v1` | Brain — keep local for privacy |
| `REDIS_URL` | _(unset)_ | Governance state, audit, interaction logs |
| `TRACK_A_URL` / `CP_URL` | `:8000` / `:8100` | Engine + control plane |
| `WEAVE_DISABLE` | off | Set `1` to disable W&B traces |

Controllers persist in `./controllers/`. See `.env.example` and `PERSONALIZATION.md` for the memory integration contract.

---

## Repo map

```
engine/ + controller_service.py       Track A — NTK train / compose / act
control_plane/ + control_plane_service.py   Track B — governance + /personalize
agents/ + agent_service.py            Track D — orchestrator, tools, self-improvement
ui/                                   Track C — unified dashboard
ml/weaveself/                         Memory — consolidation + curation *(from main, integrating)*
PERSONALIZATION.md                    Memory ↔ governance contract
```

---

## Honesty

- Style/memory personalizes **HOW**, not arbitrary facts — pair with RAG for WHAT.
- Tool skills use narrow synthetic call formats so results are attributable.
- External tools egress when invoked; models stay local.
- `python` is timeout-bounded subprocess + human approval, not a hardened sandbox.
- Memory consolidation loop is **landing** from `main`; governance path is **live today**.

---

## Thesis

**Who you are and what you can do should both be small composable weight controllers — grantable, revocable, consolidatable — on hardware you control. Not prompts. Not context stuffing. Not someone else's API.**
