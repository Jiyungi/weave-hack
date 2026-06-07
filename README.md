# OpenMirror

**Local agent platform — overnight weight-memory for *how you talk*, governed composable skills for *what you can do*, on models you host.**

Closed APIs send your prompts to someone else's server. Context-window "memory" re-reads a growing transcript every turn — it rots, costs tokens, and drifts. Prompt guards ("don't use tool X") are jailbreakable: the capability is still in the model.

OpenMirror treats **both** personalization and permissions as the same object: **~200 KB NTK-Mirror controllers** on a frozen Qwen2.5-7B. Compose to add, subtract to revoke, mint in ~36 s. Raw chat logs consolidate into weights overnight, then get **deleted**. Your data stays on your box.

Built on [NTK-Mirror](https://github.com/leochlon/ntkmirror) skill arithmetic — LoRA can't subtract cleanly; prompts can't revoke cleanly. This can.

**Sponsor integrations:** [Redis](https://redis.io) (required state + audit), [Weave / W&B](https://wandb.ai/godsonajodo2020-microsoft/OpenMirror/overview) (tracing + proof), [CopilotKit](https://copilotkit.ai) (agent chat + HITL in the Next.js dashboard), MCP (external tool registration), OpenAI optional (memory curation before mint).

**Demo video:** [Google Drive](https://drive.google.com/drive/folders/1cMMrqpS31PtUyuELDcg8pZwPFppcR54k?usp=sharing)

---

## Adapter model (read this — we are explicit)

OpenMirror creates **separate adapters**, not one adapter that learns tools and personality together.

| Adapter type | ID pattern | Trained on | Scope | Updates |
|--------------|------------|------------|-------|---------|
| **Personalization** | `user_style-{user_id}` | Styled chat pairs (tone, format, verbosity — **HOW**) | **Broad** — biases every reply | Consolidation → `POST /personalize` |
| **Tool** | `weather`, `python`, `arxiv_search`, … | Tool-call pairs (`weather("Paris")`) — **WHAT you may emit** | **Narrow** — fires on matching prompts only | Seed, register MCP, or self-improvement → `POST /skills` |

**We do not** train a single adapter on mixed tool + style examples.

**At session time**, the control plane **composes** separate stored adapters:

```
session_controller = compose([ user_style[u_123], weather, calculator, python ])
```

Validated by `smoke_style_plus_tool.py`.

---

## Architecture

![OpenMirror architecture](docs/architecture.png)

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · editable source: [`docs/architecture.svg`](docs/architecture.svg)

| Service | Port | Role |
|---------|------|------|
| **Dashboard** | 3000 | Chat, agents, memory panel, policies, approvals, audit |
| **Orchestrator** | 8200 | Planner + role workers (`research-agent`, `ops-agent`, `support-agent`) |
| **Control plane** | 8100 | Policy, sessions, memory log, `/personalize`, runtime guard |
| **NTK engine** | 8000 | Train / compose / act on frozen Qwen2.5-7B |
| **Brain** | 8001 | Qwen2.5-14B via vLLM — orchestration planning only (not the governed actuator) |
| **Memory** | CLI | `python -m memory.consolidate` — curate → `/personalize` → delete logs |

The frozen **7B + composed NTK controllers** govern tool emission; the **14B brain** delegates across role workers.

**WeaveSelf reference:** `ml/weaveself/` (merged from `main`) holds the original consolidation research code; OpenMirror's production path mints style controllers via the NTK engine only. Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Adapter contract: [`PERSONALIZATION.md`](PERSONALIZATION.md).

---

## Live demo

1. **Tools** — Stock price via `ops-agent`; expand delegations for ALLOWED tool steps.
2. **Governance** — Revoke a capability (session) → REQUEST → approve → tool works again; audit stream in Redis.
3. **Multi-agent** — *"Weather in Berlin and compute 15% tip on $84 with python"* → `support-agent` + `ops-agent`.
4. **Memory** — Chat with `user_id` → **Consolidate** → `user_style-{id}` minted; raw logs deleted.

**Multi-agent verification:** `python verify_orchestrator.py` (offline stub) or `--live` against `:8200`.

---

## Run it

```bash
cd ~/weave-hack && bash setup_brev.sh   # defaults to branch main
bash start_all.sh
brev port-forward <instance> --port 3000:3000   # laptop → :3000
```

Full collaborator guide (Brev setup, overnight shutdown, controller backup): [`docs/COLLABORATOR_GUIDE.md`](docs/COLLABORATOR_GUIDE.md).

Head-to-head eval (OpenMirror vs OpenClaw vs Hermes): [`docs/HEAD_TO_HEAD_EVAL.md`](docs/HEAD_TO_HEAD_EVAL.md).

Consolidate a user's chats (after logging interactions via UI or `POST /memory/log`):

```bash
python -m memory.consolidate --user alice
```

---

## Repo map

```
engine/ + controller_service.py          NTK engine
control_plane/ + control_plane_service.py    Control plane (+ memory endpoints)
agents/ + agent_service.py               Orchestrator
ui/                                      Dashboard
memory/                                  Production memory loop (log → curate → personalize)
ml/weaveself/                            WeaveSelf research code (merged from main)
PERSONALIZATION.md                       Adapter contract
```

---

## Thesis

**Who you are and what you can do should both be small composable weight controllers — grantable, revocable, consolidatable — on hardware you control.**
