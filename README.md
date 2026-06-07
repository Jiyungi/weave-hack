# OpenMirror

**Local agent platform — overnight weight-memory for *how you talk*, governed composable skills for *what you can do*, on models you host.**

Closed APIs send your prompts to someone else's server. Context-window "memory" re-reads a growing transcript every turn — it rots, costs tokens, and drifts. Prompt guards ("don't use tool X") are jailbreakable: the capability is still in the model.

OpenMirror treats **both** personalization and permissions as the same object: **~200 KB NTK-Mirror controllers** on a frozen Qwen2.5-7B. Compose to add, subtract to revoke, mint in ~36 s. Raw chat logs consolidate into weights overnight, then get **deleted**. Your data stays on your box.

Built on [NTK-Mirror](https://github.com/leochlon/ntkmirror) skill arithmetic — LoRA can't subtract cleanly; prompts can't revoke cleanly. This can.

---

## Why not LoRA, full fine-tune, or context memory?

Most agent stacks mix **who you are** and **what you can do** into one blob — then try to govern it with prompts or policy files. That works until you need to **revoke** something cleanly.

| Approach | Personalization | Tool / capability control | Revoke a skill mid-session | Delete raw logs after learning |
|----------|-----------------|---------------------------|----------------------------|----------------------------------|
| **Growing context** | Re-read chat history every turn | System prompt + tool list | Remove from prompt only — model may still "remember" from prior turns; tokens grow forever | No — history is the memory |
| **Prompt / policy guards** | Instructions in system message | Allowlists, "don't call X" | Policy change does not remove weights; jailbreaks and instruction drift remain | N/A |
| **Single LoRA / fine-tune** | Often merged with task adapters | Capabilities baked into one delta | **Cannot subtract** — merged adapters interfere; disabling a tool in config does not un-learn it from weights | Retrain or keep data |
| **OpenMirror (NTK-Mirror)** | Separate `user_style` controller | Separate per-tool controllers (~200 KB each) | **Compose (+1) / subtract (−1)** — revoking `weather` removes that controller from the session composition; emission changes at the weight level | Yes — consolidate → mint → delete logs |

**What is novel:** personalization and authorization use the **same composable primitive**. A style controller and a `python` controller are both small NTK-Mirror weights on a **frozen** base model. At session open you add only what is authorized; on revoke you subtract that controller without retraining and without touching the user's style controller.

That is structurally different from fine-tuning:

- **Fine-tune / LoRA** merges behavior into one adapter. You cannot cleanly remove "weather" from a multi-skill LoRA without retraining or accepting capability bleed. Policy revoke is config-only — the model may still emit the tool.
- **Context memory** cannot revoke knowledge already in the transcript; you can only stop sending it, at rising cost and with leakage across turns.
- **OpenMirror** makes revoke **operational**: subtract the tool controller, runtime guard blocks the emit, audit records the block — behavior changes in the same session, demo-able on video.

Separate adapters for HOW vs WHAT (see below) avoid the usual failure mode of training one fine-tune on mixed personality + tool examples, where revoking one capability is impossible without damaging the rest.

**Sponsor integrations:** Redis (required), Weave/W&B (tracing), CopilotKit (dashboard), MCP (external tools), OpenAI optional (curation). See [**Partner technologies**](#partner-technologies) for concrete usage.

**Demo video:** [Google Drive](https://drive.google.com/drive/folders/1cMMrqpS31PtUyuELDcg8pZwPFppcR54k?usp=sharing) · **Weave project:** [wandb.ai/.../OpenMirror](https://wandb.ai/godsonajodo2020-microsoft/OpenMirror/overview)

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

## Partner technologies

What each sponsor tool is **for** in the OpenMirror stack — the infrastructure role, not an implementation walkthrough.

### Redis (required — persistence + audit)

Redis is the **system of record** for everything that must survive restarts and be shared across services. Without it, the control plane does not come up.

**What lives in Redis:**

- **Capability registry** — which tool controllers exist and which agents are allowed to use them
- **Live sessions** — what is composed into each agent session right now, including mid-session revokes
- **Personalization index** — which `user_style` controller belongs to which user
- **Memory buffer** — raw chat turns waiting for overnight consolidation (deleted after mint)
- **Approval queue** — pending capability requests when human-in-the-loop is on
- **Audit stream** — append-only log of grants, blocks, revokes, approvals, and memory events that powers the dashboard Audit panel

**Where it sits:** between the orchestrator, control plane, and dashboard. Every chat turn, tool allow/block, revoke, and consolidate flows through Redis so multiple workers and UI clients see the same governance state.

### Weave / W&B (observability + proof)

Weave is the **inspectability layer** across our distributed local stack (orchestrator → control plane → NTK engine → memory).

**What we trace:**

- Multi-agent runs (plan → delegate → worker loops → FINAL)
- Session open/compose and revoke/subtract on the control plane
- Tool emission at the runtime guard (allowed vs blocked)
- Controller minting and memory consolidation

Traces span process boundaries so a single run shows up as one tree — useful for demos, evals, and proving that grant/revoke actually changes behavior rather than only changing config. Project: [OpenMirror on W&B](https://wandb.ai/godsonajodo2020-microsoft/OpenMirror/overview)

### CopilotKit (dashboard + human-in-the-loop)

CopilotKit powers the **operator-facing agent UI** on top of our custom Next.js dashboard.

**What it provides:**

- The conversational sidebar for driving the platform in natural language
- Live read access to governance state, audit tail, tool catalog, and worker policies
- Callable actions that mirror the dashboard: seed skills, set policies, open sessions, run the orchestrator, register tools, revoke capabilities

The main product surface (chat, capabilities, approvals, audit, memory) is custom UI; CopilotKit adds an agent-native control channel wired to the **local 14B brain**, not a cloud default. Human approve/deny for new capabilities works the same whether the request came from chat, the orchestrator, or the copilot.

### MCP (external tool plane)

MCP is how **third-party and out-of-repo tools** enter the same governed pipeline as built-in skills (weather, python, stock_price).

**Infrastructure role:**

- Discover tools advertised by an external MCP server
- Register them as first-class executors at runtime (no redeploy)
- Mint a narrow NTK controller for each, grant to role workers, enforce via the same control plane and audit path as native tools

This lets OpenMirror grow its capability surface through standard MCP servers rather than hard-coding every integration. HTTP URL adapters use the same registration path for non-MCP endpoints.

### OpenAI (optional — memory curation)

OpenAI is used **only upstream of personalization**, not for runtime agent inference (that stays on local Qwen2.5).

**Infrastructure role:**

- After users chat, raw `(user, assistant)` turns sit in Redis
- Before minting a `user_style` controller, an optional OpenAI pass cleans and formats those logs into training pairs (tone/format, not new facts)
- If OpenAI is unavailable, a local heuristic curator runs instead; consolidation and local mint proceed either way

Runtime planning, tool emission, and orchestration remain fully local-first.

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
