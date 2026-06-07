# OpenMirror

**Grant, revoke, and acquire agent capabilities at the model level — on hardware you control.**

Prompt guards are soft: the model still *knows* how to call `weather()` or run Python; a jailbreak or a sloppy planner can still try. OpenMirror treats each tool skill as a **~200 KB NTK-Mirror controller** you **add** (grant), **subtract** (revoke), or **mint on demand** (~36 s) — then enforces a **runtime guard** as a hard backstop.

The reasoning brain (14B via vLLM) stays **self-hosted and untrusted**. Governance lives downstream on a frozen **Qwen2.5-7B** you never send customer data to.

Built on [NTK-Mirror](https://github.com/leochlon/ntkmirror) skill arithmetic: `compose([A,B],[1,1])` grants both; `compose([A+B,B],[1,-1])` revokes B losslessly. LoRA can't subtract cleanly; prompts can't revoke cleanly. This can.

---

## Why this matters

| Problem | OpenMirror |
|--------|------------|
| Closed APIs (Claude, OpenAI) exfiltrate prompts & context | Brain + governed 7B run on **your box** (vLLM) |
| "Don't use tool X" is jailbreakable | Revoked skill **can't be emitted** by the session controller |
| New capability = redeploy or hope the prompt sticks | Agent **requests** skill → human approves sensitive ones → **mint + compose** into live session |
| Agent permissions are all-or-nothing | Per-principal **policy** + per-session **composition** |

---

## Architecture

![OpenMirror architecture](docs/architecture.png)

| Track | Port | Role |
|-------|------|------|
| **C — UI** | 3000 | Run agents, policies, external tools (MCP/HTTP), capability approvals, audit |
| **D — Agents** | 8200 | Orchestrator + governed workers; brain proposes, control plane decides |
| **B — Control plane** | 8100 | Policy, sessions, runtime guard, self-improvement approvals, Redis state |
| **A — Engine** | 8000 | NTK train / compose / act on frozen Qwen2.5-7B |
| **Brain** | 8001 | Qwen2.5-14B via vLLM — reasoning only, swappable, ungoverned |

**Defense in depth:** (1) model-level — composed controller only emits granted skills; (2) runtime — guard blocks any unauthorized call even if the model leaks.

**Weave:** distributed traces across D → B → A when `WANDB_API_KEY` is set.

---

## What we prove

Real Qwen2.5-7B, no mocks. Reproduce from a clean checkout:

| Claim | Script |
|-------|--------|
| Grant / revoke / reversibility | `smoke_compose_subtract.py`, `verify_service.py` |
| Revocation erases (doesn't just reduce) | `verify_risks.py` |
| Policy → compose → block → revoke E2E | `verify_control_plane.py` |

Numbers that matter: **~36 s** to mint a skill, **~200 KB** per controller, compose/revoke is **free** (gate arithmetic).

---

## Live demo (3 minutes)

1. **Revoke** — Run orchestrator with weather; revoke `weather` mid-session → governed model stops emitting it.
2. **External tool** — Register an MCP server (e.g. arXiv) → mint controller → delegate a search task → governed call succeeds.
3. **Self-acquire** — *"Write Python to 5-color this graph and RUN it to verify adjacency: {0:[1,2,3], …}"* → worker has no `python` → **REQUEST** (sensitive) → you **Approve** → mint gate → code runs locally → verified answer.

Sensitive skills (`python`, key-backed tools) require **human approval**. Safe skills auto-approve.

---

## Run it

**One-time setup** (Brev A100 or similar):

```bash
cd ~/weave-hack && bash setup_brev.sh
```

**Start everything:**

```bash
bash start_all.sh          # tmux: brain, A, B, D, UI
bash start_all.sh attach   # logs (Ctrl-b d to detach)
```

**On your laptop:**

```bash
brev port-forward <instance> --port 3000:3000
# → http://localhost:3000
```

Only port **3000** needs forwarding — the UI proxies to B and D.

**Smoke tests** (Track A + B up, no UI):

```bash
python verify_service.py && python verify_control_plane.py
```

---

## Config (essentials)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENMIRROR_BRAIN_BASE_URL` | `http://localhost:8001/v1` | Brain endpoint — keep local for privacy |
| `REDIS_URL` | _(unset)_ | Durable governance state + audit |
| `TRACK_A_URL` | `http://localhost:8000` | Controller engine |
| `CP_URL` | `http://localhost:8100` | Control plane |
| `WEAVE_DISABLE` | off | Set `1` to disable W&B traces |

Full list in `.env.example`. Controllers persist in `./controllers/`.

---

## Repo map

```
engine/ + controller_service.py     Track A — NTK operations
control_plane/ + control_plane_service.py   Track B — governance
agents/ + agent_service.py          Track D — orchestrator + tools
ui/                                 Track C — dashboard
agents/adapters.py                  MCP + HTTP tool registration
agents/loop.py                      Governed ReAct + self-improvement
PERSONALIZATION.md                  Memory/style track (separate spec)
```

---

## Honesty

- Demo skills use narrow synthetic tool-call formats so results are attributable. Arbitrary prompt-injection robustness is future work (`verify_risks.py` tests capability-level revocation on held-out phrasing).
- External tools (MCP, weather, arXiv) **do** egress when invoked — you choose what leaves the box; the models don't have to.
- `python` runs in a timeout-bounded subprocess, not a hardened sandbox. Governance + human approval is the safety boundary.

---

## Thesis (one line)

**Agent permissions should be composable controllers you can grant, revoke, and mint under policy — on self-hosted models — not wishful thinking in a system prompt.**
