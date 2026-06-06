# Personalization (Memory) Track — Build Spec

This is the contract for the **memory / personalization** track so it composes
cleanly with the **tooling / governance** track (Tracks A + B). Read the root
`README.md` first for the system overview.

## TL;DR

A personalization adapter is **the same object as a tool skill**: a ~200 KB
NTK-Mirror controller. The unique operations (`compose` = add, `subtract` =
revoke) work across *any* controllers, so a user's effective model is:

```
base ⊕ user_style[user_id] ⊕ (the capabilities they're authorized for)
```

Two orthogonal axes — **who you are** (style) and **what you can do** (tools) —
combined by one `compose()` call in the control plane.

## The hard rule (read this first)

**Do not train models yourself.** Composition requires every controller to come
from the *same* base model and layer config — `compose_states` rejects mismatched
`hidden_size` / `n_layers` / `layer_path`. So:

> The memory track produces **styled training examples**. The control plane
> (Track A) mints, stores, composes, and governs the controller.

This removes all compatibility risk and keeps your scope tight.

## What you own

1. **Preference extraction** — turn a user's interactions into `(prompt, completion)`
   examples that demonstrate their *style/behavior* (HOW), not facts (WHAT).
   This is where OpenAI fits: summarize the user's style, then synthesize styled
   example pairs.
2. **Update cadence** — decide when to re-mint (every N turns / T minutes /
   session end). Using the adapter is free per request; the re-fit is the
   ~36 s cold-path job. Never re-train on the hot path.
3. **Acceptance metric** — base vs personalized on held-out prompts (LLM-judge or
   rubric) to prove the adapter actually shifted style.

## Steer, not teach (the constraint that decides success)

NTK-Mirror rescales what the base already knows. It is good at HOW, unreliable at
WHAT. Stay in the HOW lane:

- **Encode (HOW):** tone, verbosity, format (bullets vs prose), persona, structure,
  recurring preferences, domain conventions.
- **Do NOT encode (WHAT):** facts — "my address is…", "my flight is on the 9th",
  "order #12345". Those are unreliable in weights; use RAG/context for them.
  If you put facts in, the demo flat-lines and a judge will catch it.

Keep it to ~10–30 focused styled pairs so the artifact stays ~200 KB.

## The seam (the only two API calls you need)

The control plane exposes these so you never touch the model, gates, or
composition. (Both are live on the `controller-engine` branch.)

### 1. Mint / update a user's personalization adapter

```
POST {CONTROL_PLANE}/personalize
{
  "user_id": "u_123",
  "examples": [
    {"prompt": "User: summarize this PR.\nAssistant:", "completion": " - point one\n - point two"},
    {"prompt": "User: explain the bug.\nAssistant:", "completion": " TL;DR: <one line>. Details: ..."}
  ]
}
-> { "user_id": "u_123", "controller_id": "...", "loss_first": ..., "loss_last": ..., "artifact_bytes": ... }
```

Call this on your update cadence. Each call re-fits and swaps in the new version
for that user.

### 2. Open a session that composes style + authorized tools

```
POST {CONTROL_PLANE}/session
{ "principal": "exec-assistant", "skills": ["weather", "calendar"], "user_id": "u_123" }
-> { "session_id": "...", "authorized": [...], "personalized": true, "controller_id": "..." }
```

The control plane composes `[user_style[u_123], *authorized_skill_controllers]`
into the session controller. From there, `/act` runs inference with the user's
style baked in and the tools governed — the personalization adds zero context
cost and zero per-turn latency.

## Definition of done (your acceptance test)

Using the style + tool composition (validated by `smoke_style_plus_tool.py`):

1. `compose([user_style[uid], weather_tool])` → `weather(...)` still fires on tool
   prompts **and** the style shows on general prompts.
2. `subtract(user_style[uid])` → tool remains, persona reverts.
3. base vs personalized → measurable style-match gain on held-out prompts.

## Why this combination is strong (for the pitch)

The personalization-via-adapter idea has prior art (Doc-to-LoRA, context
distillation, "Towards Million Personal Models"). What none of them have is
**composition + revocation + governance**, which is exactly the tooling track:

> "The field says 'millions of personal models' is the future, but nobody can
> compose or revoke them. Ours are ~200 KB — compose to grant, subtract to
> revoke — personalization and governance in the same primitive, on a local model."

Stay in the style/HOW lane, pair with RAG for facts, and that claim holds.

## Division of labor

| | owns | produces | meets at |
|---|---|---|---|
| Tooling / governance | capabilities + control plane | tool controllers, authorization, runtime guard, audit, **compose/subtract** | `compose()` in Track B |
| Memory / personalization | the user axis | styled examples + update cadence + style metric | `POST /personalize`, `user_id` on `/session` |

You never touch the model, gates, or composition. You own "interactions → styled
examples → when to refresh." The control plane owns "train → compose → govern."
