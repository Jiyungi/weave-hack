# WeaveSelf — Project Context & Decisions

> **Legacy context** from the original WeaveSelf hackathon track (merged into `unified`).
> For the **production OpenMirror product** (controller-engine + memory + governance),
> read the root [`README.md`](README.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first.
>
> This file remains as historical background for `ml/weaveself/` and the per-user
> weight-memory thesis — not the current four-track runtime layout.

## One-line

**A local, overnight "weight-memory" personalization engine: every user (or user-category)
gets a ~100 KB NKT-Mirror adapter that is retrained in nightly batches from their
interactions, served privately on a frozen open-source model — and we *prove* the
personalization with an objective evaluation (held-out perplexity + a cross-user
identification matrix).**

## The core thesis

Today's assistants (ChatGPT "Dreaming," Hermes, etc.) personalize by **storing text and
re-injecting it into the context window** (RAG / memory stores). That approach suffers
"context rot" and "persona drift," costs tokens every turn, and lives on someone else's
server. **Nobody mainstream personalizes by fine-tuning per user.**

Our wedge: **weight-memory.** We bake *how a user likes things* (style, preferences,
behavior) into a tiny 100 KB adapter that:
- never rots / falls out of a context window,
- costs zero extra context tokens at inference,
- is private + portable (a 100 KB file you own, runs on a local open model),
- is small enough that "a personal model per user" is actually feasible at scale
  (impossible with 154 MB LoRAs).

We are NOT trying to beat ChatGPT on remembering **facts**. We win on the **style/preference**
slice, at **scale + privacy + zero-context-cost**, for the **open-source / local-model** crowd.

## The method: NKT-Mirror (the hero, and its hard limits)

- NKT-Mirror = per-channel **activation gating** on a **frozen base** model. ~5K trainable
  params, **~100 KB artifact** (vs LoRA's ~40M params / ~154 MB). Source package:
  `github.com/leochlon/ntkmirror`. Our test harness: the `nkt-mirror-test` repo.
- Measured (Qwen2.5-7B, GSM8K, strict): on the **instruct** model it **ties LoRA/QLoRA**
  (~0.711 vs 0.703–0.713) at ~340× smaller. On the **base** model it trails (~0.625).
  => **Always build on the INSTRUCT model.**
- **Mechanism limit ("steer, don't teach"):** gating can only **reweight features the base
  already computes** — amplify/suppress existing circuits. So it handles **style,
  preferences, task-skill sharpening, and "soft facts" expressible as a bias toward known
  concepts** (e.g. "user is vegetarian/terse/a beginner"). It **cannot** reliably store
  **arbitrary new factual associations** (e.g. "order #7741", a novel name). The exact
  capacity ceiling is unknown and is worth measuring (see Track C "fact-capacity test").

## Architecture (decided)

```
        ┌─────────────── CopilotKit (React frontend, AG-UI) ───────────────┐
        │   chat with personalized assistant + dashboard (matrix, examples) │
        └───────────────────────────┬───────────────────────────────────────┘
                                     │ AG-UI (SSE)
        ┌────────────────────────────▼──────────────────────────────────────┐
        │  LangGraph orchestrator (Python, LOCAL) — runs the nightly batch:   │
        │  collect → curate → train → eval → store                           │
        │   - GPT (lesser role): ONE node, data curation only                │
        │   - everything else local                                          │
        └───┬───────────────┬───────────────┬───────────────┬───────────────┘
            │               │               │               │
      ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
      │  Redis    │  │ NKT-Mirror  │  │   Weave     │  │ Frozen     │
      │ adapter   │  │ trainer +   │  │ eval +      │  │ open model │
      │ library + │  │ custom      │  │ observ.     │  │ (instruct) │
      │ routing   │  │ serving     │  │ (charts)    │  │            │
      └───────────┘  └─────────────┘  └─────────────┘  └────────────┘
```

### Roles / sponsor-tool usage (all load-bearing)
- **Redis (prize):** adapter library (thousands of 100 KB blobs) + router (embed user/query
  → fetch right adapter) + day's interaction queue / working memory.
- **Weave / W&B (prize):** the PROOF layer — logs generations, computes & plots held-out
  perplexity, the cross-user confusion matrix, the NKT-vs-LoRA size chart, improvement over
  "days." Also tracks training runs.
- **CopilotKit (prize):** frontend — chat UI + dashboard. Connects via **AG-UI** to the
  Python LangGraph agent. Model-agnostic (factory/custom-agent mode), so it runs our local
  open model fine.
- **OpenAI / GPT (lesser role):** ONLY the data-curation node (turn messy interactions into
  clean training pairs). Framed as "optional teacher, swappable for a local model."
- **LangGraph:** the local orchestrator graph (not a sponsor, but the control plane).
- **Cursor:** build tool.

## Key DECISIONS (do not re-litigate)
1. **Training is BATCH / overnight (24h), NOT live.** Removes the unverified "trains in
   seconds" risk. Demo is time-compressed with pre-baked adapters.
2. **Build on the INSTRUCT open model** (NKT-Mirror only ties LoRA on instruct).
3. **Personalize STYLE/PREFERENCES, not facts.** Pair with RAG for facts if needed (out of
   scope for the demo).
4. **Granularity:** per-user where data is rich; **per-category** as the safe, demoable unit
   (categories pool enough data and dodge cold-start). The eval works for either.
5. **GPT = curation only**, everything else local-first.
6. **Proof = objective:** held-out perplexity + cross-user/category **identification
   confusion matrix** (clean diagonal = personalization works) + beat a context-memory
   baseline at zero context cost.

## CRITICAL CONSTRAINTS & RISKS (from integration audit)
- **R1 (highest): Custom serving.** NKT-Mirror is activation gating, NOT LoRA — vLLM/LoRAX
  multi-LoRA serving will NOT work. We hand-roll serving: load frozen base once, swap the
  100 KB gate tensors per request. Build & verify this FIRST; nothing downstream matters
  without it.
- **R2: Reproduce the method.** The `nkt-mirror-test` results may be projected, not run.
  Reproduce ntkmirror_instruct ≈ 0.711 > base ≈ 0.536 on a GPU before trusting anything.
- **R3: Two-runtime stack.** CopilotKit runtime = Node/TS; ML = Python. LangGraph-Python SDK
  bridges via AG-UI, but it's two processes — budget wiring time.
- **R4: CopilotKit premium.** Core runtime self-hosts; some features (threads/inspector)
  need a free dev license/Helm — use only what's needed.
- **R5: Privacy framing.** GPT curation sends data out; frame as optional/local-swappable.
- **R6: Data quality dominates.** A 5K-param adapter is brutally sensitive to training-data
  quality — clean beats complex. Curation quality > pipeline complexity.
- **R7: Cold-start.** Thin users → noisy adapters → muddy matrix. Use high-volume
  users/categories for the demo.

## Data
- **Stack Exchange** communities/users (CC dump; HF mirrors: `habedi/stack-exchange-dataset`,
  `lvwerra/stack-exchange-paired`) — question → accepted answer; built-in categories +
  graded reference answers. Preferred for the eval (per-category or per-prolific-user).
- Backup: customer-support datasets (`Tobi-Bueck/customer-support-tickets` has agent answers
  + category labels; "Customer Support on Twitter" is split by brand).

## Base model
- Qwen2.5-7B-**Instruct** (matches the harness; instruct is required). Smaller Qwen2.5-1.5B-
  Instruct acceptable if GPU-constrained.

## What "done" looks like (demo)
1. CopilotKit UI: pick a user/category → chat → it sounds like them; base doesn't; show the
   real held-out reference next to both.
2. Weave: **confusion-matrix heatmap** with a clean diagonal + base-vs-adapter perplexity +
   "matches context-memory at zero context tokens."
3. Size slide: 100 KB × N users vs 154 MB LoRA.
4. Vision slide: orchestrator becomes a model fine-tuned to fine-tune models.

## Evaluation — how we PROVE personalization (the heart of the project)

Personalization is fuzzy, so we make it objective with three measures. None requires human
judgment.

1. **Held-out perplexity (objective).** Split each unit's (user/category) data into
   train / held-out. Train the adapter on train only. On held-out text the adapter never saw,
   measure perplexity. **Pass if** `perplexity(adapter) < perplexity(base)` on that unit's
   held-out data. Held-out (not train) ⇒ proves generalization, not memorization.

2. **Cross-unit identification confusion matrix (the clincher).** Train adapters for units
   A, B, C, D. For each unit's held-out text, score it under *every* adapter; the winner is
   the lowest-perplexity adapter. Build a matrix: rows = true unit, cols = adapter that
   scored it best. **A clean diagonal = each adapter learned its own unit** (improvement is
   unit-specific, not generic). This is harder to game than (1) and is the headline demo
   visual (heatmap).

3. **Beat the context-memory baseline (competitive proof).** Run the same held-out eval for
   base / context-memory (stuff the unit's examples into the prompt) / adapter. **Pass if**
   adapter ≥ context-memory **at zero extra context tokens.** Substantiates "weight-memory
   works and is free at inference."

Plus qualitative side-by-side generations (base vs adapter vs the real reference) so a human
*feels* it. Optional fact-capacity test: plant N preferences/facts per unit, chart held-out
recall as N grows — finds where 100 KB gates saturate ("how much fits in 100 KB").

## Demo run-of-show (3 minutes, time-compressed)

1. **Hook (20s):** "Every AI personalizes by stuffing text into context — it rots, drifts,
   costs tokens, lives on their servers. We bake *how you like things* into a 100 KB brain."
2. **Live UI (60s):** CopilotKit — pick a unit (e.g. Cooking), ask a question. Show base
   answer vs adapter answer vs the real reference. Adapter matches the unit's voice; base
   doesn't.
3. **Proof (60s):** Weave — the **confusion-matrix heatmap** (clean diagonal), the
   base-vs-adapter perplexity drop, and "adapter ≥ context-memory at 0 context tokens."
4. **Scale (20s):** size chart — 100 KB × N units vs 154 MB LoRA. "This is why per-unit is
   only feasible with us."
5. **Vision (20s):** "Today: overnight weight-memory personalization on open models.
   Tomorrow: the orchestrator becomes a model fine-tuned to fine-tune models."

## Anticipated judge objections & answers
- *"Isn't this just Doc-to-LoRA / context distillation?"* → Those are one-shot
  document→adapter via heavy hypernetworks. Ours is a tiny, accumulating, **per-unit** adapter
  small enough (100 KB) to serve at scale + retrained on a nightly cadence. Execution + size,
  not concept.
- *"Why not just RAG / a system prompt?"* → For facts, use RAG — we agree. For *style at
  scale*, context costs tokens every turn, drifts, and lives server-side. We win on
  zero-context-cost, no drift, private, on-device, millions-of-users-feasible.
- *"Does it actually personalize or just improve generally?"* → The confusion matrix: each
  adapter scores its OWN unit best ⇒ unit-specific, not generic.
- *"Can it remember facts?"* → It reweights known concepts (preferences/soft-facts), not
  arbitrary new associations — and we *measured* the ceiling (fact-capacity test). Honest by
  design; pair with RAG for hard facts.

## Build order (dependency-correct)
1. **Custom adapter serving + reproduce method (R1, R2)** — riskiest, do first.
2. **Eval (perplexity + confusion matrix)** — the proof; can stand alone.
3. **LangGraph nightly batch loop** — thin orchestration over 1+2.
4. **Redis library + routing.**
5. **CopilotKit UI** — most replaceable, do last.
