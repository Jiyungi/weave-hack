# Morning checklist — activate the new Tier 1–3 agent tools

The code is written, tested locally, and pushed to `origin/unified`
(commit `ee26d57`). I could **not** run any of this on the Brev box — I have no
access to it. So this is the short, copy-paste sequence **you** run on the box
to make the 20 new tools live and verify them. Everything here is honest about
what's been proven vs. what still needs your eyes.

## What shipped (and what's actually been verified)

- `agents/tools_extra.py` — 20 executors (read_file, list_dir, write_file,
  shell, apply_patch, note, pdf_read, doc_index, doc_search, sql_query,
  csv_query, wikipedia, unit_convert, currency, timezone, translate,
  stock_price, crypto_price, geocode, news).
- `agents/tools.py` — registers the extras at import (guarded; a bug in extras
  can't break the built-in tools).
- `scripts/mint_tools.py` — mints one NTK controller per tool via the control
  plane `POST /register`.
- `agents/test_tools_extra.py` — **ran locally: 20 passed, 2 skipped** (pint
  not installed on my machine + network tools skipped offline).

**Verified locally:** registry wiring, arg parsing, the workspace jail, and
every offline executor (file/shell/patch/note/sql/csv/doc/timezone).
**NOT yet verified (your morning checklist):** minting on the box, live network
tools (currency/stock/news/etc.), and the governed 7B actually emitting the new
tool calls end-to-end in chat.

## Steps on the box

```bash
# 1. pull the new code
cd ~/weave-hack
git pull origin unified

# 2. install the new optional deps into the venv
source ~/venv/bin/activate
VIRTUAL_ENV=~/venv uv pip install -r requirements.txt
#   (or: pip install pint pandas "pdfminer.six>=20221105")

# 3. make sure the workspace sandbox exists (defaults to ./workspace)
mkdir -p workspace

# 4. restart services so tools_extra is loaded everywhere
bash start_all.sh restart
bash start_all.sh status        # track-a/b/d + ui should be up

# 5. sanity-check the registry loaded all 20 tools (no minting yet)
python -c "from agents import tools; print(sorted(tools.registry()))"
#   expect to see read_file, shell, currency, wikipedia, ... (20 new names)

# 6. mint controllers for the new tools (needs track-a:8000 + track-b:8100 up)
#    ~36s each => ~12 min for all 20. Do a couple first if you want a fast check:
python -m scripts.mint_tools --names currency unit_convert timezone
#    then the rest:
python -m scripts.mint_tools
```

## Live verification (in the UI at http://localhost:3000)

Open the Agents/chat panel and try prompts that should trigger the new skills.
The control plane already grants every minted skill to `exec-assistant`, so the
orchestrator can use them immediately — no roster edit needed.

- "What's 250 USD in EUR?"            → `currency`
- "Convert 10 km to miles."           → `unit_convert`
- "What time is it now in Tokyo?"     → `timezone`
- "Look up Alan Turing on Wikipedia." → `wikipedia`
- "Price of bitcoin?"                 → `crypto_price`
- "Latest news about AI."             → `news`
- "Save a note: I prefer metric." then "Recall my notes." → `note`
- "Write a file todo.txt with '- ship tools', then read it back."
  → `write_file` + `read_file` (these are SENSITIVE → expect an approval gate)

## If something's off

- **A tool didn't mint:** re-run `python -m scripts.mint_tools --names <tool>`.
  Minting needs track-a (:8000) and track-b (:8100) `up` in `start_all.sh status`.
- **Registry missing tools:** `python -c "import agents.tools_extra"` — if that
  prints an error, that's the cause (a missing import). The core stack still
  runs; only the extras are affected.
- **A network tool errors:** it returns a clear `[tool error] ...` string rather
  than crashing the loop. Free public endpoints (exchangerate.host, stooq,
  coingecko, nominatim, google news RSS) occasionally rate-limit — retry.
- **shell/write_file/apply_patch:** these are `sensitive=True` by design, so the
  governance layer asks for human approval before the capability runs. That's
  intended, not a bug.

## Notes on design choices (so you can defend them)

- **Sandbox:** file/shell/patch tools are jailed under `WORKSPACE_DIR` via
  `_jailed()`, which rejects any path that escapes the workspace.
- **No keys required:** every Tier 2/3 tool uses a free public endpoint or the
  stdlib; nothing needs an API key to demo.
- **Graceful degradation:** `csv_query` falls back to stdlib `csv` if pandas is
  absent; `doc_index`/`note` fall back to process-local memory if Redis is down.
- **Consistent with the thesis:** `note` and `doc_index` write to Redis as
  working state — they are *not* the personalization mechanism. Personalization
  still happens in the weights (NTK-Mirror controllers), not in stored context.
