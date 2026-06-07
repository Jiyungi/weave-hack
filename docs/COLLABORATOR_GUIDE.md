# OpenMirror — collaborator runbook (Brev → laptop)

End-to-end guide: spin up on a Brev GPU box, use the dashboard from your laptop, shut down overnight without losing minted controllers, and come back the next day.

**Branch:** `unified` (demo + chat UI + governance fixes)

**Repo:** https://github.com/Jiyungi/weave-hack

---

## What runs where

| Service | Port | Process | Needs GPU |
|---------|------|---------|-----------|
| **Dashboard (Next.js + CopilotKit)** | 3000 | `ui/` | no |
| **Orchestrator (Track D)** | 8200 | `agent_service.py` | no (calls brain API) |
| **Control plane (Track B)** | 8100 | `control_plane_service.py` | no |
| **NTK engine (Track A)** | 8000 | `controller_service.py` | yes (7B) |
| **Brain (vLLM, optional)** | 8001 | `vllm serve` | yes (14B) |
| **Redis** | 6379 | local or **Redis Cloud** | no |

On the box, `bash start_all.sh` starts all of this in a **tmux** session named `openmirror`.

---

## Prerequisites (each collaborator)

1. **Brev account** with access to the team GPU instance type (A100-class).
2. **Brev CLI** on your laptop: https://brev.dev/docs/reference/brev-cli
3. **GitHub** read access to `weave-hack`.
4. **Team `.env`** (ask the lead — never commit it). Minimum:
   - `REDIS_URL` — use the **shared Redis Cloud** URL so policies/skills survive instance delete
   - Optional: `BRIGHTDATA_API_KEY`, `OPENAI_API_KEY`, `WANDB_API_KEY`
5. **Laptop** with a browser; you only need port **3000** forwarded for the UI.

---

## Part 1 — First-time setup on a fresh Brev box

SSH or open the **Jupyter / terminal** on the instance.

### 1. Clone and bootstrap (one time per instance)

```bash
export REPO_URL=https://github.com/Jiyungi/weave-hack.git
export BRANCH=unified
export VENV=$HOME/venv

# Raw GitHub URLs must NOT include .git (use repo path only):
curl -fsSL "https://github.com/Jiyungi/weave-hack/raw/${BRANCH}/setup_brev.sh" -o /tmp/setup_brev.sh
bash /tmp/setup_brev.sh
```

What this does (~15–30 min first run):

- Python venv at `~/venv`
- PyTorch (CUDA build matched to the box)
- `ntkmirror` clone at `~/ntkmirror_src`
- Repo at `~/weave-hack` on branch `unified`
- Pre-download **Qwen2.5-7B** weights (Track A)
- Node 20 + `npm install` in `ui/`
- Local **redis-server** if not using cloud-only Redis

### 2. Install tmux (if missing)

```bash
sudo apt-get install -y tmux
```

`setup_brev.sh` installs **vLLM 0.11** (CUDA 12.8) and pins **transformers below 5.0**. Skip with `INSTALL_VLLM=0 bash setup_brev.sh`.

If brain crashes after a manual `pip install`:

```bash
source ~/venv/bin/activate
pip install "transformers>=4.55.2,<5.0.0"
VIRTUAL_ENV=~/venv uv pip install "vllm==0.11.0" --torch-backend=cu128
```

Without vLLM, governance demos (Track A/B) still work; orchestrator chat reasoning is degraded.

### 3. Configure environment

```bash
cd ~/weave-hack
cp .env.example .env
# Paste team secrets into .env (REDIS_URL, keys, etc.)

# Important: controller .pt files live here (Track A reads CONTROLLER_DIR):
grep CONTROLLER .env || echo 'CONTROLLER_DIR=./controllers' >> .env
```

Copy the team `.env` from 1Password / Slack / lead — **do not commit `.env`**.

Ensure Redis Cloud URL uses `redis://` not `rediss://` if you hit TLS version errors (see comment in `.env.example`).

### 4. Restore saved controllers (if you have a backup)

Skip on a truly fresh demo; do this when resuming after deleting last night’s instance:

```bash
cd ~/weave-hack
bash scripts/restore_controllers.sh ~/openmirror-backups/controllers-YYYYMMDD.tar.gz
# or: bash scripts/restore_controllers.sh /path/to/controllers/
```

### 5. Start everything

```bash
cd ~/weave-hack
bash start_all.sh
```

Check:

```bash
bash start_all.sh status
```

Attach logs: `bash start_all.sh attach` (detach: **Ctrl-b** then **d**)

---

## Part 2 — Use from your laptop

Replace `<instance>` with your Brev name (e.g. `narwhal`).

```bash
brev port-forward <instance> --port 3000:3000
```

Open **http://localhost:3000**

### Header controls

- **`auto-approve`** — when **off**, every capability REQUEST waits for you (good for testing revokes).
- Health chips — `track_a`, `state`, `skills`, `sessions`, `agents up`.

### Typical demo flow

1. **Capabilities** — **Seed demo (weather + calendar)** sets role policies; register extra tools (~36 s each).
2. **Chat** — set **User ID** (e.g. `alice`); orchestrator routes sub-tasks to **research-agent** (lookup), **ops-agent** (code), or **support-agent** (weather).
3. **Multi-agent demo prompt:** *"Weather in Berlin and compute 15% tip on $84 with python"* — expect **≥2 workers** in the delegation tree; expand details for BLOCKED vs ALLOWED.
4. **Verify:** `python verify_orchestrator.py` on the box (offline stub; `--live` for full stack).
5. **Capability requests** — approve/deny when agents REQUEST new skills.
6. **Revoke** — session revoke (this chat only) vs policy revoke (all future sessions).
7. **Memory** — log turns → **Consolidate → mint style** for `user_style-{user}`.

**Legacy principals:** `exec-assistant` / `support-bot` remain for `verify_control_plane.py` only. Migrate Redis policies with `python scripts/migrate_worker_policies.py --dry-run`.

---

## Part 3 — End of day (stop billing)

Brev GPU instances **do not pause** — you **stop processes** and **delete the instance** when done.

### Step A — Stop services (on the box)

```bash
cd ~/weave-hack
bash start_all.sh stop
```

### Step B — Save controllers to your laptop (critical)

Minted NTK controllers are **`*.pt` files** (~100 KB each) under:

```text
~/weave-hack/controllers/     # default CONTROLLER_DIR
```

**Redis Cloud** keeps skill *names* and policy mappings, but Track A must have the matching `.pt` files on disk after a fresh box.

#### Option 1 — From the box, push tarball to laptop

On the **Brev box**:

```bash
cd ~/weave-hack
bash scripts/backup_controllers.sh
# creates ~/openmirror-backups/controllers-YYYYMMDD-HHMMSS.tar.gz
```

On your **laptop** (use Brev SSH host or `brev shell` + scp):

```bash
mkdir -p ~/openmirror-backups
scp <brev-host>:~/openmirror-backups/controllers-*.tar.gz ~/openmirror-backups/
```

#### Option 2 — One-liner from laptop via SSH

```bash
INSTANCE=narwhal   # your instance name
mkdir -p ~/openmirror-backups
ssh $(brev open "$INSTANCE" --print-ssh) \
  'cd ~/weave-hack && tar czf - controllers/' \
  > ~/openmirror-backups/controllers-$(date +%Y%m%d).tar.gz
```

Adjust if your controllers live under `data/adapters/` — check `CONTROLLER_DIR` in `.env`.

### Step C — Optional extras to save

| Artifact | Path | Why |
|----------|------|-----|
| Team env | `~/weave-hack/.env` | secrets + URLs (store securely locally, not in git) |
| Audit log | `~/weave-hack/control_plane_audit.jsonl` | local audit copy if not only on Redis |
| HF cache | `~/.cache/huggingface/` or `HF_HOME` | **large (~30 GB+)** — usually re-download instead |

### Step D — Delete the Brev instance

In Brev console or CLI — **delete** the instance so GPU billing stops.

**Safe to delete after:** controllers tarball on laptop + team `.env` saved + using **shared Redis Cloud** (governance state persists there).

---

## Part 4 — Next morning (new instance)

1. Create a **new** Brev GPU instance (same type as before).
2. Run **Part 1** again (`setup_brev.sh` — repo + venv + weights; faster if HF cache warm on same image).
3. Copy team **`.env`** back to `~/weave-hack/.env`.
4. **Restore controllers:**

   ```bash
   cd ~/weave-hack
   bash scripts/restore_controllers.sh ~/openmirror-backups/controllers-YYYYMMDD.tar.gz
   ```

5. `bash start_all.sh`
6. Laptop: `brev port-forward <new-instance> --port 3000:3000`

You should see existing skills in the UI (**Tool catalog → registered**) because Redis still maps skill names → controller IDs, and the `.pt` files are back on disk.

If Redis was **local only** (not cloud), re-seed: **Seed demo** + re-register tools, or restore a Redis RDB dump (ask lead).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| UI shows old copy, no auto-approve toggle | On box: `git pull origin unified && bash start_all.sh restart`; hard-refresh browser |
| `control plane unreachable` | `bash start_all.sh status`; check track-b window in tmux |
| `unknown controller` / act fails after restore | Controllers not restored or wrong `CONTROLLER_DIR`; run `curl localhost:8000/controllers` |
| Revoke ignored, tools come back via REQUEST | Pull latest `unified`; turn **auto-approve off**; session-revoked skills need manual Approve to override |
| Brain / chat stuck | vLLM window in tmux; install vLLM (Part 1 step 2) |
| Redis errors | `redis-cli -u "$REDIS_URL" ping` |
| CUDA / driver mismatch | See `TORCH_CUDA_INDEX` in `setup_brev.sh`; match vLLM wheel to box |

---

## Quick reference

```bash
# Box
cd ~/weave-hack
bash start_all.sh              # start
bash start_all.sh status       # health
bash start_all.sh attach       # logs
bash start_all.sh restart      # after git pull
bash start_all.sh stop         # end of day

bash scripts/backup_controllers.sh
bash scripts/restore_controllers.sh <tarball-or-dir>

# Laptop
brev port-forward <instance> --port 3000:3000
open http://localhost:3000
```

---

## Architecture pointer

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- [`PERSONALIZATION.md`](../PERSONALIZATION.md)
- [`README.md`](../README.md)
