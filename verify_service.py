"""
verify_service.py
=================

End-to-end check that the Track A controller service reproduces the smoke
through HTTP. It does NOT touch ntkmirror/torch directly — it only speaks to the
running FastAPI app, exactly as Tracks B/C will. So a PASS here means the whole
service path (train -> compose -> subtract -> evaluate) works, not just the math.

Mirrors smoke_compose_subtract.py over the wire:
  1. POST /train   weather   -> controller A
  2. POST /train   calendar  -> controller B
  3. POST /compose [A,B] [1, 1]      -> A+B      (grant both)
  4. POST /compose [A+B,B] [1,-1]    -> (A+B)-B  (revoke B)
  5. POST /evaluate each state on held-out prompts (needle match)
  6. print the same table + PASS/FAIL verdict

Run (service must be up: uvicorn controller_service:app --port 8000):
  python verify_service.py
  BASE_URL=http://localhost:8000 python verify_service.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
import urllib.error
import json

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
MAX_NEW = int(os.environ.get("VERIFY_MAX_NEW_TOKENS", "16"))

CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Accra", "Quito", "Hanoi"]
DATES = ["2026-06-06", "2026-07-01", "2026-08-15", "2026-09-30", "2026-12-25"]

SKILL_A = {
    "task_id": "weather",
    "needle": "weather(",
    "train": [{"prompt": f"User: what's the weather in {c}?\nAssistant:",
               "completion": f' weather("{c}")'} for c in CITIES],
    "eval_prompts": [f"User: what's the weather in {c}?\nAssistant:"
                     for c in ["Berlin", "Madrid", "Nairobi", "Seoul", "Bogota"]],
}
SKILL_B = {
    "task_id": "calendar",
    "needle": "calendar(",
    "train": [{"prompt": f"User: any events on {d}?\nAssistant:",
               "completion": f' calendar("{d}")'} for d in DATES],
    "eval_prompts": [f"User: any events on {d}?\nAssistant:"
                     for d in ["2026-01-01", "2026-02-14", "2026-03-17",
                               "2026-04-22", "2026-11-11"]],
}


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"POST {path} -> {e.code}: {e.read().decode()}")


def get(path: str) -> dict:
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        sys.exit(f"GET {path} failed ({e}). Is the service up at {BASE}?")


def emit_fraction(controller_id: str | None, prompts: list[str], needle: str) -> float:
    hits = 0
    for p in prompts:
        out = post("/execute", {"controller_id": controller_id,
                                "prompt": p, "max_new_tokens": MAX_NEW})
        hits += int(needle in out["completion"])
    return hits / len(prompts)


def row(label: str, controller_id: str | None) -> tuple[float, float]:
    a = emit_fraction(controller_id, SKILL_A["eval_prompts"], SKILL_A["needle"])
    b = emit_fraction(controller_id, SKILL_B["eval_prompts"], SKILL_B["needle"])
    print(f"  {label:12s}  weather={a:.2f}  calendar={b:.2f}")
    return a, b


def main() -> None:
    health = get("/health")
    print("=== Track A service verify ===")
    print(f"  {BASE}  model={health['model']}  device={health['device']}  "
          f"gates={health['gates']}  max_log_gate={health['max_log_gate']}")

    print("\n[1/3] training the two skill controllers (real 600-step fits)")
    a = post("/train", {"task_id": SKILL_A["task_id"], "examples": SKILL_A["train"]})
    b = post("/train", {"task_id": SKILL_B["task_id"], "examples": SKILL_B["train"]})
    print(f"  weather  -> {a['controller_id']}  loss {a['loss_first']:.3f} -> "
          f"{a['loss_last']:.3f}  ({a['train_seconds']}s, {a['artifact_bytes']} B)")
    print(f"  calendar -> {b['controller_id']}  loss {b['loss_first']:.3f} -> "
          f"{b['loss_last']:.3f}  ({b['train_seconds']}s, {b['artifact_bytes']} B)")

    print("\n[2/3] compose (grant) and subtract (revoke) over HTTP")
    add = post("/compose", {"controller_ids": [a["controller_id"], b["controller_id"]],
                            "weights": [1.0, 1.0], "new_id": "verify-A+B"})
    sub = post("/compose", {"controller_ids": [add["controller_id"], b["controller_id"]],
                            "weights": [1.0, -1.0], "new_id": "verify-(A+B)-B"})
    pair = post("/pair", {"a": a["controller_id"], "b": b["controller_id"]})

    print("\n[3/3] results (fraction of held-out prompts emitting each skill)")
    base = row("base", None)
    _ = row("A only", a["controller_id"])
    _ = row("B only", b["controller_id"])
    add_r = row("A+B", add["controller_id"])
    sub_r = row("(A+B)-B", sub["controller_id"])

    print("\n=== gate geometry ===")
    print(f"  overlap(A,B) jaccard = {pair['jaccard']:.3f}  cosine = {pair['gate_cosine']:.3f}")

    print("\n=== verdict ===")
    base_clean = base[0] <= 0.2 and base[1] <= 0.2
    composition_ok = add_r[0] >= 0.8 and add_r[1] >= 0.8
    revocation_ok = sub_r[0] >= 0.8 and sub_r[1] <= 0.2

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  base baseline ~0 (attributable)     : {mark(base_clean)}")
    print(f"  composition  (A+B does both)        : {mark(composition_ok)}")
    print(f"  revocation   ((A+B)-B keeps A,no B)  : {mark(revocation_ok)}")
    if base_clean and composition_ok and revocation_ok:
        print("\n  SERVICE PATH HOLDS -> Tracks B/C can build on these endpoints.")
    else:
        print("\n  Service path differs from the smoke; inspect /train losses and "
              "/pair overlap before wiring the control plane.")


if __name__ == "__main__":
    main()
