"""Extra agent tools (Tier 1–3) for OpenMirror.

Kept in a separate module so the core ``tools.py`` stays small; ``tools.py``
calls :func:`register_all` at import time to add these to the live registry.
Every executor follows the house contract: takes ONE string arg, returns a
string, raises :class:`tools.ToolError` on failure, and lazy-imports any heavy
dependency so importing this module is cheap and never crashes.

Arg conventions (the governed model emits ``name("arg")``; arbitrary-content
tools use ``arg_mode="block"`` so the content comes from the brain's fenced
block instead of a quoted arg):

* scalar tools  -> ``unit_convert("10 km to miles")``
* file paths    -> jailed under ``WORKSPACE_DIR`` (no escaping the sandbox)
* block tools   -> ``write_file`` / ``apply_patch`` take "first line = path,
  remaining lines = content"; ``shell`` / ``sql_query`` take the whole block.
"""

from __future__ import annotations

import csv as _csv
import io
import json
import math
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .tools import (
    Tool,
    ToolError,
    _format_hits,
    _http_get,
    _strip_html,
    _wiki_search_hits,
    register,
)

# ---------------------------------------------------------------------------
# Workspace sandbox (file / shell / patch tools are jailed here)
# ---------------------------------------------------------------------------


def _workspace() -> Path:
    root = os.environ.get("WORKSPACE_DIR", "").strip()
    base = Path(root) if root else Path(__file__).resolve().parents[1] / "workspace"
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _jailed(relpath: str) -> Path:
    """Resolve ``relpath`` strictly inside the workspace; reject escapes."""
    base = _workspace()
    p = (base / relpath.strip().lstrip("/\\")).resolve()
    if base not in p.parents and p != base:
        raise ToolError(f"path escapes workspace: {relpath!r}")
    return p


# ---------------------------------------------------------------------------
# Tier 1 — core agent muscle
# ---------------------------------------------------------------------------


def _read_file(arg: str) -> str:
    p = _jailed(arg)
    if not p.exists() or not p.is_file():
        raise ToolError(f"no such file in workspace: {arg!r}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:8000] + ("\n...[truncated]" if len(text) > 8000 else "")


def _list_dir(arg: str) -> str:
    p = _jailed(arg or ".")
    if not p.exists():
        return f"(empty) {arg or '.'}"
    if p.is_file():
        return f"{p.name} ({p.stat().st_size} bytes)"
    rows = []
    for child in sorted(p.iterdir()):
        kind = "dir " if child.is_dir() else "file"
        size = child.stat().st_size if child.is_file() else 0
        rows.append(f"[{kind}] {child.name}" + (f" ({size}B)" if child.is_file() else ""))
    return "\n".join(rows) or "(empty)"


def _write_file(arg: str) -> str:
    # block mode: first line = path, remainder = content
    first, _, content = arg.partition("\n")
    path = first.strip()
    if not path:
        raise ToolError("write_file needs 'first line = path, rest = content'")
    p = _jailed(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


# Conservative shell allow/deny. shell is SENSITIVE (human-approved) AND jailed.
_SHELL_DENY = re.compile(
    r"(\brm\s+-rf\b|\bmkfs\b|\b:\(\)\s*\{|\bdd\s+if=|/dev/sd|\bshutdown\b|\breboot\b"
    r"|\bchmod\s+-R\b|\bchown\s+-R\b|>\s*/dev/|\bcurl\b.*\|\s*(sh|bash)|\bwget\b.*\|\s*(sh|bash))"
)


def _shell(arg: str) -> str:
    cmd = arg.strip()
    if not cmd:
        raise ToolError("shell requires a command")
    if _SHELL_DENY.search(cmd):
        raise ToolError("shell command blocked by safety policy")
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=20, cwd=str(_workspace()),
        )
    except subprocess.TimeoutExpired:
        raise ToolError("shell command timed out (20s)")
    out = (proc.stdout or "")
    if proc.stderr:
        out += ("\n[stderr]\n" + proc.stderr)
    out = out.strip() or f"(exit {proc.returncode}, no output)"
    return out[:4000] + ("\n...[truncated]" if len(out) > 4000 else "")


def _apply_patch(arg: str) -> str:
    """Apply a minimal unified diff to a single file under the workspace.

    Supports one file per call (the common agent case). Recognizes the
    ``--- a/<path>`` / ``+++ b/<path>`` header and ``@@`` hunks; applies
    additions/deletions by context. Falls back to a clear error rather than a
    partial write if context doesn't match.
    """
    lines = arg.splitlines()
    target = None
    for ln in lines:
        if ln.startswith("+++ "):
            target = ln[4:].strip()
            break
    if not target:
        raise ToolError("apply_patch: no '+++ <path>' header found")
    target = re.sub(r"^[ab]/", "", target)
    p = _jailed(target)
    original = p.read_text(encoding="utf-8").splitlines(keepends=False) if p.exists() else []

    new: list[str] = []
    src_idx = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("@@"):
            m = re.search(r"-(\d+)", ln)
            start = (int(m.group(1)) - 1) if m else src_idx
            new.extend(original[src_idx:start])
            src_idx = start
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                h = lines[i]
                if h.startswith(" "):
                    new.append(original[src_idx] if src_idx < len(original) else h[1:])
                    src_idx += 1
                elif h.startswith("-"):
                    src_idx += 1
                elif h.startswith("+"):
                    new.append(h[1:])
                i += 1
        else:
            i += 1
    new.extend(original[src_idx:])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(new) + "\n", encoding="utf-8")
    return f"patched {target} ({len(new)} lines)"


# Scratch memory: Redis-backed when REDIS_URL is set, else process-local.
_NOTES_MEM: dict[str, list[str]] = {}


def _notes_key() -> str:
    return f"notes:{os.environ.get('OPENMIRROR_NOTE_SCOPE', 'default')}"


def _note(arg: str) -> str:
    text = arg.strip()
    recall = (not text) or text.lower() in ("recall", "list", "show")
    key = _notes_key()
    try:
        import redis  # type: ignore

        r = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=True, socket_connect_timeout=2,
        )
        if recall:
            items = r.lrange(key, 0, -1)
            return "notes:\n" + "\n".join(f"- {n}" for n in items) if items else "(no notes)"
        body = re.sub(r"^(save|note|add)\s*:\s*", "", text, flags=re.I)
        r.rpush(key, body)
        return f"noted: {body}"
    except Exception:
        # process-local fallback
        items = _NOTES_MEM.setdefault(key, [])
        if recall:
            return "notes:\n" + "\n".join(f"- {n}" for n in items) if items else "(no notes)"
        body = re.sub(r"^(save|note|add)\s*:\s*", "", text, flags=re.I)
        items.append(body)
        return f"noted: {body}"


# ---------------------------------------------------------------------------
# Tier 2 — knowledge & data
# ---------------------------------------------------------------------------


def _pdf_read(arg: str) -> str:
    src = arg.strip()
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        raise ToolError("pdf_read needs pdfminer.six (pip install pdfminer.six)")
    data: bytes
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "OpenMirror-Agent/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
    else:
        p = _jailed(src)
        if not p.exists():
            raise ToolError(f"no such PDF in workspace: {src!r}")
        data = p.read_bytes()
    text = extract_text(io.BytesIO(data)) or ""
    text = text.strip()
    if not text:
        return "(no extractable text in PDF)"
    return text[:8000] + ("\n...[truncated]" if len(text) > 8000 else "")


# Lightweight, model-free embedding (hashed bag-of-words) for RAG over Redis.
_EMB_DIM = 256


def _embed(text: str) -> list[float]:
    vec = [0.0] * _EMB_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = 2166136261
        for ch in tok:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        vec[h % _EMB_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


_DOCS_MEM: dict[str, dict] = {}


def _doc_index(arg: str) -> str:
    # block mode: first line = doc id, rest = text
    first, _, body = arg.partition("\n")
    doc_id = first.strip() or f"doc{len(_DOCS_MEM) + 1}"
    text = body.strip() or first.strip()
    if not text:
        raise ToolError("doc_index needs 'first line = id, rest = text'")
    rec = {"id": doc_id, "text": text[:4000], "embedding": _embed(text)}
    try:
        import redis  # type: ignore

        r = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=True, socket_connect_timeout=2,
        )
        r.hset("docs:index", doc_id, json.dumps(rec))
        return f"indexed doc {doc_id!r} ({len(text)} chars)"
    except Exception:
        _DOCS_MEM[doc_id] = rec
        return f"indexed doc {doc_id!r} ({len(text)} chars, local)"


def _doc_search(arg: str) -> str:
    q = arg.strip()
    if not q:
        raise ToolError("doc_search requires a query")
    qv = _embed(q)
    records: list[dict] = []
    try:
        import redis  # type: ignore

        r = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=True, socket_connect_timeout=2,
        )
        for raw in r.hgetall("docs:index").values():
            records.append(json.loads(raw))
    except Exception:
        records = list(_DOCS_MEM.values())
    if not records:
        return "(no indexed documents)"

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    ranked = sorted(records, key=lambda d: cos(qv, d["embedding"]), reverse=True)[:3]
    return "\n".join(
        f"{i}. [{d['id']}] {d['text'][:200]}" for i, d in enumerate(ranked, 1)
    )


def _sql_query(arg: str) -> str:
    """Read-only SQLite. Whole arg is the SELECT; DB is workspace/data.db.

    Rejects anything that isn't a single read-only SELECT/WITH statement.
    """
    import sqlite3

    sql = arg.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise ToolError("sql_query allows only read-only SELECT/WITH statements")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|create|attach|pragma)\b", sql):
        raise ToolError("sql_query: write/DDL statements are not allowed")
    db = _jailed("data.db")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        con = sqlite3.connect(db)  # create empty read store if absent
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(50)
        cols = [c[0] for c in cur.description] if cur.description else []
    finally:
        con.close()
    if not rows:
        return "(no rows)"
    out = [" | ".join(cols)] if cols else []
    out += [" | ".join(str(c) for c in row) for row in rows]
    return "\n".join(out)


def _csv_query(arg: str) -> str:
    """CSV ops. Arg: '<path> <op>' where op in head|columns|shape|describe|sum:<col>."""
    parts = arg.strip().split(None, 1)
    if not parts:
        raise ToolError("csv_query needs '<path> <op>'")
    path = parts[0]
    op = parts[1].strip() if len(parts) > 1 else "head"
    p = _jailed(path)
    if not p.exists():
        raise ToolError(f"no such CSV in workspace: {path!r}")
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(p)
        if op == "columns":
            return ", ".join(map(str, df.columns))
        if op == "shape":
            return f"{df.shape[0]} rows x {df.shape[1]} cols"
        if op == "describe":
            return df.describe(include="all").to_string()[:4000]
        if op.startswith("sum:"):
            col = op.split(":", 1)[1]
            return f"sum({col}) = {df[col].sum()}"
        return df.head(10).to_string()[:4000]
    except Exception:
        # stdlib fallback (head/columns/shape only)
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            reader = list(_csv.reader(fh))
        if not reader:
            return "(empty csv)"
        header, data = reader[0], reader[1:]
        if op == "columns":
            return ", ".join(header)
        if op == "shape":
            return f"{len(data)} rows x {len(header)} cols"
        preview = [" | ".join(header)] + [" | ".join(r) for r in data[:10]]
        return "\n".join(preview)


def _wikipedia(arg: str) -> str:
    q = arg.strip()
    if not q:
        raise ToolError("wikipedia requires a topic")
    try:
        title = urllib.parse.quote(q.replace(" ", "_"))
        data = json.loads(
            _http_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
        )
        if data.get("extract"):
            return f"{data.get('title', q)}: {data['extract']}"
    except Exception:
        pass
    hits = _wiki_search_hits(q)
    return _format_hits(hits) if hits else f"no Wikipedia results for {q!r}"


# ---------------------------------------------------------------------------
# Tier 3 — everyday utilities
# ---------------------------------------------------------------------------


def _unit_convert(arg: str) -> str:
    m = re.match(r"\s*([-\d.]+)\s*(.+?)\s+(?:to|in)\s+(.+)\s*$", arg, re.I)
    if not m:
        raise ToolError("unit_convert format: '<value> <from> to <to>' e.g. '10 km to miles'")
    value, src, dst = m.group(1), m.group(2).strip(), m.group(3).strip()
    try:
        import pint  # type: ignore

        ureg = pint.UnitRegistry()
        q = ureg.Quantity(float(value), src)
        res = q.to(dst)
        return f"{value} {src} = {res.magnitude:.6g} {dst}"
    except Exception as e:
        raise ToolError(f"unit_convert failed ({e}); needs pint and valid units")


def _currency(arg: str) -> str:
    m = re.match(r"\s*([-\d.]+)\s*([A-Za-z]{3})\s+(?:to|in)\s+([A-Za-z]{3})\s*$", arg, re.I)
    if not m:
        raise ToolError("currency format: '<amount> <FROM> to <TO>' e.g. '250 USD to EUR'")
    amount, src, dst = float(m.group(1)), m.group(2).upper(), m.group(3).upper()
    url = f"https://api.exchangerate.host/convert?from={src}&to={dst}&amount={amount}"
    try:
        data = json.loads(_http_get(url, timeout=15))
        result = data.get("result")
        if result is None:
            raise ValueError("no result")
        return f"{amount:g} {src} = {result:.2f} {dst}"
    except Exception as e:
        raise ToolError(f"currency lookup failed: {e}")


def _timezone(arg: str) -> str:
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        raise ToolError("timezone needs Python 3.9+ zoneinfo")

    aliases = {
        "utc": "UTC", "gmt": "UTC", "pst": "America/Los_Angeles",
        "pdt": "America/Los_Angeles", "est": "America/New_York",
        "edt": "America/New_York", "ct": "America/Chicago",
        "cst": "America/Chicago", "ist": "Asia/Kolkata", "jst": "Asia/Tokyo",
        "tokyo": "Asia/Tokyo", "london": "Europe/London", "paris": "Europe/Paris",
        "lagos": "Africa/Lagos", "berlin": "Europe/Berlin", "lima": "America/Lima",
    }

    def tz(name: str) -> ZoneInfo:
        key = name.strip().lower()
        return ZoneInfo(aliases.get(key, name.strip()))

    m = re.match(r"\s*now\s+in\s+(.+)$", arg, re.I)
    if m:
        return datetime.now(tz(m.group(1))).isoformat(timespec="seconds")
    m = re.match(r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(.+?)\s+(?:to|in)\s+(.+)$", arg, re.I)
    if m:
        hh = int(m.group(1)) % 12
        if (m.group(3) or "").lower() == "pm":
            hh += 12
        mm = int(m.group(2) or 0)
        src, dst = tz(m.group(4)), tz(m.group(5))
        now = datetime.now(src).replace(hour=hh, minute=mm, second=0, microsecond=0)
        return f"{now.isoformat(timespec='minutes')} -> {now.astimezone(dst).isoformat(timespec='minutes')}"
    raise ToolError("timezone format: 'now in Tokyo' or '3pm UTC to PST'")


def _translate(arg: str) -> str:
    m = re.match(r"\s*(.+?)\s+to\s+([A-Za-z\- ]+)\s*$", arg.strip())
    if not m:
        raise ToolError("translate format: '<text> to <language>' e.g. 'good morning to French'")
    text, lang = m.group(1).strip(), m.group(2).strip().lower()
    codes = {
        "french": "fr", "spanish": "es", "german": "de", "japanese": "ja",
        "italian": "it", "portuguese": "pt", "chinese": "zh", "korean": "ko",
        "arabic": "ar", "russian": "ru", "hindi": "hi", "dutch": "nl",
    }
    target = codes.get(lang, lang[:2])
    url = ("https://api.mymemory.translated.net/get?"
           + urllib.parse.urlencode({"q": text, "langpair": f"en|{target}"}))
    try:
        data = json.loads(_http_get(url, timeout=15))
        out = data.get("responseData", {}).get("translatedText")
        if not out:
            raise ValueError("no translation")
        return f"{text} -> ({target}) {out}"
    except Exception as e:
        raise ToolError(f"translate failed: {e}")


def _stooq_blocked(body: str) -> bool:
    text = body.lstrip()
    return (
        "requires JavaScript" in body
        or text.startswith("<!DOCTYPE")
        or text.startswith("<html")
    )


def _stock_price(arg: str) -> str:
    prior_re = re.compile(
        r"\b(yesterday|previous|prior|last\s+close|last\s+day|prior\s+day)\b",
        re.I,
    )
    raw = arg.strip()
    want_prev = bool(prior_re.search(raw))
    sym = prior_re.sub("", raw).strip()
    sym = (sym.split()[0] if sym.split() else raw.split()[0]).lower().lstrip("$")
    if not re.match(r"^[a-z.\-]{1,10}$", sym):
        raise ToolError("stock_price needs a ticker, e.g. NVDA or 'NVDA yesterday'")
    ticker = sym if "." in sym else f"{sym}.us"
    display = sym.split(".")[0].upper()

    if want_prev:
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        try:
            raw = _http_get(url, timeout=15)
            if _stooq_blocked(raw):
                raise ToolError("quote source unavailable (provider blocked automated access)")
            lines = raw.strip().splitlines()
            if len(lines) < 3:
                raise ValueError("no history")
            header = lines[0].split(",")
            rows = [line.split(",") for line in lines[1:] if line.strip()]
            if len(rows) < 2:
                raise ValueError("not enough rows")
            fields = dict(zip(header, rows[-2]))
            close = fields.get("Close")
            if not close or close in ("N/D", ""):
                return f"no price found for {display}"
            return f"{display}: {close} (date {fields.get('Date', '?')}, previous trading day)"
        except Exception as e:
            raise ToolError(f"stock_price failed: {e}") from e

    url = f"https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=csv"
    try:
        raw = _http_get(url, timeout=15)
        if _stooq_blocked(raw):
            raise ToolError("quote source unavailable (provider blocked automated access)")
        body = raw.strip().splitlines()
        if len(body) < 2:
            raise ValueError("no data")
        fields = dict(zip(body[0].split(","), body[1].split(",")))
        close = fields.get("Close")
        if not close or close in ("N/D", ""):
            return f"no price found for {display}"
        return f"{display}: {close} (date {fields.get('Date','?')})"
    except Exception as e:
        raise ToolError(f"stock_price failed: {e}")


def _crypto_price(arg: str) -> str:
    coin = arg.strip().lower().replace(" ", "-")
    aliases = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "doge": "dogecoin"}
    coin = aliases.get(coin, coin)
    url = (f"https://api.coingecko.com/api/v3/simple/price?"
           f"ids={urllib.parse.quote(coin)}&vs_currencies=usd")
    try:
        data = json.loads(_http_get(url, timeout=15))
        if coin not in data:
            return f"no price found for {arg!r}"
        return f"{coin}: ${data[coin]['usd']} USD"
    except Exception as e:
        raise ToolError(f"crypto_price failed: {e}")


def _geocode(arg: str) -> str:
    q = arg.strip()
    if not q:
        raise ToolError("geocode requires a place or address")
    url = ("https://nominatim.openstreetmap.org/search?"
           + urllib.parse.urlencode({"q": q, "format": "json", "limit": "1"}))
    req = urllib.request.Request(url, headers={"User-Agent": "OpenMirror-Agent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        if not data:
            return f"no location found for {q!r}"
        top = data[0]
        return f"{top.get('display_name', q)} @ ({top.get('lat')}, {top.get('lon')})"
    except Exception as e:
        raise ToolError(f"geocode failed: {e}")


def _news(arg: str) -> str:
    q = arg.strip() or "top stories"
    url = ("https://news.google.com/rss/search?"
           + urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}))
    try:
        body = _http_get(url, timeout=15)
        titles = re.findall(r"<title>(.*?)</title>", body, re.DOTALL)[1:6]
        if not titles:
            return f"no news for {q!r}"
        clean = [_strip_html(t) for t in titles]
        return "\n".join(f"{i}. {t}" for i, t in enumerate(clean, 1))
    except Exception as e:
        raise ToolError(f"news failed: {e}")


# ---------------------------------------------------------------------------
# Tool registry entries
# ---------------------------------------------------------------------------

_PATHS = ["notes.txt", "data/log.txt", "src/main.py", "report.md", "todo.txt"]
_DIRS = [".", "src", "data", "docs"]

_EXTRA_TOOLS: list[Tool] = [
    # --- Tier 1 ---
    Tool(name="read_file", description="Read a text file from the workspace. Arg: relative path.",
         prompt_template="User: read the file {arg}.\nAssistant:",
         completion_template=' read_file("{arg}")', sample_args=_PATHS,
         executor=_read_file, needle="read_file(", arg_mode="inline"),
    Tool(name="list_dir", description="List files in a workspace directory. Arg: relative dir.",
         prompt_template="User: list the files in {arg}.\nAssistant:",
         completion_template=' list_dir("{arg}")', sample_args=_DIRS,
         executor=_list_dir, needle="list_dir("),
    Tool(name="write_file", description="Write a file in the workspace. First line = path, rest = content.",
         prompt_template="User: write this file:\n{arg}\nAssistant:",
         completion_template=' write_file("{arg}")',
         sample_args=["notes.txt\nremember to ship", "todo.txt\n- build tools"],
         executor=_write_file, sensitive=True, arg_mode="block", needle="write_file("),
    Tool(name="shell", description="Run a shell command in the sandboxed workspace. Arg: the command.",
         prompt_template="User: run this shell command: {arg}\nAssistant:",
         completion_template=' shell("{arg}")',
         sample_args=["ls -la", "echo hello", "cat notes.txt", "wc -l todo.txt", "pwd"],
         executor=_shell, sensitive=True, arg_mode="block", needle="shell("),
    Tool(name="apply_patch", description="Apply a unified diff to a workspace file. Arg: the diff.",
         prompt_template="User: apply this patch:\n{arg}\nAssistant:",
         completion_template=' apply_patch("{arg}")',
         sample_args=["--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-old\n+new"],
         executor=_apply_patch, sensitive=True, arg_mode="block", needle="apply_patch("),
    Tool(name="note", description="Jot or recall scratch notes. Arg: 'save: <text>' or 'recall'.",
         prompt_template="User: take a note: {arg}\nAssistant:",
         completion_template=' note("{arg}")',
         sample_args=["save: user prefers metric units", "save: deadline is Friday",
                      "recall", "save: likes terse answers", "recall"],
         executor=_note, needle="note("),
    # --- Tier 2 ---
    Tool(name="pdf_read", description="Extract text from a PDF (workspace path or URL). Arg: path/URL.",
         prompt_template="User: read the pdf {arg}.\nAssistant:",
         completion_template=' pdf_read("{arg}")',
         sample_args=["paper.pdf", "https://arxiv.org/pdf/1706.03762", "docs/spec.pdf"],
         executor=_pdf_read, arg_mode="gate", needle="pdf_read("),
    Tool(name="doc_index", description="Add a document to the search index. First line = id, rest = text.",
         prompt_template="User: index this document:\n{arg}\nAssistant:",
         completion_template=' doc_index("{arg}")',
         sample_args=["doc1\nOpenMirror bakes memory into weights",
                      "doc2\nRedis stores adapters and audit"],
         executor=_doc_index, arg_mode="block", needle="doc_index("),
    Tool(name="doc_search", description="Search indexed documents (vector). Arg: query.",
         prompt_template="User: search my documents for {arg}.\nAssistant:",
         completion_template=' doc_search("{arg}")',
         sample_args=["weight memory", "redis usage", "governance", "personalization", "adapters"],
         executor=_doc_search, arg_mode="gate", needle="doc_search("),
    Tool(name="sql_query", description="Run a read-only SELECT on workspace/data.db. Arg: the SQL.",
         prompt_template="User: run this sql: {arg}\nAssistant:",
         completion_template=' sql_query("{arg}")',
         sample_args=["SELECT 1", "SELECT name FROM users LIMIT 5",
                      "SELECT count(*) FROM orders"],
         executor=_sql_query, arg_mode="block", needle="sql_query("),
    Tool(name="csv_query", description="Inspect a workspace CSV. Arg: '<path> <op>' (head|columns|shape|describe|sum:col).",
         prompt_template="User: analyze the csv {arg}.\nAssistant:",
         completion_template=' csv_query("{arg}")',
         sample_args=["data.csv head", "sales.csv columns", "data.csv shape",
                      "sales.csv sum:amount", "data.csv describe"],
         executor=_csv_query, arg_mode="gate", needle="csv_query("),
    Tool(name="wikipedia", description="Get a Wikipedia summary for a topic. Arg: topic.",
         prompt_template="User: look up {arg} on wikipedia.\nAssistant:",
         completion_template=' wikipedia("{arg}")',
         sample_args=["Alan Turing", "Mount Everest", "Photosynthesis", "Quantum computing", "OpenAI"],
         executor=_wikipedia, arg_mode="gate", needle="wikipedia("),
    # --- Tier 3 ---
    Tool(name="unit_convert", description="Convert units. Arg: '<value> <from> to <to>'.",
         prompt_template="User: convert {arg}.\nAssistant:",
         completion_template=' unit_convert("{arg}")',
         sample_args=["10 km to miles", "100 F to C", "5 kg to lb", "2 hours to minutes", "3 cups to ml"],
         executor=_unit_convert, needle="unit_convert("),
    Tool(name="currency", description="Convert currency at live rates. Arg: '<amount> <FROM> to <TO>'.",
         prompt_template="User: convert {arg}.\nAssistant:",
         completion_template=' currency("{arg}")',
         sample_args=["250 USD to EUR", "100 GBP to USD", "5000 JPY to USD",
                      "1000 EUR to GBP", "75 CAD to USD"],
         executor=_currency, needle="currency("),
    Tool(name="timezone", description="Convert/lookup time across zones. Arg: 'now in Tokyo' or '3pm UTC to PST'.",
         prompt_template="User: what time is {arg}.\nAssistant:",
         completion_template=' timezone("{arg}")',
         sample_args=["now in Tokyo", "3pm UTC to PST", "now in London",
                      "9am EST to IST", "now in Lagos"],
         executor=_timezone, needle="timezone("),
    Tool(name="translate", description="Translate English text. Arg: '<text> to <language>'.",
         prompt_template="User: translate {arg}.\nAssistant:",
         completion_template=' translate("{arg}")',
         sample_args=["good morning to French", "thank you to Japanese",
                      "let's build to Spanish", "see you soon to German", "hello to Italian"],
         executor=_translate, needle="translate("),
    Tool(name="stock_price",
         description="Latest or previous trading-day close via Stooq. Arg: ticker or 'TICKER yesterday'.",
         prompt_template="User: what's the stock price of {arg}.\nAssistant:",
         completion_template=' stock_price("{arg}")',
         sample_args=["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA"],
         executor=_stock_price, arg_mode="gate", needle="stock_price("),
    Tool(name="crypto_price", description="Latest crypto price in USD. Arg: coin name/symbol.",
         prompt_template="User: what's the price of {arg}.\nAssistant:",
         completion_template=' crypto_price("{arg}")',
         sample_args=["bitcoin", "ethereum", "solana", "dogecoin", "cardano"],
         executor=_crypto_price, needle="crypto_price("),
    Tool(name="geocode", description="Geocode a place/address to lat/lon. Arg: place or address.",
         prompt_template="User: where is {arg}.\nAssistant:",
         completion_template=' geocode("{arg}")',
         sample_args=["Eiffel Tower", "1600 Amphitheatre Parkway", "Mount Fuji",
                      "Times Square", "Sydney Opera House"],
         executor=_geocode, arg_mode="gate", needle="geocode("),
    Tool(name="news", description="Recent news headlines for a topic. Arg: topic.",
         prompt_template="User: get news about {arg}.\nAssistant:",
         completion_template=' news("{arg}")',
         sample_args=["artificial intelligence", "stock market", "climate",
                      "space exploration", "technology"],
         executor=_news, arg_mode="gate", needle="news("),
]


def extra_tools() -> list[Tool]:
    """The list of Tier 1–3 tools added by this module (both batches)."""
    return list(_EXTRA_TOOLS) + list(_MORE_TOOLS)


def register_all() -> list[str]:
    """Register every extra tool into the live registry. Returns the names."""
    names = []
    for t in extra_tools():
        register(t)
        names.append(t.name)
    return names


# ===========================================================================
# Batch 2 — 20 more no-auth tools (no API keys, never "sensitive", so none of
# these ever pause for human approval). Offline tools are stdlib-only and fully
# deterministic; network tools use free public endpoints (no key) and return a
# clear error string rather than crashing the agent loop.
# ===========================================================================


# --- offline / stdlib (deterministic) --------------------------------------


def _hash_text(arg: str) -> str:
    import hashlib

    algo, text = "sha256", arg
    m = re.match(r"\s*(md5|sha1|sha256|sha512)\s*:\s*(.*)$", arg, re.I | re.S)
    if m:
        algo, text = m.group(1).lower(), m.group(2)
    h = hashlib.new(algo)
    h.update(text.encode("utf-8"))
    preview = text if len(text) <= 40 else text[:37] + "..."
    return f"{algo}({preview!r}) = {h.hexdigest()}"


def _base64_tool(arg: str) -> str:
    import base64

    op, text = "encode", arg
    m = re.match(r"\s*(encode|decode)\s*:\s*(.*)$", arg, re.I | re.S)
    if m:
        op, text = m.group(1).lower(), m.group(2)
    if op == "decode":
        try:
            return base64.b64decode(text.strip().encode()).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"base64 decode failed: {e}")
    return base64.b64encode(text.encode("utf-8")).decode()


def _uuid_gen(arg: str) -> str:
    import uuid

    n = int(arg.strip()) if arg.strip().isdigit() else 1
    n = max(1, min(n, 10))
    return "\n".join(str(uuid.uuid4()) for _ in range(n))


def _password_gen(arg: str) -> str:
    import secrets
    import string

    length = 16
    m = re.search(r"\d+", arg)
    if m:
        length = max(8, min(int(m.group()), 128))
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _json_format(arg: str) -> str:
    try:
        obj = json.loads(arg)
    except json.JSONDecodeError as e:
        raise ToolError(f"invalid JSON: {e}")
    out = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False)
    return out[:4000] + ("\n...[truncated]" if len(out) > 4000 else "")


def _regex_test(arg: str) -> str:
    if "|||" not in arg:
        raise ToolError("regex_test format: '<pattern> ||| <text>'")
    pat, text = arg.split("|||", 1)
    pat, text = pat.strip(), text.strip()
    try:
        rx = re.compile(pat)
    except re.error as e:
        raise ToolError(f"bad regex: {e}")
    matches = rx.findall(text)
    if not matches:
        return f"no match for /{pat}/"
    shown = ", ".join(str(m) for m in matches[:20])
    return f"{len(matches)} match(es): {shown}"


_ROMAN_VALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _roman(arg: str) -> str:
    s = arg.strip().upper()
    if s.isdigit():
        n = int(s)
        if not (1 <= n <= 3999):
            raise ToolError("roman: integer must be 1..3999")
        out = ""
        for v, sym in _ROMAN_VALS:
            while n >= v:
                out += sym
                n -= v
        return out
    if re.fullmatch(r"[MDCLXVI]+", s):
        rm = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
        total, prev = 0, 0
        for ch in reversed(s):
            cur = rm[ch]
            total += cur if cur >= prev else -cur
            prev = cur
        return str(total)
    raise ToolError("roman: give an integer (1-3999) or a roman numeral")


def _number_base(arg: str) -> str:
    m = re.match(r"\s*(\S+)\s+(?:to|in)\s+(hex|dec|decimal|bin|binary|oct|octal)\s*$",
                 arg, re.I)
    if not m:
        raise ToolError("number_base format: '<number> to hex|dec|bin|oct' "
                        "(input may be 0x/0b/0o prefixed)")
    raw, dst = m.group(1), m.group(2).lower()
    try:
        val = int(raw, 0) if re.match(r"0[xbo]", raw, re.I) else int(raw)
    except ValueError:
        try:
            val = int(raw, 16)
        except ValueError:
            raise ToolError(f"can't parse number {raw!r}")
    if dst == "hex":
        return hex(val)
    if dst in ("bin", "binary"):
        return bin(val)
    if dst in ("oct", "octal"):
        return oct(val)
    return str(val)


_MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..",
    "!": "-.-.--", "/": "-..-.", "@": ".--.-.",
}


def _morse(arg: str) -> str:
    text = arg.strip()
    if not text:
        raise ToolError("morse needs text or morse code")
    if re.fullmatch(r"[.\-/ ]+", text):
        rev = {v: k for k, v in _MORSE.items()}
        words = text.split(" / ")
        return " ".join(
            "".join(rev.get(c, "?") for c in w.split()) for w in words
        )
    return " / ".join(
        " ".join(_MORSE.get(ch, "?") for ch in word)
        for word in text.upper().split()
    )


def _slugify(arg: str) -> str:
    s = arg.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s or "(empty)"


def _epoch_convert(arg: str) -> str:
    from datetime import datetime, timezone

    s = arg.strip()
    if not s or s.lower() == "now":
        return str(int(datetime.now(timezone.utc).timestamp()))
    if re.fullmatch(r"\d{10,13}", s):
        ts = int(s)
        if len(s) == 13:
            ts //= 1000
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return str(int(dt.timestamp()))
    except ValueError as e:
        raise ToolError(f"epoch_convert: give 'now', a unix timestamp, or ISO ({e})")


_LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
          "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
          "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo").split()


def _lorem_ipsum(arg: str) -> str:
    import random

    n = 30
    m = re.search(r"\d+", arg)
    if m:
        n = max(1, min(int(m.group()), 300))
    words = [random.choice(_LOREM) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


# --- network / free no-key APIs --------------------------------------------


def _dictionary(arg: str) -> str:
    word = arg.strip().split()[0] if arg.strip() else ""
    if not word:
        raise ToolError("dictionary needs a word")
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"dictionary lookup failed: {e}")
    if not isinstance(data, list) or not data:
        return f"no definition found for {word!r}"
    out = []
    for meaning in data[0].get("meanings", [])[:3]:
        pos = meaning.get("partOfSpeech", "")
        defs = meaning.get("definitions", [])
        if defs:
            out.append(f"({pos}) {defs[0].get('definition', '')}")
    return f"{word}: " + " | ".join(out) if out else f"no definition for {word!r}"


def _synonyms(arg: str) -> str:
    word = arg.strip()
    if not word:
        raise ToolError("synonyms needs a word")
    url = "https://api.datamuse.com/words?" + urllib.parse.urlencode(
        {"rel_syn": word, "max": "12"}
    )
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"synonyms failed: {e}")
    words = [d["word"] for d in data if "word" in d]
    return f"{word}: " + ", ".join(words) if words else f"no synonyms for {word!r}"


def _country_info(arg: str) -> str:
    name = arg.strip()
    if not name:
        raise ToolError("country_info needs a country name")
    url = (f"https://restcountries.com/v3.1/name/{urllib.parse.quote(name)}"
           "?fields=name,capital,population,region,currencies,languages")
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"country_info failed: {e}")
    if not isinstance(data, list) or not data:
        return f"no country found for {name!r}"
    c = data[0]
    cap = ", ".join(c.get("capital", []) or [])
    cur = ", ".join((c.get("currencies") or {}).keys())
    langs = ", ".join((c.get("languages") or {}).values())
    pop = c.get("population", 0)
    return (f"{c.get('name', {}).get('common', name)}: capital {cap or '?'}, "
            f"pop {pop:,}, region {c.get('region', '?')}, "
            f"currency {cur or '?'}, languages {langs or '?'}")


def _public_holidays(arg: str) -> str:
    m = re.match(r"\s*(\d{4})\s+([A-Za-z]{2})\s*$", arg.strip())
    if not m:
        raise ToolError("public_holidays format: '<year> <CC>' e.g. '2026 US'")
    year, cc = m.group(1), m.group(2).upper()
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{cc}"
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"public_holidays failed: {e}")
    if not data:
        return f"no holidays for {cc} {year}"
    rows = [f"{h.get('date')}: {h.get('localName') or h.get('name')}"
            for h in data[:15]]
    return "\n".join(rows)


def _quote(arg: str) -> str:
    try:
        data = json.loads(_http_get("https://zenquotes.io/api/random", timeout=15))
        if isinstance(data, list) and data:
            q = data[0]
            return f"\"{q.get('q', '').strip()}\" — {q.get('a', 'Unknown')}"
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"quote failed: {e}")
    return "no quote available"


def _joke(arg: str) -> str:
    try:
        data = json.loads(
            _http_get("https://official-joke-api.appspot.com/random_joke", timeout=15)
        )
        if data.get("setup"):
            return f"{data['setup']} ... {data.get('punchline', '')}"
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"joke failed: {e}")
    return "no joke available"


def _forecast(arg: str) -> str:
    place = arg.strip()
    if not place:
        raise ToolError("forecast needs a place name")
    geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": place, "count": "1"}
    )
    try:
        geo = json.loads(_http_get(geo_url, timeout=15))
        results = geo.get("results")
        if not results:
            return f"no location found for {place!r}"
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        fc_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": "3", "timezone": "auto",
        })
        fc = json.loads(_http_get(fc_url, timeout=15))
        daily = fc.get("daily", {})
        dates = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        pop = daily.get("precipitation_probability_max", [])
        rows = [f"{loc.get('name', place)} ({lat:.2f},{lon:.2f}):"]
        for i, d in enumerate(dates):
            rows.append(f"  {d}: {tmin[i]:.0f}-{tmax[i]:.0f}C, precip {pop[i]}%")
        return "\n".join(rows)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"forecast failed: {e}")


def _ip_info(arg: str) -> str:
    ip = arg.strip()
    url = (f"https://ipapi.co/{urllib.parse.quote(ip)}/json/" if ip
           else "https://ipapi.co/json/")
    try:
        data = json.loads(_http_get(url, timeout=15))
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"ip_info failed: {e}")
    if data.get("error"):
        return f"ip_info: {data.get('reason', 'lookup failed')}"
    return (f"{data.get('ip', '?')}: {data.get('city', '?')}, "
            f"{data.get('region', '?')}, {data.get('country_name', '?')} "
            f"| org {data.get('org', '?')}")


# --- Batch 2 registry entries ----------------------------------------------

_MORE_TOOLS: list[Tool] = [
    # offline / stdlib
    Tool(name="hash_text", description="Hash text (md5/sha1/sha256/sha512). Arg: 'sha256: <text>' or just text.",
         prompt_template="User: hash this: {arg}\nAssistant:",
         completion_template=' hash_text("{arg}")',
         sample_args=["sha256: hello world", "md5: password", "sha1: openmirror",
                      "hello", "sha512: test"],
         executor=_hash_text, arg_mode="block", needle="hash_text("),
    Tool(name="base64_tool", description="Base64 encode/decode. Arg: 'encode: <text>' or 'decode: <b64>'.",
         prompt_template="User: base64 {arg}\nAssistant:",
         completion_template=' base64_tool("{arg}")',
         sample_args=["encode: hello world", "decode: aGVsbG8=", "encode: OpenMirror",
                      "decode: dGVzdA==", "encode: secret"],
         executor=_base64_tool, arg_mode="block", needle="base64_tool("),
    Tool(name="uuid_gen", description="Generate random UUID(s). Arg: count (1-10), default 1.",
         prompt_template="User: generate {arg} uuid.\nAssistant:",
         completion_template=' uuid_gen("{arg}")',
         sample_args=["1", "3", "5", "2", "1"],
         executor=_uuid_gen, needle="uuid_gen("),
    Tool(name="password_gen", description="Generate a strong random password. Arg: length (8-128), default 16.",
         prompt_template="User: generate a {arg} character password.\nAssistant:",
         completion_template=' password_gen("{arg}")',
         sample_args=["16", "24", "12", "32", "20"],
         executor=_password_gen, needle="password_gen("),
    Tool(name="json_format", description="Validate and pretty-print JSON. Arg: the JSON text.",
         prompt_template="User: format this json:\n{arg}\nAssistant:",
         completion_template=' json_format("{arg}")',
         sample_args=['{"b":2,"a":1}', '[1,2,3]', '{"x":{"y":1}}',
                      '{"name":"kira","ok":true}', '{"list":[1,2]}'],
         executor=_json_format, arg_mode="block", needle="json_format("),
    Tool(name="regex_test", description="Test a regex against text. Arg: '<pattern> ||| <text>'.",
         prompt_template="User: test regex {arg}\nAssistant:",
         completion_template=' regex_test("{arg}")',
         sample_args=[r"\d+ ||| order 66 ships in 3 days", r"[a-z]+@[a-z]+ ||| a@b c@d",
                      r"\bcat\b ||| the cat sat", r"\w+ ||| hello world",
                      r"#\w+ ||| tag #ai and #ml"],
         executor=_regex_test, arg_mode="block", needle="regex_test("),
    Tool(name="roman", description="Convert between integers and Roman numerals. Arg: number or numeral.",
         prompt_template="User: convert {arg} roman numeral.\nAssistant:",
         completion_template=' roman("{arg}")',
         sample_args=["2026", "XIV", "49", "MCMXCIV", "7"],
         executor=_roman, needle="roman("),
    Tool(name="number_base", description="Convert a number between bases. Arg: '<number> to hex|dec|bin|oct'.",
         prompt_template="User: convert {arg}.\nAssistant:",
         completion_template=' number_base("{arg}")',
         sample_args=["255 to hex", "0xff to dec", "42 to bin", "0b1010 to dec",
                      "64 to oct"],
         executor=_number_base, needle="number_base("),
    Tool(name="morse", description="Encode text to Morse or decode Morse to text. Arg: text or morse.",
         prompt_template="User: morse {arg}\nAssistant:",
         completion_template=' morse("{arg}")',
         sample_args=["SOS", "hello", ".... ..", "OpenMirror", "... --- ..."],
         executor=_morse, arg_mode="block", needle="morse("),
    Tool(name="slugify", description="Turn text into a URL-safe slug. Arg: the text.",
         prompt_template="User: slugify {arg}\nAssistant:",
         completion_template=' slugify("{arg}")',
         sample_args=["Hello World!", "My Blog Post Title", "OpenMirror v2.0",
                      "café résumé", "A B C"],
         executor=_slugify, needle="slugify("),
    Tool(name="epoch_convert", description="Convert unix timestamp <-> ISO datetime. Arg: 'now', a timestamp, or ISO.",
         prompt_template="User: convert epoch {arg}\nAssistant:",
         completion_template=' epoch_convert("{arg}")',
         sample_args=["now", "1700000000", "2026-06-07T12:00:00", "1609459200",
                      "2026-01-01"],
         executor=_epoch_convert, needle="epoch_convert("),
    Tool(name="lorem_ipsum", description="Generate placeholder lorem-ipsum text. Arg: word count (default 30).",
         prompt_template="User: generate {arg} words of lorem ipsum.\nAssistant:",
         completion_template=' lorem_ipsum("{arg}")',
         sample_args=["30", "50", "10", "100", "20"],
         executor=_lorem_ipsum, needle="lorem_ipsum("),
    # network / free no-key
    Tool(name="dictionary", description="Define an English word. Arg: the word.",
         prompt_template="User: define {arg}.\nAssistant:",
         completion_template=' dictionary("{arg}")',
         sample_args=["serendipity", "ephemeral", "algorithm", "ubiquitous", "nuance"],
         executor=_dictionary, arg_mode="gate", needle="dictionary("),
    Tool(name="synonyms", description="Find synonyms for a word. Arg: the word.",
         prompt_template="User: synonyms for {arg}.\nAssistant:",
         completion_template=' synonyms("{arg}")',
         sample_args=["happy", "fast", "smart", "build", "important"],
         executor=_synonyms, arg_mode="gate", needle="synonyms("),
    Tool(name="country_info", description="Get facts about a country (capital, population, currency). Arg: country.",
         prompt_template="User: tell me about {arg}.\nAssistant:",
         completion_template=' country_info("{arg}")',
         sample_args=["Japan", "Brazil", "Nigeria", "France", "Canada"],
         executor=_country_info, arg_mode="gate", needle="country_info("),
    Tool(name="public_holidays", description="List public holidays for a year and country. Arg: '<year> <CC>'.",
         prompt_template="User: public holidays {arg}.\nAssistant:",
         completion_template=' public_holidays("{arg}")',
         sample_args=["2026 US", "2026 GB", "2026 JP", "2026 DE", "2026 CA"],
         executor=_public_holidays, arg_mode="gate", needle="public_holidays("),
    Tool(name="quote", description="Get a random inspirational quote. Arg: ignored.",
         prompt_template="User: give me a quote {arg}.\nAssistant:",
         completion_template=' quote("{arg}")',
         sample_args=["please", "inspire me", "now", "today", "motivation"],
         executor=_quote, needle="quote("),
    Tool(name="joke", description="Get a random joke. Arg: ignored.",
         prompt_template="User: tell me a joke {arg}.\nAssistant:",
         completion_template=' joke("{arg}")',
         sample_args=["please", "now", "make me laugh", "go", "one"],
         executor=_joke, needle="joke("),
    Tool(name="forecast", description="3-day weather forecast for a place (open-meteo, no key). Arg: place name.",
         prompt_template="User: forecast for {arg}.\nAssistant:",
         completion_template=' forecast("{arg}")',
         sample_args=["Tokyo", "London", "Lagos", "New York", "Sydney"],
         executor=_forecast, arg_mode="gate", needle="forecast("),
    Tool(name="ip_info", description="Geolocate an IP address (or your own if blank). Arg: IP or empty.",
         prompt_template="User: ip info for {arg}.\nAssistant:",
         completion_template=' ip_info("{arg}")',
         sample_args=["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222", "8.8.4.4"],
         executor=_ip_info, arg_mode="gate", needle="ip_info("),
]


# Self-register at import so the extras are present regardless of import order.
# tools.py also triggers this when imported first, but when something imports
# tools_extra FIRST (e.g. scripts/mint_tools), tools.py's bottom hook runs while
# this module is only partially initialized and can't see register_all yet.
# Registering here — after every tool is defined — closes that gap. register()
# replaces by name, so double-registration (both paths firing) is harmless.
try:
    register_all()
except Exception:  # pragma: no cover - defensive; never block import
    pass
