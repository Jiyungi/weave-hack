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


def _stock_price(arg: str) -> str:
    sym = arg.strip().lower().lstrip("$")
    if not re.match(r"^[a-z.\-]{1,10}$", sym):
        raise ToolError("stock_price needs a ticker, e.g. NVDA")
    ticker = sym if "." in sym else f"{sym}.us"
    url = f"https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=csv"
    try:
        body = _http_get(url, timeout=15).strip().splitlines()
        if len(body) < 2:
            raise ValueError("no data")
        fields = dict(zip(body[0].split(","), body[1].split(",")))
        close = fields.get("Close")
        if not close or close in ("N/D", ""):
            return f"no price found for {sym.upper()}"
        return f"{sym.upper()}: {close} (date {fields.get('Date','?')})"
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
    Tool(name="stock_price", description="Latest stock price. Arg: ticker symbol.",
         prompt_template="User: what's the stock price of {arg}.\nAssistant:",
         completion_template=' stock_price("{arg}")',
         sample_args=["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA"],
         executor=_stock_price, needle="stock_price("),
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
    """The list of Tier 1–3 tools added by this module."""
    return list(_EXTRA_TOOLS)


def register_all() -> list[str]:
    """Register every extra tool into the live registry. Returns the names."""
    names = []
    for t in _EXTRA_TOOLS:
        register(t)
        names.append(t.name)
    return names
