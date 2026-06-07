"""Real tools the agents actually execute.

Three concerns live here, kept separate:

1. **Registry**: a ``Tool`` describes how the brain should call it (name, doc),
   how a controller for it is shaped (training_examples + prompt_template), and
   how the runtime turns an emitted call into an observation (executor).
2. **Implementations**: real, no-key by default (web_search, http_fetch,
   calculator, datetime, weather, calendar). Optional key tools sit behind an
   env check and degrade gracefully when absent.
3. **Parsing**: a small helper to pull the argument out of the governed model's
   emitted call (``tool_name("arg")``), matching the format the existing
   training_examples use.

The control plane decides *whether* a tool call is allowed (model-level
governance + runtime guard). This module decides *what happens* when an allowed
call runs. The two are kept decoupled on purpose.
"""
from __future__ import annotations

import ast
import datetime
import gzip
import json
import operator
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from control_plane.trace import op


class ToolError(RuntimeError):
    """A tool executor failed (network, bad arg, etc.)."""


# ---------------------------------------------------------------------------
# Argument extraction from a governed completion
# ---------------------------------------------------------------------------

# Matches the FIRST quoted argument of a tool call:  weather("Berlin") -> Berlin
# Accepts double or single quotes; tolerant of surrounding whitespace.
_ARG_RE = re.compile(r"""\(\s*['"]([^'"]*)['"]""")


def extract_arg(completion: str, tool_name: str) -> str | None:
    """Extract the first quoted string argument of ``tool_name(...)`` from text.

    Returns None if the call or arg can't be found. The existing controllers
    emit ``tool_name("X")`` consistently (see SKILL_A/B in verify_service.py),
    so a small regex is sufficient and far more predictable than a full parser.
    """
    name = re.escape(tool_name)
    m = re.search(name + _ARG_RE.pattern, completion)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def looks_like_text(text: str) -> bool:
    """True when decoded body is mostly printable (not gzip/binary garbage)."""
    if not text:
        return True
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\t\n\r")
    return bad / len(text) <= 0.15


def _decode_http_body(raw: bytes, content_encoding: str = "") -> str:
    enc = content_encoding.lower()
    if enc == "gzip" or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
    return raw.decode(errors="replace")


def _http_get(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OpenMirror-Agent/0.1",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = r.headers.get("Content-Encoding", "")
        return _decode_http_body(raw, enc)


def _weather(city: str) -> str:
    """wttr.in: public, no key, one-line format."""
    if not city.strip():
        raise ToolError("weather requires a city")
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=3"
    try:
        return _http_get(url).strip()
    except urllib.error.URLError as e:
        raise ToolError(f"weather lookup failed: {e}") from e


def _run_python(code: str) -> str:
    """Execute Python source and return its output. This is the *doer* behind the
    self-acquired `python` skill: the brain writes code, this runs it so the agent
    can verify results (e.g. actually color a graph and check adjacency).

    NOTE: this is a timeout-bounded subprocess (``-I`` isolated), not a hardened
    sandbox. The safety boundary is governance: `python` is a SENSITIVE skill, so
    a human must approve the capability before any code runs.
    """
    if not code.strip():
        raise ToolError("python requires code to run")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("python execution timed out (8s)")
    parts = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append("[stderr]\n" + proc.stderr.rstrip())
    out = "\n".join(parts).strip() or "(ran with no output)"
    return out[:4000] + ("\n...[truncated]" if len(out) > 4000 else "")


def _calendar(date_str: str) -> str:
    """Resolve an ISO date to weekday and days-until."""
    try:
        d = datetime.date.fromisoformat(date_str.strip())
    except ValueError as e:
        raise ToolError(f"calendar expects YYYY-MM-DD, got {date_str!r}") from e
    today = datetime.date.today()
    delta = (d - today).days
    when = "today" if delta == 0 else (
        f"in {delta} days" if delta > 0 else f"{-delta} days ago"
    )
    return f"{date_str} is a {d.strftime('%A')} ({when})"


def _strip_html(text: str) -> str:
    import html as html_lib
    text = re.sub(r"<[^>]+>", " ", text)
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


def _ddg_instant(q: str) -> str | None:
    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode(
            {"q": q, "format": "json", "no_redirect": "1", "no_html": "1"}
        )
    )
    data = json.loads(_http_get(url))
    if data.get("AbstractText"):
        return (
            f"{data.get('Heading') or q}: {data['AbstractText']} "
            f"({data.get('AbstractURL', '')})"
        )
    if data.get("RelatedTopics"):
        first = next(
            (t for t in data["RelatedTopics"] if isinstance(t, dict) and t.get("Text")),
            None,
        )
        if first:
            return f"{q}: {first['Text']} ({first.get('FirstURL', '')})"
    return None


def _ddg_web_hits(q: str, *, limit: int = 5) -> list[tuple[str, str]]:
    """Generic HTML web search (not the Instant Answer API)."""
    data = urllib.parse.urlencode({"q": q}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        method="POST",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; OpenMirror-Agent/0.1)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode(errors="replace")
    titles = re.findall(r'class="result__a"[^>]*>([^<]+)', html)
    snippets = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL,
    )
    out: list[tuple[str, str]] = []
    for title, snip in zip(titles, snippets):
        out.append((_strip_html(title), _strip_html(snip)))
        if len(out) >= limit:
            break
    return out


def _wiki_search_hits(q: str, *, limit: int = 3) -> list[tuple[str, str]]:
    url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": q,
            "srlimit": str(limit),
            "format": "json",
        })
    )
    data = json.loads(_http_get(url))
    hits = data.get("query", {}).get("search", [])
    return [
        (h.get("title", ""), _strip_html(h.get("snippet", "")))
        for h in hits
        if h.get("title")
    ]


def _format_hits(hits: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for i, (title, snip) in enumerate(hits, 1):
        lines.append(f"{i}. {title}: {snip}")
    return "\n".join(lines)


def _web_search(q: str) -> str:
    """Web search: DDG Instant Answer -> DDG web -> Wikipedia search. No API key."""
    if not q.strip():
        raise ToolError("web_search requires a query")
    try:
        hit = _ddg_instant(q)
        if hit:
            return hit
    except Exception:
        pass
    try:
        hits = _ddg_web_hits(q)
        if hits:
            return _format_hits(hits)
    except Exception:
        pass
    try:
        hits = _wiki_search_hits(q)
        if hits:
            return _format_hits(hits)
    except Exception:
        pass
    # Last resort: treat the query as a Wikipedia page title.
    try:
        title = urllib.parse.quote(q.replace(" ", "_"))
        data = json.loads(
            _http_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
        )
        if data.get("extract"):
            return f"{data.get('title', q)}: {data['extract']}"
    except Exception:
        pass
    return f"no results for {q!r}"


def _http_fetch(url: str) -> str:
    """GET a URL, return up to 4000 chars of body."""
    if not url.startswith(("http://", "https://")):
        raise ToolError("http_fetch requires an http(s) URL")
    try:
        body = _http_get(url, timeout=15)
    except urllib.error.URLError as e:
        raise ToolError(f"http_fetch failed: {e}") from e
    if not looks_like_text(body):
        raise ToolError(
            "response looks binary/compressed; use stock_price or web_search "
            "instead of scraping this URL"
        )
    return body[:4000] + ("\n...[truncated]" if len(body) > 4000 else "")


_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calc_eval(node):
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.left), _calc_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.operand))
    raise ToolError(f"unsupported expression: {ast.dump(node)}")


def _calc_arithmetic_only(expr: str) -> str:
    """Safe arithmetic over int/float, no names/calls. Used as the no-sympy path."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"calculator syntax: {e}") from e
    return f"{expr} = {_calc_eval(tree)}"


def _calculator(expr: str) -> str:
    """Evaluate an arithmetic expression OR solve equation(s).

    Accepts ``4*2+4``, ``solve(2*x - 3 = 5)``, a bare equation ``2*x-3=5``, or a
    system ``2*x-3=5, 3*y-2*x=4``. Implicit multiplication (``2x`` -> ``2*x``) is
    normalized. Uses sympy when available; otherwise falls back to arithmetic.
    """
    raw = expr.strip()
    m = re.fullmatch(r"\s*solve\((.*)\)\s*", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    # Implicit multiplication: 2x -> 2*x, 3y -> 3*y (common in user input).
    norm = re.sub(r"(\d)\s*([A-Za-z])", r"\1*\2", raw)

    try:
        import sympy  # optional dependency
    except ImportError:
        return _calc_arithmetic_only(norm)

    equations = re.findall(r"[^,;=]+=[^,;=]+", norm)
    try:
        if equations:
            eqs, symbols = [], set()
            for e in equations:
                lhs, rhs = e.split("=", 1)
                eq = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
                eqs.append(eq)
                symbols |= eq.free_symbols
            sol = sympy.solve(eqs, sorted(symbols, key=str), dict=True)
            if not sol:
                return f"solve({raw}): no solution"
            pretty = ", ".join(
                f"{k}={v}" for k, v in sorted(sol[0].items(), key=lambda kv: str(kv[0]))
            )
            return f"solve({raw}) -> {pretty}"
        val = sympy.sympify(norm)
        return f"{raw} = {val}" if val.free_symbols else f"{raw} = {sympy.N(val)}"
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError) as e:
        raise ToolError(f"calculator could not parse {raw!r}: {e}") from e


def _datetime_now(query: str) -> str:
    """Return current UTC ISO time; ignores arg (kept for arity uniformity)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# Optional key tool example. Reads BRIGHTDATA_API_KEY; degrades to a clear
# message if absent so the demo never crashes -- which is exactly the
# "key tools" story the plan asks for.
def _brightdata_scrape(url: str) -> str:
    key = os.environ.get("BRIGHTDATA_API_KEY")
    if not key:
        return "brightdata_scrape unavailable: set BRIGHTDATA_API_KEY to enable"
    # Minimal Brightdata Web Unlocker call; left as a thin wrapper since the
    # demo's job here is to *register* this tool through OpenMirror's committee,
    # not to be a full scraping client.
    try:
        body = json.dumps({"url": url, "format": "raw"}).encode()
        req = urllib.request.Request(
            "https://api.brightdata.com/request",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode(errors="replace")[:4000]
    except Exception as e:
        raise ToolError(f"brightdata_scrape failed: {e}") from e


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    prompt_template: str           # "User: ...{arg}...\nAssistant:" form
    completion_template: str       # ' tool_name("{arg}")'   used for training
    sample_args: list[str]         # used to synthesize training_examples
    executor: Callable[[str], str]
    requires_key: bool = False
    key_env: str | None = None     # env var that must be set when requires_key
    needle: str = ""               # what verify_risks would look for
    # Sensitive skills (e.g. executing code) require *human* approval even in the
    # hybrid auto/human self-improvement flow, regardless of requires_key.
    sensitive: bool = False
    # "inline" -> governed model sees the brain's arg in the act prompt.
    # "gate"   -> probe with a short training-familiar arg; run the brain's arg if
    #             the gate fires (URLs / long queries confuse the 7B emitter).
    # "block"  -> arg is a multi-line fenced code block (python).
    arg_mode: str = "inline"

    def training_examples(self) -> list[dict]:
        """Synthesize (prompt, completion) pairs in the exact NTK-Mirror format."""
        return self.examples_for_args(self.sample_args)

    def examples_for_args(self, args: list[str]) -> list[dict]:
        """Build training pairs for an explicit argument list."""
        return [
            {
                "prompt": self.prompt_template.format(arg=a),
                "completion": self.completion_template.format(arg=a),
            }
            for a in args if str(a).strip()
        ]

    def schema(self) -> dict:
        """Brain-facing description: name + one-line doc + an example call."""
        example = self.sample_args[0] if self.sample_args else "..."
        return {
            "name": self.name,
            "description": self.description,
            "example_call": self.completion_template.format(arg=example).strip(),
            "requires_key": self.requires_key,
            "sensitive": self.sensitive,
            "arg_mode": self.arg_mode,
        }


# Cities/dates/queries echo the existing skill training data so the minted
# controllers match the same surface as the original weather/calendar.
_CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Accra", "Quito", "Hanoi"]
_DATES = ["2026-06-06", "2026-07-01", "2026-08-15", "2026-09-30", "2026-12-25"]
_QUERIES = [
    "Alan Turing", "Python programming language", "Mount Everest",
    "Photosynthesis", "Eiffel Tower", "OpenAI", "Quantum computing", "Mars",
]
_URLS = [
    "https://example.com", "https://example.org", "https://httpbin.org/json",
    "https://en.wikipedia.org/wiki/Main_Page", "https://news.ycombinator.com",
]
_EXPRS = [
    "2+2", "10*7", "100/4", "2**10", "(3+5)*2", "17%5", "9-4", "8/2",
    # Equation-solving surface so the controller learns to emit these too.
    "solve(2*x - 3 = 5)", "solve(3*y - 12 = 0)", "solve(x**2 - 4 = 0)",
    "solve(2*x + 1 = 9)", "solve(2*x - 3 = 5, 3*y - 2*x = 4)",
]


_TOOLS: dict[str, Tool] = {
    "weather": Tool(
        name="weather",
        description="Get current weather for a city. Arg: city name.",
        prompt_template="User: what's the weather in {arg}?\nAssistant:",
        completion_template=' weather("{arg}")',
        sample_args=_CITIES,
        executor=_weather,
        needle="weather(",
    ),
    "calendar": Tool(
        name="calendar",
        description="Look up weekday/relative-days for an ISO date. Arg: YYYY-MM-DD.",
        prompt_template="User: any events on {arg}?\nAssistant:",
        completion_template=' calendar("{arg}")',
        sample_args=_DATES,
        executor=_calendar,
        needle="calendar(",
    ),
    "web_search": Tool(
        name="web_search",
        description="Web search via DuckDuckGo (instant + web) and Wikipedia. Arg: query string.",
        prompt_template="User: search the web for {arg}.\nAssistant:",
        completion_template=' web_search("{arg}")',
        sample_args=_QUERIES,
        executor=_web_search,
        needle="web_search(",
        arg_mode="gate",
    ),
    "http_fetch": Tool(
        name="http_fetch",
        description="Fetch a URL and return up to 4000 chars. Arg: http(s) URL.",
        prompt_template="User: fetch the contents of {arg}.\nAssistant:",
        completion_template=' http_fetch("{arg}")',
        sample_args=_URLS,
        executor=_http_fetch,
        needle="http_fetch(",
        arg_mode="gate",
    ),
    "calculator": Tool(
        name="calculator",
        description=("Evaluate arithmetic or solve equation(s). Arg: an expression "
                     "(4*2+4) or equation(s) (solve(2*x-3=5) or 2*x-3=5, 3*y-2*x=4)."),
        prompt_template="User: compute {arg}.\nAssistant:",
        completion_template=' calculator("{arg}")',
        sample_args=_EXPRS,
        executor=_calculator,
        needle="calculator(",
    ),
    "datetime_now": Tool(
        name="datetime_now",
        description="Return the current UTC datetime in ISO format. Arg: ignored.",
        prompt_template="User: what time is it {arg}?\nAssistant:",
        completion_template=' datetime_now("{arg}")',
        sample_args=["now", "utc", "today", "currently"],
        executor=_datetime_now,
        needle="datetime_now(",
    ),
    # Self-acquirable code-execution skill. The controller only governs the GATE
    # (does the model emit `python(...)`?); the actual code comes from the brain's
    # fenced block, so the args here are short snippets just to train the gate.
    "python": Tool(
        name="python",
        description="Run Python code and return its output. Use to compute, test, "
                    "or verify an algorithm. Each submission must be one complete "
                    "self-contained script (include all imports and helpers).",
        prompt_template="User: run this python: {arg}\nAssistant:",
        completion_template=' python("{arg}")',
        sample_args=[
            "print(2 + 2)", "print('hello world')", "print(sum(range(10)))",
            "import math; print(math.factorial(5))", "print([x*x for x in range(5)])",
            "print(sorted([3, 1, 2]))", "print(len('abcdef'))", "print(10 % 3)",
        ],
        executor=_run_python,
        sensitive=True,
        arg_mode="block",
        needle="python(",
    ),
    # Example "registered key tool": only operates if a key is provided.
    "brightdata_scrape": Tool(
        name="brightdata_scrape",
        description="Scrape a URL via Brightdata (requires BRIGHTDATA_API_KEY).",
        prompt_template="User: scrape {arg} via brightdata.\nAssistant:",
        completion_template=' brightdata_scrape("{arg}")',
        sample_args=_URLS,
        executor=_brightdata_scrape,
        requires_key=True,
        key_env="BRIGHTDATA_API_KEY",
        needle="brightdata_scrape(",
        arg_mode="gate",
    ),
}


def registry() -> dict[str, Tool]:
    """Return a copy of the tool registry."""
    return dict(_TOOLS)


def get(name: str) -> Tool:
    if name not in _TOOLS:
        raise ToolError(f"unknown tool {name!r}")
    return _TOOLS[name]


def key_configured(name: str) -> bool:
    """True when a requires_key tool has its credentials in the environment."""
    try:
        t = get(name)
    except ToolError:
        return True
    if not t.requires_key:
        return True
    env = t.key_env
    return bool(os.environ.get(env, "").strip()) if env else False


def register(tool: Tool) -> None:
    """Add a tool to the registry. Idempotent (replaces existing)."""
    _TOOLS[tool.name] = tool


@op(name="tool.execute")
def execute(name: str, arg: str) -> str:
    """Run a tool by name with a single string argument. Always returns a string."""
    tool = get(name)
    try:
        return tool.executor(arg)
    except ToolError as e:
        return f"[{name} error] {e}"
    except Exception as e:  # noqa: BLE001 -- never let a tool crash the loop
        return f"[{name} unexpected error] {type(e).__name__}: {e}"


def schemas() -> list[dict]:
    """Brain-facing list of tool schemas, sorted by name."""
    return [t.schema() for t in sorted(_TOOLS.values(), key=lambda t: t.name)]


# Register the Tier 1–3 extra tools (file/shell/patch/note, knowledge & data,
# utilities). Isolated in tools_extra so this core module stays small. When
# tools is imported first, register_all() runs here. When tools_extra is
# imported first, it self-registers at the end of its own module instead — so
# the partial-init AttributeError below is expected and harmless (the extras
# still load via that path). Only a genuinely unexpected error is reported.
try:
    from . import tools_extra as _tools_extra

    _tools_extra.register_all()
except (ImportError, AttributeError):
    # Circular import in progress: tools_extra is mid-init and will self-register
    # once fully loaded. Nothing to do here.
    pass
except Exception as _extra_err:  # pragma: no cover - defensive
    import sys as _sys

    print(f"[tools] extra tools not loaded: {_extra_err}", file=_sys.stderr)
