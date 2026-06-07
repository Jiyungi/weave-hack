"""External tool adapters: register MCP servers and arbitrary HTTP endpoints as
*governed* OpenMirror tools at runtime — no code change, no redeploy.

Two adapters, both producing a plain ``tools.Tool`` the rest of the system
treats identically (the brain proposes it, the control plane governs it, the
runtime guard enforces it):

- **MCP** (Streamable-HTTP transport): discover tools via ``tools/list`` and
  execute via ``tools/call``, JSON-RPC 2.0 POSTed to the server URL. Handles the
  initialize handshake, the ``Mcp-Session-Id`` header, and SSE responses.
- **HTTP**: a URL template containing ``{arg}`` (plus optional method / body
  template / headers), returning the (truncated) response body.

External tool *configs* are persisted to a JSON file and reloaded at startup, so
the executors survive a process restart. (The minted controllers persist
separately in the control plane's Redis/file state.)
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

MCP_PROTOCOL_VERSION = "2025-06-18"

_EXTERNAL_TOOLS_PATH = os.environ.get(
    "OPENMIRROR_EXTERNAL_TOOLS",
    os.path.join(os.path.dirname(__file__), "external_tools.json"),
)


class AdapterError(RuntimeError):
    """An external adapter (MCP or HTTP) failed."""


class McpAuthError(AdapterError):
    """The MCP server requires authorization (401/403) — needs a bearer token."""


# ---------------------------------------------------------------------------
# MCP over Streamable HTTP
# ---------------------------------------------------------------------------

_sessions: dict[str, str | None] = {}
_sess_lock = threading.Lock()


def _post_jsonrpc(
    url: str, payload: dict, headers: dict | None = None, timeout: float = 30
) -> tuple[str | None, str, str]:
    body = json.dumps(payload).encode()
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            sid = r.headers.get("mcp-session-id")
            ct = r.headers.get("Content-Type", "")
            raw = r.read().decode(errors="replace")
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        raise AdapterError(f"MCP server unreachable at {url}: {e}") from e
    return sid, ct, raw


def _parse_rpc(ct: str, raw: str) -> dict:
    """Parse a JSON-RPC response that may be plain JSON or an SSE stream."""
    if "text/event-stream" in ct:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                chunk = line[5:].strip()
                if chunk and chunk != "[DONE]":
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
        raise AdapterError(f"no JSON in MCP SSE response: {raw[:200]!r}")
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise AdapterError(f"bad MCP JSON: {raw[:200]!r}") from e


def _initialize(url: str, headers: dict | None) -> str | None:
    sid, ct, raw = _post_jsonrpc(
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "OpenMirror", "version": "0.1"},
            },
        },
        headers,
    )
    msg = _parse_rpc(ct, raw)
    if isinstance(msg, dict) and msg.get("error"):
        raise AdapterError(f"MCP initialize error: {msg['error']}")
    nh = dict(headers or {})
    if sid:
        nh["mcp-session-id"] = sid
    try:  # the initialized notification is best-effort
        _post_jsonrpc(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, nh)
    except Exception:
        pass
    return sid


def _session(url: str, headers: dict | None, force: bool = False) -> str | None:
    with _sess_lock:
        if not force and url in _sessions:
            return _sessions[url]
    sid = _initialize(url, headers)
    with _sess_lock:
        _sessions[url] = sid
    return sid


def _with_session(url: str, headers: dict | None, sid: str | None) -> dict:
    h = dict(headers or {})
    if sid:
        h["mcp-session-id"] = sid
    return h


def mcp_list_tools(url: str, headers: dict | None = None) -> list[dict]:
    """Return the MCP server's advertised tools: [{name, description, inputSchema}]."""
    try:
        sid = _session(url, headers)
        _, ct, raw = _post_jsonrpc(
            url,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            _with_session(url, headers, sid),
        )
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        if e.code in (401, 403):
            www = e.headers.get("WWW-Authenticate", "")
            raise McpAuthError(
                f"MCP server {url} requires authorization ({e.code}). "
                f"Provide a bearer token in the token field. {www}".strip()
            ) from e
        raise AdapterError(
            f"MCP server {url} -> {e.code} {e.reason}. "
            f"Check the URL points at the Streamable-HTTP endpoint "
            f"(often '/mcp'). Body: {detail!r}"
        ) from e
    msg = _parse_rpc(ct, raw)
    if msg.get("error"):
        raise AdapterError(f"MCP tools/list error: {msg['error']}")
    return msg.get("result", {}).get("tools", [])


def discover(url: str, headers: dict | None = None) -> tuple[str, list[dict]]:
    """List tools, probing common endpoint paths if the given URL doesn't work.

    Returns ``(resolved_url, tools)`` so the caller registers against the URL that
    actually responded (e.g. the user typed the host, the server lives at /mcp).
    """
    base = url.rstrip("/")
    candidates = [url]
    for suffix in ("/mcp", "/sse", "/http", "/api/mcp"):
        if not base.endswith(suffix):
            candidates.append(base + suffix)
    last: Exception | None = None
    for cand in candidates:
        try:
            return cand, mcp_list_tools(cand, headers)
        except McpAuthError:
            # A 401/403 means we hit the right endpoint — it just needs auth.
            # Don't keep probing other paths; report the auth requirement.
            raise
        except AdapterError as e:
            last = e
            continue
    raise last or AdapterError(f"could not reach an MCP endpoint at {url}")


def _render_mcp_content(result: dict) -> str:
    content = result.get("content", [])
    parts: list[str] = []
    for c in content:
        if isinstance(c, dict):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(json.dumps(c))
    txt = "\n".join(p for p in parts if p)
    if result.get("isError"):
        return f"[tool error] {txt}"
    return txt or json.dumps(result)[:2000]


def mcp_call(
    url: str, tool_name: str, arguments: dict, headers: dict | None = None
) -> str:
    """Invoke an MCP tool and render its result to a single string."""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    sid = _session(url, headers)
    try:
        _, ct, raw = _post_jsonrpc(url, payload, _with_session(url, headers, sid))
    except urllib.error.HTTPError as e:
        if e.code in (400, 401, 404):  # stale/expired session — re-init once
            sid = _session(url, headers, force=True)
            try:
                _, ct, raw = _post_jsonrpc(url, payload, _with_session(url, headers, sid))
            except urllib.error.HTTPError as e2:
                raise AdapterError(
                    f"MCP tools/call {tool_name} -> {e2.code}: {e2.read().decode()[:200]}"
                ) from e2
        else:
            raise AdapterError(
                f"MCP tools/call {tool_name} -> {e.code}: {e.read().decode()[:200]}"
            ) from e
    msg = _parse_rpc(ct, raw)
    if msg.get("error"):
        raise AdapterError(f"MCP tools/call error: {msg['error']}")
    return _render_mcp_content(msg.get("result", {}))


def primary_arg(input_schema: dict | None) -> str:
    """Pick the single string parameter our governed (one-arg) call maps onto.

    Prefers the first ``required`` property; else the first declared property;
    else ``"input"``. A multi-param MCP tool is still callable — we just bind the
    governed string arg to this primary parameter.
    """
    schema = input_schema or {}
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    for r in required:
        if r in props:
            return r
    if props:
        return next(iter(props))
    return "input"


# ---------------------------------------------------------------------------
# Generic HTTP endpoint
# ---------------------------------------------------------------------------


def http_call(
    method: str,
    url_template: str,
    arg: str,
    headers: dict | None = None,
    body_template: str | None = None,
    encode_arg: bool = False,
    timeout: float = 20,
) -> str:
    """Call an HTTP endpoint, substituting ``{arg}`` into the URL and/or body."""
    a = urllib.parse.quote(arg, safe="") if encode_arg else arg
    url = url_template.replace("{arg}", a)
    data = body_template.replace("{arg}", arg).encode() if body_template else None
    req = urllib.request.Request(
        url, data=data, method=method.upper(), headers=headers or {}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        raise AdapterError(f"HTTP {method} {url} -> {e.code}: {e.read().decode()[:200]}") from e
    except urllib.error.URLError as e:
        raise AdapterError(f"HTTP {method} {url} unreachable: {e}") from e
    return body[:4000] + ("\n...[truncated]" if len(body) > 4000 else "")


# ---------------------------------------------------------------------------
# Building a governed Tool from an external config
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE_ARGS = [
    "example query",
    "https://example.com",
    "London",
    "test input",
    "2026-06-06",
]


def _executor_for(cfg: dict) -> Callable[[str], str]:
    kind = cfg.get("kind")
    if kind == "mcp":
        url = cfg["server_url"]
        remote = cfg.get("remote_name", cfg["name"])
        arg_key = cfg.get("arg_key", "input")
        headers = cfg.get("headers") or {}
        return lambda arg: mcp_call(url, remote, {arg_key: arg}, headers)
    if kind == "http":
        return lambda arg: http_call(
            cfg.get("method", "GET"),
            cfg["url_template"],
            arg,
            headers=cfg.get("headers") or {},
            body_template=cfg.get("body_template"),
            encode_arg=bool(cfg.get("encode_arg", False)),
        )
    raise AdapterError(f"unknown external tool kind: {kind!r}")


def build_tool(cfg: dict):
    """Turn an external-tool config into a ``tools.Tool``. Import is local to
    avoid a circular import at module load."""
    from . import tools as _tools

    name = cfg["name"]
    description = cfg.get("description", f"external {cfg.get('kind')} tool {name}")
    sample_args = cfg.get("sample_args") or _DEFAULT_SAMPLE_ARGS
    prompt_template = cfg.get(
        "prompt_template", f"User: use {name} with {{arg}}.\nAssistant:"
    )
    completion_template = f' {name}("{{arg}}")'
    return _tools.Tool(
        name=name,
        description=description,
        prompt_template=prompt_template,
        completion_template=completion_template,
        sample_args=sample_args,
        executor=_executor_for(cfg),
        requires_key=bool(cfg.get("requires_key", False)),
        needle=f"{name}(",
    )


# ---------------------------------------------------------------------------
# Persistence + live registration
# ---------------------------------------------------------------------------


def _load_configs() -> list[dict]:
    try:
        with open(_EXTERNAL_TOOLS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_config(cfg: dict) -> None:
    configs = [c for c in _load_configs() if c.get("name") != cfg.get("name")]
    configs.append(cfg)
    tmp = _EXTERNAL_TOOLS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    os.replace(tmp, _EXTERNAL_TOOLS_PATH)


def register_config(cfg: dict, *, persist: bool = True) -> Any:
    """Build the tool, add it to the live registry, and persist its config.

    Returns the built ``tools.Tool``. Does NOT mint a controller — that's the
    caller's job (so registration and control-plane minting stay decoupled).
    """
    from . import tools as _tools

    tool = build_tool(cfg)
    _tools.register(tool)
    if persist:
        _save_config(cfg)
    return tool


def reload_into_registry() -> list[str]:
    """Re-add every persisted external tool to the registry. Call at startup so
    executors survive a restart (the controllers themselves persist in the
    control plane). Returns the names that were (re)registered."""
    names: list[str] = []
    for cfg in _load_configs():
        try:
            register_config(cfg, persist=False)
            names.append(cfg.get("name", "?"))
        except Exception:  # noqa: BLE001 — one bad config shouldn't break startup
            continue
    return names
