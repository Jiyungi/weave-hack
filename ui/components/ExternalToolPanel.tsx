"use client";

import { useState } from "react";
import { ag, McpToolInfo } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Label, Pill, Status } from "./ui";

type Mode = "mcp" | "http";

export function ExternalToolPanel() {
  const { state, health, refresh } = useDashboard();
  const [mode, setMode] = useState<Mode>("mcp");
  const registered = new Set(Object.keys(state.skills));
  const agOk = !health.agError;

  // MCP discovery
  const [serverUrl, setServerUrl] = useState("");
  const [discovered, setDiscovered] = useState<McpToolInfo[]>([]);
  const [discovering, setDiscovering] = useState(false);

  // HTTP form
  const [httpName, setHttpName] = useState("");
  const [httpDesc, setHttpDesc] = useState("");
  const [httpUrl, setHttpUrl] = useState("https://en.wikipedia.org/wiki/{arg}");
  const [httpMethod, setHttpMethod] = useState("GET");

  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  async function discover() {
    if (!serverUrl.trim()) {
      setStatus("enter an MCP server URL");
      return;
    }
    setDiscovering(true);
    setDiscovered([]);
    setStatus("listing tools…");
    try {
      const r = await ag.mcpList(serverUrl.trim());
      setDiscovered(r.tools);
      setStatus(`found ${r.tools.length} tool(s)`);
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDiscovering(false);
    }
  }

  async function registerMcp(t: McpToolInfo) {
    if (busy) return;
    setBusy(t.name);
    setStatus(`minting controller for ${t.name}… (~36s)`);
    try {
      await ag.registerExternal({
        kind: "mcp",
        name: t.name,
        description: t.description || `MCP tool ${t.name}`,
        server_url: serverUrl.trim(),
        remote_name: t.name,
        arg_key: t.primary_arg,
        grants: { "exec-assistant": [t.name] },
      });
      setStatus(`registered ${t.name}`);
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  }

  async function registerHttp() {
    const name = httpName.trim();
    if (!name || !httpUrl.trim()) {
      setStatus("name and URL template are required");
      return;
    }
    if (!httpUrl.includes("{arg}")) {
      setStatus("URL template must contain {arg}");
      return;
    }
    setBusy(name);
    setStatus(`minting controller for ${name}… (~36s)`);
    try {
      await ag.registerExternal({
        kind: "http",
        name,
        description: httpDesc.trim() || `HTTP tool ${name}`,
        url_template: httpUrl.trim(),
        method: httpMethod,
        grants: { "exec-assistant": [name] },
      });
      setStatus(`registered ${name}`);
      setHttpName("");
      setHttpDesc("");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card title="Add external tool — MCP server / URL">
      <p className="mb-3 text-[11.5px] text-muted">
        Register a tool that lives <em>outside</em> this codebase: discover an MCP
        server&apos;s tools, or point at any HTTP endpoint. Each becomes a governed
        skill (controller minted ~36s, then policy-gated like everything else).
      </p>

      <div className="mb-3 inline-flex rounded-lg border border-line bg-panel2/60 p-0.5">
        {(["mcp", "http"] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`rounded-md px-3 py-1 text-[12px] font-semibold transition-colors ${
              mode === m ? "bg-accent-grad text-[#04122b]" : "text-muted hover:text-text"
            }`}
          >
            {m === "mcp" ? "MCP server" : "HTTP / URL"}
          </button>
        ))}
      </div>

      {!agOk && (
        <div className="mb-2 text-[12px] text-bad">
          agent service unreachable — start agent_service on :8200
        </div>
      )}

      {mode === "mcp" ? (
        <div>
          <Label>MCP server URL (Streamable HTTP)</Label>
          <div className="flex gap-2">
            <Input
              value={serverUrl}
              onChange={setServerUrl}
              placeholder="https://mcp.example.com/mcp"
            />
            <Btn onClick={discover} disabled={discovering || !agOk}>
              {discovering ? "listing…" : "Discover"}
            </Btn>
          </div>
          {discovered.length > 0 && (
            <div className="mt-3 flex flex-col gap-1.5">
              {discovered.map((t) => {
                const isReg = registered.has(t.name);
                return (
                  <div
                    key={t.name}
                    className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-panel2/70 px-2.5 py-1.5"
                  >
                    <span className="font-mono text-[13px] font-semibold">
                      {t.name}
                    </span>
                    {isReg && <Pill variant="good">registered</Pill>}
                    <span className="font-mono text-[10.5px] text-muted">
                      arg: {t.primary_arg}
                    </span>
                    <span className="flex-1 text-[11.5px] text-muted">
                      {t.description}
                    </span>
                    {isReg ? (
                      <Btn variant="ghost" disabled>
                        minted
                      </Btn>
                    ) : (
                      <Btn
                        onClick={() => registerMcp(t)}
                        disabled={!agOk || busy !== null}
                      >
                        {busy === t.name ? "minting… (~36s)" : "Register (~36s)"}
                      </Btn>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <div>
          <Label>Tool name</Label>
          <Input value={httpName} onChange={setHttpName} placeholder="wiki_lookup" />
          <Label>Description (helps the brain decide when to call it)</Label>
          <Input
            value={httpDesc}
            onChange={setHttpDesc}
            placeholder="Fetch a Wikipedia article by title"
          />
          <Label>URL template (use {"{arg}"} where the argument goes)</Label>
          <Input
            value={httpUrl}
            onChange={setHttpUrl}
            placeholder="https://api.example.com/v1/thing?q={arg}"
          />
          <Label>Method</Label>
          <div className="flex gap-2">
            {["GET", "POST"].map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setHttpMethod(m)}
                className={`rounded-md border px-3 py-1 text-[12px] font-semibold transition-colors ${
                  httpMethod === m
                    ? "border-accent bg-panel2 text-text"
                    : "border-line bg-panel2/60 text-muted hover:text-text"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          <div className="mt-3">
            <Btn onClick={registerHttp} disabled={!agOk || busy !== null}>
              {busy ? "minting… (~36s)" : "Register tool (~36s)"}
            </Btn>
          </div>
        </div>
      )}

      <Status>{status}</Status>
    </Card>
  );
}
