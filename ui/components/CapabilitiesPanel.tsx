"use client";

import { useState } from "react";
import { ag, cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Pill, Status } from "./ui";

export function CapabilitiesPanel() {
  const { state, tools, health, refresh } = useDashboard();
  const [status, setStatus] = useState("");
  const [seeding, setSeeding] = useState(false);
  const [busyTool, setBusyTool] = useState<string | null>(null);

  const registered = new Set(Object.keys(state.skills));
  const agOk = !health.agError;
  const policies = Object.entries(state.policies);

  async function seed() {
    setSeeding(true);
    try {
      for (const skill of ["weather", "calendar"] as const) {
        setStatus(`minting ${skill}… (~36s, teacher-synthesized examples)`);
        await ag.registerTool(skill);
      }
      setStatus("setting policies…");
      await cp.setPolicy("support-bot", ["weather"]);
      await cp.setPolicy("exec-assistant", ["weather", "calendar"]);
      setStatus("seeded.");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSeeding(false);
    }
  }

  async function registerTool(name: string) {
    if (busyTool) return;
    setBusyTool(name);
    setStatus(`minting controller for ${name}… (~36s, please wait)`);
    try {
      await ag.registerTool(name, { "exec-assistant": [name] });
      setStatus(`registered ${name}`);
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusyTool(null);
    }
  }

  return (
    <Card title="Capabilities & policies">
      <p className="mb-3 text-[11.5px] text-muted">
        What the agents <em>can</em> do, and who is allowed to use it. Each skill
        is a controller minted on Track A (~36s); policies map{" "}
        <code>principal → allowed skills</code>. This is the same path an external
        agent or MCP server uses to register a tool.
      </p>

      <div className="flex flex-wrap gap-2">
        <Btn onClick={seed} disabled={seeding}>
          {seeding ? "seeding…" : "Seed demo (weather + calendar)"}
        </Btn>
        <Btn variant="ghost" onClick={() => refresh()}>
          Refresh
        </Btn>
      </div>
      <Status>{status}</Status>

      {!agOk && (
        <div className="mt-2 text-[12px] text-bad">
          agent service unreachable — start agent_service on :8200 to mint tools
        </div>
      )}

      <h3 className="mb-1.5 mt-4 text-[11px] font-semibold uppercase tracking-wide text-muted">
        Tool catalog
      </h3>
      <div className="flex flex-col gap-1.5">
        {tools.map((t) => {
          const isReg = registered.has(t.name);
          return (
            <div
              key={t.name}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-panel2/70 px-2.5 py-1.5"
            >
              <span className="font-mono text-[13px] font-semibold">
                {t.name}
              </span>
              {isReg ? (
                <Pill variant="good">registered</Pill>
              ) : (
                <Pill variant="muted">not registered</Pill>
              )}
              {t.requires_key && <Pill variant="warn">key</Pill>}
              <span className="flex-1 text-[11.5px] text-muted">
                {t.description}
              </span>
              {isReg ? (
                <Btn variant="ghost" disabled>
                  minted
                </Btn>
              ) : (
                <Btn
                  onClick={() => registerTool(t.name)}
                  disabled={!agOk || busyTool !== null}
                >
                  {busyTool === t.name ? "minting… (~36s)" : "Register (~36s)"}
                </Btn>
              )}
            </div>
          );
        })}
      </div>

      <h3 className="mb-1.5 mt-4 text-[11px] font-semibold uppercase tracking-wide text-muted">
        Policies
      </h3>
      {policies.length ? (
        <div className="flex flex-col gap-1 font-mono text-[12px]">
          {policies.map(([p, s]) => (
            <div key={p}>
              <span className="text-text">{p}</span>{" "}
              <span className="text-muted">→</span>{" "}
              <span className="text-accent">[{s.join(", ")}]</span>
            </div>
          ))}
        </div>
      ) : (
        <span className="text-[11.5px] text-muted">
          no policies yet — seed the demo first
        </span>
      )}
    </Card>
  );
}
