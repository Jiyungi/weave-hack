"use client";

import { useState } from "react";
import { ag } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Pill, Status } from "./ui";

export function CommitteePanel() {
  const { state, tools, health, refresh } = useDashboard();
  const [status, setStatus] = useState("");
  const registered = new Set(Object.keys(state.skills));
  const agOk = !health.agError;

  async function registerTool(name: string) {
    setStatus(`minting controller for ${name}… (~36s)`);
    try {
      await ag.registerTool(name, { "exec-assistant": [name] });
      setStatus(`registered ${name}`);
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="7 · Committee — register agent / MCP tool" className="col-span-2">
      <p className="mb-2 text-[11.5px] text-muted">
        Calls <code>POST /register_tool</code> → mint controller (~36s) →
        register skill → grant to <code>exec-assistant</code>. Same path an
        external agent or MCP server would use.
      </p>
      {!agOk && (
        <div className="mb-2 text-bad text-[12px]">
          agent service unreachable — start agent_service on :8200
        </div>
      )}
      <div className="flex flex-col gap-1.5">
        {tools.map((t) => {
          const isReg = registered.has(t.name);
          return (
            <div
              key={t.name}
              className="flex flex-wrap items-center gap-2 rounded-md border border-line bg-panel2 px-2 py-1.5"
            >
              <span className="font-mono font-semibold">{t.name}</span>
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
                  already minted
                </Btn>
              ) : (
                <Btn onClick={() => registerTool(t.name)} disabled={!agOk}>
                  Register (~36s)
                </Btn>
              )}
            </div>
          );
        })}
      </div>
      <Status>{status}</Status>
    </Card>
  );
}
