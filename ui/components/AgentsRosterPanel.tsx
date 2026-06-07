"use client";

import { useDashboard } from "@/lib/dashboard-context";
import { Card, Pill } from "./ui";

export function AgentsRosterPanel() {
  const { agents, health } = useDashboard();
  const agOk = !health.agError;

  return (
    <Card
      title="Governed workers"
      badge={
        agents?.workers?.length ? (
          <Pill variant="good">{agents.workers.length} roles</Pill>
        ) : (
          <Pill variant="muted">roster</Pill>
        )
      }
    >
      <p className="mb-3 text-[11.5px] text-muted">
        Each worker is a control-plane principal with its own policy. The
        orchestrator delegates sub-tasks; the control plane enforces grants at the
        weight level.
      </p>
      {!agOk && (
        <p className="text-[12px] text-bad">agent service (:8200) unreachable</p>
      )}
      {agents?.workers?.length ? (
        <div className="flex flex-col gap-2">
          {agents.workers.map((w) => (
            <div
              key={w.name}
              className="rounded-lg border border-line bg-panel2/70 px-3 py-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-[13px] font-semibold text-accent">
                  {w.name}
                </span>
                <Pill variant="muted">
                  policy=[{(w.policy ?? []).join(", ") || "none"}]
                </Pill>
              </div>
              <p className="mt-1 text-[11.5px] text-muted">{w.description}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[11.5px] text-muted">
          No roster yet — seed capabilities or start agent_service on :8200.
        </p>
      )}
    </Card>
  );
}
