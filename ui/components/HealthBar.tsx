"use client";

import { useDashboard } from "@/lib/dashboard-context";
import { Pill } from "./ui";

export function HealthBar() {
  const { health, state } = useDashboard();
  const cp = health.cp;
  const ag = health.ag;

  if (health.cpError) {
    return (
      <span className="font-mono text-[12px] text-bad">
        control plane unreachable
      </span>
    );
  }

  const parts = [
    `track_a ${String(cp?.track_a_url ?? "?")}`,
    `state ${String(cp?.state_backend ?? "memory")}`,
    `audit ${String(cp?.audit_backend ?? "?")}`,
    `weave ${cp?.weave_tracing ? "on" : "off"}`,
    `skills ${Object.keys(state.skills).length}`,
    `sessions ${Object.keys(state.sessions).length}`,
  ];

  const agOk = !health.agError;
  return (
    <span className="flex items-center gap-2 font-mono text-[12px] text-muted">
      {parts.join(" · ")}
      {agOk ? (
        <Pill variant="good">agents up</Pill>
      ) : (
        <Pill variant="bad">agents down</Pill>
      )}
    </span>
  );
}
