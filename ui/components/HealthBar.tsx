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

  const chips: { k: string; v: string }[] = [
    { k: "track_a", v: String(cp?.track_a_url ?? "?") },
    { k: "state", v: String(cp?.state_backend ?? "memory") },
    { k: "audit", v: String(cp?.audit_backend ?? "?") },
    { k: "weave", v: cp?.weave_tracing ? "on" : "off" },
    { k: "skills", v: String(Object.keys(state.skills).length) },
    { k: "sessions", v: String(Object.keys(state.sessions).length) },
  ];

  const agOk = !health.agError;
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {chips.map((c) => (
        <span
          key={c.k}
          className="inline-flex items-center gap-1 rounded-md border border-line bg-panel2/70 px-1.5 py-0.5 font-mono text-[11px]"
        >
          <span className="text-muted">{c.k}</span>
          <span className="text-text/90">{c.v}</span>
        </span>
      ))}
      {agOk ? (
        <Pill variant="good">agents up</Pill>
      ) : (
        <Pill variant="bad">agents down</Pill>
      )}
    </span>
  );
}
