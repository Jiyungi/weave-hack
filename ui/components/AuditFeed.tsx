"use client";

import { useDashboard } from "@/lib/dashboard-context";
import { Card, Pill } from "./ui";

export function AuditFeed() {
  const { audit, health } = useDashboard();
  const backend = String(health.cp?.audit_backend ?? "memory+file");

  return (
    <Card
      title="Audit feed"
      badge={<Pill variant="muted">{backend}</Pill>}
    >
      <div className="flex max-h-[340px] flex-col gap-1.5 overflow-auto">
        {[...audit].reverse().map((e, i) => {
          const time = new Date((e.ts || 0) * 1000).toLocaleTimeString();
          const { ts, event, ...rest } = e;
          const cls =
            event === "act" || event === "act_gate"
              ? e.permitted === false
                ? "border-l-bad bg-[#1b0f0f]"
                : "border-l-good"
              : event === "revoke" || event === "revoke_policy"
                ? "border-l-bad"
                : event === "open_session" || event === "personalize"
                  ? "border-l-accent"
                  : "border-l-line";
          return (
            <div
              key={`${e.ts}-${i}`}
              className={`rounded-md border border-line border-l-[3px] bg-panel2 px-2 py-1.5 font-mono text-[11.5px] ${cls}`}
            >
              <span className="text-muted">{time}</span>{" "}
              <b>{String(event)}</b> {JSON.stringify(rest)}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
