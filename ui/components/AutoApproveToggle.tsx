"use client";

import { useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Pill } from "./ui";

/** Global hybrid-approval switch — visible in header + capability requests. */
export function AutoApproveToggle({ compact = false }: { compact?: boolean }) {
  const { state, refresh } = useDashboard();
  const [busy, setBusy] = useState(false);
  const enabled = state.settings?.auto_approve_enabled ?? true;

  async function toggle() {
    setBusy(true);
    try {
      await cp.setAutoApprove(!enabled);
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  if (compact) {
    return (
      <label
        title={
          enabled
            ? "Safe skills auto-grant on REQUEST (click to require manual approval)"
            : "All capability requests need your approval (click to enable auto-approve)"
        }
        className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-line bg-panel2/70 px-2 py-0.5 font-mono text-[11px]"
      >
        <span className="text-muted">auto-approve</span>
        <input
          type="checkbox"
          className="accent-accent"
          checked={enabled}
          disabled={busy}
          onChange={() => void toggle()}
        />
        <span className={enabled ? "text-good" : "text-warn"}>
          {enabled ? "on" : "off"}
        </span>
      </label>
    );
  }

  return (
    <div className="flex items-start justify-between gap-3 rounded-md border border-line bg-panel2/50 px-3 py-2.5">
      <div>
        <div className="flex items-center gap-2 text-[12px] font-medium text-text">
          Auto-approve safe skills
          <Pill variant={enabled ? "good" : "warn"}>{enabled ? "on" : "off"}</Pill>
        </div>
        <p className="mt-0.5 text-[11.5px] leading-relaxed text-muted">
          When on, read-only skills grant instantly on REQUEST. When off, every
          request waits for you — including re-grants after a session revoke.
        </p>
      </div>
      <label className="flex shrink-0 cursor-pointer items-center gap-2 text-[12px] text-muted">
        <input
          type="checkbox"
          className="accent-accent"
          checked={enabled}
          disabled={busy}
          onChange={() => void toggle()}
        />
        {enabled ? "Enabled" : "Disabled"}
      </label>
    </div>
  );
}
