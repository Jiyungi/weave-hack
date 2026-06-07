"use client";

import { useState } from "react";
import { cp, CapabilityRequest } from "@/lib/api";
import { needsHumanApproval } from "@/lib/capability-approval";
import { useDashboard } from "@/lib/dashboard-context";
import { AutoApproveToggle } from "./AutoApproveToggle";
import { Btn, Card, Pill } from "./ui";

function StatusPill({ status }: { status: CapabilityRequest["status"] }) {
  if (status === "approved") return <Pill variant="good">approved</Pill>;
  if (status === "denied") return <Pill variant="bad">denied</Pill>;
  return <Pill variant="warn">pending</Pill>;
}

export function ApprovalsPanel() {
  const { state, refresh } = useDashboard();
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string>("");

  const autoApprove = state.settings?.auto_approve_enabled ?? true;
  const requests = state.requests ?? [];
  const awaiting = requests.filter((r) => needsHumanApproval(r, autoApprove));
  const processing = autoApprove
    ? requests.filter(
        (r) =>
          r.status === "pending" &&
          !r.sensitive &&
          !r.session_revoke_block,
      )
    : [];
  const decided = requests.filter((r) => r.status !== "pending").slice(0, 6);

  async function decide(r: CapabilityRequest, approve: boolean) {
    setBusy(r.request_id);
    setErr("");
    try {
      if (approve) await cp.approveCapability(r.request_id);
      else await cp.denyCapability(r.request_id);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card
      title="Capability requests"
      badge={
        awaiting.length > 0 ? (
          <Pill variant="warn">{awaiting.length} awaiting you</Pill>
        ) : (
          <Pill variant="muted">
            {autoApprove ? "hybrid approval" : "manual approval"}
          </Pill>
        )
      }
    >
      <div className="mb-3">
        <AutoApproveToggle />
      </div>

      <p className="mb-3 text-[12px] leading-relaxed text-muted">
        When an agent hits a wall, it <b>asks</b> for the skill it needs — it never
        grants itself. Sensitive skills (needs a key / spends money) always wait
        for you when auto-approve is on.
      </p>

      {err && (
        <div className="mb-3 rounded-md border border-bad/30 bg-bad/10 px-3 py-2 text-[12px] text-bad">
          {err}
        </div>
      )}

      {awaiting.length === 0 && processing.length === 0 && decided.length === 0 && (
        <div className="rounded-md border border-line bg-panel2/60 px-3 py-4 text-center text-[12px] text-muted">
          No requests yet. Run an agent on a task it can&apos;t do and watch it ask.
        </div>
      )}

      {processing.map((r) => (
        <div
          key={r.request_id}
          className="mb-2 flex items-center gap-2 rounded-md border border-line bg-panel2/50 px-2.5 py-1.5 text-[11.5px]"
        >
          <Pill variant="muted">auto-approving…</Pill>
          <span className="font-mono text-text">{r.skill}</span>
          <span className="text-muted">→ {r.principal}</span>
          {r.has_examples && (
            <span className="ml-auto text-muted/70">minting ~36s</span>
          )}
        </div>
      ))}

      {awaiting.map((r) => (
        <div
          key={r.request_id}
          className="mb-2.5 rounded-lg border border-warn/30 bg-warn/[0.06] p-3"
        >
          <div className="mb-1 flex items-center gap-2">
            <span className="font-mono text-[13px] font-semibold text-text">
              {r.skill}
            </span>
            {r.sensitive && <Pill variant="warn">sensitive</Pill>}
            {r.session_revoke_block && (
              <Pill variant="warn">session-revoked</Pill>
            )}
            {!autoApprove && !r.sensitive && !r.session_revoke_block && (
              <Pill variant="muted">manual mode</Pill>
            )}
            {r.has_examples && <Pill variant="muted">will mint ~36s</Pill>}
          </div>
          <div className="mb-1 text-[12px] text-muted">
            <span className="text-text">{r.principal}</span> wants this
            {r.reason ? `: ${r.reason}` : ""}
          </div>
          {r.description && (
            <div className="mb-2 text-[11.5px] text-muted/80">{r.description}</div>
          )}
          <div className="flex gap-2">
            <Btn
              variant="primary"
              disabled={busy === r.request_id}
              onClick={() => decide(r, true)}
            >
              {busy === r.request_id ? "…" : "Approve"}
            </Btn>
            <Btn
              variant="danger"
              disabled={busy === r.request_id}
              onClick={() => decide(r, false)}
            >
              Deny
            </Btn>
          </div>
        </div>
      ))}

      {decided.length > 0 && (
        <div className="mt-2 flex flex-col gap-1">
          {decided.map((r) => (
            <div
              key={r.request_id}
              className="flex items-center gap-2 rounded-md border border-line bg-panel2/50 px-2.5 py-1.5 text-[11.5px]"
            >
              <StatusPill status={r.status} />
              <span className="font-mono text-text">{r.skill}</span>
              <span className="text-muted">→ {r.principal}</span>
              {r.decided_by && (
                <span className="ml-auto text-muted/70">by {r.decided_by}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
