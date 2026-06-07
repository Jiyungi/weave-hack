import type { CapabilityRequest } from "./api";

/** Whether a pending capability request needs a human decision in the UI. */
export function needsHumanApproval(
  r: CapabilityRequest,
  autoApproveEnabled: boolean,
): boolean {
  if (r.status !== "pending") return false;
  if (r.session_revoke_block) return true;
  if (!autoApproveEnabled) return true;
  return r.sensitive;
}
