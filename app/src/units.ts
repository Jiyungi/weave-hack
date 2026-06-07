import type { AdapterMeta } from "./contracts/index.js";
import type { UnitOption } from "./frontend/chat.js";
import { visibleAdapters } from "./frontend/eval-results.js";

/**
 * Build the Unit selector options from the adapter library.
 *
 * Mirrors the dashboard's adapter-library rule (Req 17.2): zero-size adapters
 * are hidden, so a user can never select a Unit that has no trained adapter.
 * Each option carries the unit_label (shown) and the adapter_id (sent to the
 * Inference_API as `adapter_id`).
 */
export function buildUnitOptions(adapters: readonly AdapterMeta[]): UnitOption[] {
  return visibleAdapters(adapters).map((adapter) => ({
    unitLabel: adapter.unit_label,
    adapterId: adapter.adapter_id,
  }));
}

/** The "no adapter" baseline option (Base_Model only, adapter_id = null). */
export const BASE_MODEL_OPTION: UnitOption = {
  unitLabel: "Base model (no adapter)",
  adapterId: null,
};
