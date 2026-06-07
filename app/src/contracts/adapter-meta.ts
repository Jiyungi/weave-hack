/**
 * Adapter_File metadata contract (Track 0 / Requirement 1).
 *
 * This is the shared metadata schema that Track A produces and Tracks B and C
 * consume. It mirrors the sidecar `adapter_<id>.json` document described in
 * design.md "Data Models → Adapter_File (Requirement 1)".
 *
 * All eight fields are required (Requirement 1.2). A consumer missing any
 * required field must reject the Adapter_File and report the missing field name
 * (Requirement 1.4).
 */

/** The personalization granularity for one adapter (Requirement 1.2). */
export type UnitType = "category" | "user";

/**
 * The eight-field Adapter_File metadata schema.
 *
 * Field set and types are fixed by Requirement 1.2:
 * adapter_id, base_model, unit_type, unit_label, train_rows, trained_at,
 * day_index, size_bytes.
 */
export interface AdapterMeta {
  /** Unique adapter identifier; also the `<id>` in the Redis keys and file names. */
  adapter_id: string;
  /** Identifier of the frozen instruct Base_Model the adapter applies on top of. */
  base_model: string;
  /** Whether the Unit is a "category" or a "user". */
  unit_type: UnitType;
  /** Human-readable label of the Unit this adapter personalizes. */
  unit_label: string;
  /** Number of training rows consumed when the adapter was trained. */
  train_rows: number;
  /** ISO 8601 timestamp string of when training completed. */
  trained_at: string;
  /** Time-compression index for the demo (which "night" produced this adapter). */
  day_index: number;
  /** Serialized size of the Adapter_File in bytes. */
  size_bytes: number;
}

/**
 * The ordered list of required Adapter_File metadata fields.
 *
 * Exposed so consuming components (validators, the Redis layer, the dashboard)
 * can enforce Requirement 1.4 against a single source of truth.
 */
export const ADAPTER_META_REQUIRED_FIELDS = [
  "adapter_id",
  "base_model",
  "unit_type",
  "unit_label",
  "train_rows",
  "trained_at",
  "day_index",
  "size_bytes",
] as const satisfies ReadonlyArray<keyof AdapterMeta>;

export type AdapterMetaField = (typeof ADAPTER_META_REQUIRED_FIELDS)[number];
