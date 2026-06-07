/**
 * Redis layout key builders (Track 0 / Requirement 3).
 *
 * These helpers are the single source of truth for the four Redis keys defined
 * in design.md "Redis Layout (Requirement 3)". Track A (load adapters) and
 * Track B (store adapters) both consume these helpers so every track addresses
 * the same keyspace without owning the Redis implementation.
 *
 *   adapter:blob:<adapter_id>     -> adapter bytes, or disk path        (Req 3.1)
 *   adapter:meta:<adapter_id>     -> adapter metadata JSON              (Req 3.2)
 *   adapter:index                 -> vector index of unit_label embeds  (Req 3.3)
 *   interactions:<unit_label>     -> raw daily interactions for a Unit  (Req 3.4)
 */

/** Key prefixes, exposed for tests and consumers that need to scan/match keys. */
export const REDIS_KEY_PREFIXES = {
  adapterBlob: "adapter:blob:",
  adapterMeta: "adapter:meta:",
  adapterIndex: "adapter:index",
  interactions: "interactions:",
} as const;

function assertNonEmpty(value: string, name: string): void {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
}

/**
 * Key under which an adapter's bytes (or disk path) are stored.
 * Requirement 3.1.
 */
export function adapterBlobKey(adapterId: string): string {
  assertNonEmpty(adapterId, "adapterId");
  return `${REDIS_KEY_PREFIXES.adapterBlob}${adapterId}`;
}

/**
 * Key under which an adapter's metadata JSON is stored.
 * Requirement 3.2.
 */
export function adapterMetaKey(adapterId: string): string {
  assertNonEmpty(adapterId, "adapterId");
  return `${REDIS_KEY_PREFIXES.adapterMeta}${adapterId}`;
}

/**
 * Key for the vector index of `unit_label` embeddings.
 * Requirement 3.3. This key is fixed (not parameterized).
 */
export function adapterIndexKey(): string {
  return REDIS_KEY_PREFIXES.adapterIndex;
}

/**
 * Key under which a Unit's raw daily interactions are stored.
 * Requirement 3.4.
 */
export function interactionsKey(unitLabel: string): string {
  assertNonEmpty(unitLabel, "unitLabel");
  return `${REDIS_KEY_PREFIXES.interactions}${unitLabel}`;
}
