/**
 * Track 0 shared contracts barrel.
 *
 * Re-exports the cross-track TypeScript contracts so consumers can import from
 * a single path (e.g. `import { AdapterMeta } from "../contracts"`).
 */
export type { AdapterMeta, UnitType, AdapterMetaField } from "./adapter-meta.js";
export { ADAPTER_META_REQUIRED_FIELDS } from "./adapter-meta.js";
