/**
 * Redis_Client_API contract (Track 0 / Requirement 3, design "Redis_Client_API").
 *
 * This is the clean client interface over the Redis_Layer used by Track A
 * (load adapters) and Track B (store adapters). It is the contract/interface
 * definition only — the concrete Redis-backed implementation lives in task 9.x.
 *
 * Behavior contract (enforced by the implementation, documented here for consumers):
 *  - storeAdapter persists metadata under `adapter:meta:<id>` and the blob under
 *    `adapter:blob:<id>` (Requirements 3.1, 3.2, 19.1).
 *  - fetchMeta returns stored metadata independently of the blob (Requirement 19.2).
 *  - fetchBlob round-trips bytes identically, including a 100 KB blob
 *    (Requirements 19.4, 20.2).
 *  - route returns the top-1 `adapter_id` by vector search over the unit-label
 *    embedding index (Requirements 3.5, 19.3).
 *  - appendInteraction appends a raw interaction under `interactions:<unit_label>`
 *    (Requirements 3.4, 19.5).
 */
import type { AdapterMeta } from "../contracts/index.js";

export interface RedisClientApi {
  /** Persist an Adapter_File's metadata and blob bytes under the Requirement 3 keys. */
  storeAdapter(meta: AdapterMeta, blob: Uint8Array): Promise<void>;

  /** Return the stored metadata for an adapter, without requiring blob retrieval. */
  fetchMeta(adapterId: string): Promise<AdapterMeta>;

  /** Return the stored blob bytes for an adapter, byte-identical to what was stored. */
  fetchBlob(adapterId: string): Promise<Uint8Array>;

  /** Return the top-1 `adapter_id` for a query or user identifier via vector search. */
  route(queryOrUser: string): Promise<string>;

  /** Append a raw interaction for a Unit under `interactions:<unit_label>`. */
  appendInteraction(unitLabel: string, interaction: object): Promise<void>;
}
