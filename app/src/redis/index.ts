/**
 * Redis layer barrel — key builders and the client API contract (Track C).
 *
 * The concrete Redis-backed RedisClientApi implementation is added in task 9.x;
 * this module currently exposes only the Track 0 contract surface.
 */
export {
  REDIS_KEY_PREFIXES,
  adapterBlobKey,
  adapterMetaKey,
  adapterIndexKey,
  interactionsKey,
} from "./keys.js";
export type { RedisClientApi } from "./client-api.js";
