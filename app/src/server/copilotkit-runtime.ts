import type { IncomingMessage, ServerResponse } from "node:http";
import { CopilotRuntime, copilotRuntimeNodeHttpEndpoint } from "@copilotkit/runtime";
import { createClient } from "redis";
import { WeaveSelfServiceAdapter } from "./weaveself-service-adapter.js";
import { RedisBackedClient } from "../redis/redis-client.js";
import type { RedisClientApi } from "../redis/client-api.js";

export const ADAPTER_ID_HEADER = "x-weaveself-adapter-id";
export const UNIT_LABEL_HEADER = "x-weaveself-unit-label";

export interface CopilotKitHandlerOptions {
  /** Inference_API base URL the service adapter proxies /generate to. */
  inferenceApiUrl: string;
  /** Redis URL for persisting interactions (e.g. redis://127.0.0.1:6379). */
  redisUrl?: string;
  /** Path the CopilotKit GraphQL endpoint is served from. */
  endpoint?: string;
  /** Default max_new_tokens when the client forwards none. */
  maxNewTokens?: number;
}

/**
 * Lazily connect a single shared Redis client. Returns a Redis_Client_API or
 * null if Redis is unreachable (interaction logging is best-effort and must
 * never block chat). The connection is reused across requests.
 */
let _redisPromise: Promise<RedisClientApi | null> | null = null;
function getRedis(redisUrl: string | undefined): Promise<RedisClientApi | null> {
  if (!redisUrl) {
    return Promise.resolve(null);
  }
  if (_redisPromise === null) {
    _redisPromise = (async () => {
      try {
        const client = createClient({ url: redisUrl });
        client.on("error", () => {});
        await client.connect();
        return new RedisBackedClient(client as never);
      } catch {
        return null;
      }
    })();
  }
  return _redisPromise;
}

/**
 * Build a Node HTTP handler that hosts a genuine CopilotKit `CopilotRuntime`
 * endpoint backed by the WeaveSelf custom service adapter.
 *
 * The selected Unit's adapter_id and unit_label arrive per-request via the
 * `x-weaveself-adapter-id` / `x-weaveself-unit-label` headers (set by the React
 * `<CopilotKit headers>` prop). A fresh service adapter is built for each
 * request so the chosen Unit routes generation to the correct adapter, and each
 * turn is logged to Redis under `interactions:<unit_label>` for later training.
 */
export function createCopilotKitNodeHandler(
  options: CopilotKitHandlerOptions,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const endpoint = options.endpoint ?? "/api/copilotkit";
  const runtime = new CopilotRuntime();

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const adapterId = readHeader(req, ADAPTER_ID_HEADER);
    const unitLabel = readHeader(req, UNIT_LABEL_HEADER) ?? "";
    const redis = await getRedis(options.redisUrl);
    const serviceAdapter = new WeaveSelfServiceAdapter({
      inferenceApiUrl: options.inferenceApiUrl,
      adapterId: adapterId === "" ? null : adapterId,
      unitLabel,
      redis,
      maxNewTokens: options.maxNewTokens,
    });

    const handler = copilotRuntimeNodeHttpEndpoint({
      runtime,
      serviceAdapter,
      endpoint,
    });

    await handler(req, res);
  };
}

function readHeader(req: IncomingMessage, name: string): string | null {
  const value = req.headers[name];
  if (value === undefined) {
    return null;
  }
  return Array.isArray(value) ? (value[0] ?? null) : value;
}
