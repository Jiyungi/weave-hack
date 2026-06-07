import type { IncomingMessage, ServerResponse } from "node:http";
import { CopilotRuntime, copilotRuntimeNodeHttpEndpoint } from "@copilotkit/runtime";
import { WeaveSelfServiceAdapter } from "./weaveself-service-adapter.js";

export const ADAPTER_ID_HEADER = "x-weaveself-adapter-id";

export interface CopilotKitHandlerOptions {
  /** Inference_API base URL the service adapter proxies /generate to. */
  inferenceApiUrl: string;
  /** Path the CopilotKit GraphQL endpoint is served from. */
  endpoint?: string;
  /** Default max_new_tokens when the client forwards none. */
  maxNewTokens?: number;
}

/**
 * Build a Node HTTP handler that hosts a genuine CopilotKit `CopilotRuntime`
 * endpoint backed by the WeaveSelf custom service adapter.
 *
 * The selected Unit's adapter_id arrives per-request via the
 * `x-weaveself-adapter-id` header (set by the React `<CopilotKit headers>` prop).
 * A fresh service adapter is built for each request so the chosen Unit routes
 * generation to the correct pre-baked adapter on the Inference_API.
 */
export function createCopilotKitNodeHandler(
  options: CopilotKitHandlerOptions,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const endpoint = options.endpoint ?? "/api/copilotkit";
  const runtime = new CopilotRuntime();

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const adapterId = readHeader(req, ADAPTER_ID_HEADER);
    const serviceAdapter = new WeaveSelfServiceAdapter({
      inferenceApiUrl: options.inferenceApiUrl,
      adapterId: adapterId === "" ? null : adapterId,
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
