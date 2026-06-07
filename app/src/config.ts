/**
 * Browser-side configuration for the WeaveSelf Track C React app.
 *
 * Reads Vite-exposed (VITE_-prefixed) env vars with safe, non-secret defaults.
 * Server-side secrets (OPENAI_API_KEY, etc.) are never read here.
 */

/** Inference_API base URL the CopilotKit runtime proxies /generate to. */
export const INFERENCE_API_URL: string =
  import.meta.env.VITE_INFERENCE_API_URL ?? "http://127.0.0.1:8000";

/** CopilotKit runtime endpoint mounted by the dev-server middleware / Node server. */
export const COPILOTKIT_RUNTIME_URL: string =
  import.meta.env.VITE_COPILOTKIT_RUNTIME_URL ?? "/api/copilotkit";

/**
 * Location of a real eval_results.json the dashboard fetches. Defaults to
 * '/eval_results.json' (served from app/public, a copy of data/eval_results.json).
 * If the fetch fails the dashboard falls back to the bundled MOCK_EVAL_RESULTS so
 * it always renders.
 */
export const EVAL_RESULTS_URL: string =
  import.meta.env.VITE_EVAL_RESULTS_URL && import.meta.env.VITE_EVAL_RESULTS_URL.trim() !== ""
    ? import.meta.env.VITE_EVAL_RESULTS_URL
    : "/eval_results.json";

/** Header name the client uses to forward the selected Unit's adapter_id to the runtime. */
export const ADAPTER_ID_HEADER = "x-weaveself-adapter-id";

/** Header name the client uses to forward the selected Unit's label (informational). */
export const UNIT_LABEL_HEADER = "x-weaveself-unit-label";
