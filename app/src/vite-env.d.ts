/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Browser-exposed Inference_API base URL (WeaveSelf Track C). */
  readonly VITE_INFERENCE_API_URL?: string;
  /** Optional URL/path the dashboard loads a real eval_results.json from. */
  readonly VITE_EVAL_RESULTS_URL?: string;
  /** CopilotKit runtime endpoint the React app posts chat to. */
  readonly VITE_COPILOTKIT_RUNTIME_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
