export { ChatController, InferenceApiClient, renderChatHtml } from "./chat.js";
export type { ChatMessage, GenerateRequest, GenerateResponse, UnitOption } from "./chat.js";
export {
  buildDashboardViewModel,
  buildHeatmap,
  renderDashboardHtml,
  visibleAdapters,
} from "./eval-results.js";
export type { DashboardViewModel, EvalResults, HeatmapCell } from "./eval-results.js";
export { DemoController } from "./demo-wiring.js";
export type { DemoRouteCatalog, DemoSelectionResult } from "./demo-wiring.js";
export { MOCK_ADAPTERS, MOCK_EVAL_RESULTS } from "./mock-eval-results.js";
