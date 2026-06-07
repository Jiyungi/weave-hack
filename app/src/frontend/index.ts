export { ChatController, InferenceApiClient, renderChatHtml } from "./chat.js";
export type { ChatMessage, GenerateRequest, GenerateResponse, UnitOption } from "./chat.js";
export {
  buildDashboardViewModel,
  buildHeatmap,
  renderDashboardHtml,
  visibleAdapters,
} from "./eval-results.js";
export type { DashboardViewModel, EvalResults, HeatmapCell } from "./eval-results.js";
