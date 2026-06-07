import type { AdapterMeta } from "../contracts/index.js";
import {
  ChatController,
  InferenceApiClient,
  type ChatMessage,
} from "./chat.js";
import {
  buildDashboardViewModel,
  type DashboardViewModel,
  type EvalResults,
} from "./eval-results.js";

/**
 * Track C demo glue for the Integration_Milestone (Req 21.4).
 *
 * Connects Unit selection to the proof visuals: selecting a Unit routes to the
 * correct pre-baked Adapter (via a `unitLabel -> adapterId` route catalog that
 * mirrors what `Redis_Client_API.route` resolves for the active demo day),
 * generates a response through the Inference_API, and builds the dashboard view
 * model from the corresponding `eval_results.json` payload.
 *
 * The route catalog is the active demo day's snapshot — the same mapping the
 * Python `DemoEnvironment` produces when it points the Redis route index at a
 * single time-compressed day. The frontend never trains (training is batch
 * only); it only selects, routes, generates, and displays.
 */
export interface DemoRouteCatalog {
  /** Active demo day index (informational; matches the pre-baked artifacts). */
  dayIndex: number;
  /** unitLabel -> routed adapter_id for the active demo day. */
  routes: Record<string, string>;
}

export interface DemoSelectionResult {
  unitLabel: string;
  adapterId: string;
  message: ChatMessage;
  dashboard: DashboardViewModel;
}

export class DemoController {
  private readonly chat: ChatController;

  constructor(
    inferenceApi: InferenceApiClient,
    private readonly catalog: DemoRouteCatalog,
    private readonly adapters: readonly AdapterMeta[],
    private readonly evalResults: EvalResults,
  ) {
    this.chat = new ChatController(inferenceApi);
  }

  /** Resolve the routed adapter_id for a Unit (mirrors Redis_Client_API.route). */
  route(unitLabel: string): string {
    const adapterId = this.catalog.routes[unitLabel];
    if (adapterId === undefined) {
      throw new Error(
        `No pre-baked adapter routed for unit '${unitLabel}' on day ${this.catalog.dayIndex}`,
      );
    }
    return adapterId;
  }

  /** The proof-visual dashboard built from the corresponding eval_results.json. */
  proofVisuals(): DashboardViewModel {
    return buildDashboardViewModel(this.adapters, this.evalResults);
  }

  /**
   * Unit selection -> route -> generate -> proof visuals (Req 21.4).
   *
   * Routes the selected Unit to its pre-baked Adapter, generates a response
   * through the Inference_API, and returns the generated message alongside the
   * dashboard view model for the corresponding eval_results.json.
   */
  async selectUnit(
    unitLabel: string,
    prompt: string,
    maxNewTokens = 160,
  ): Promise<DemoSelectionResult> {
    const adapterId = this.route(unitLabel);
    this.chat.selectUnit({ unitLabel, adapterId });
    const message = await this.chat.sendMessage(prompt, maxNewTokens);
    return {
      unitLabel,
      adapterId,
      message,
      dashboard: this.proofVisuals(),
    };
  }
}
