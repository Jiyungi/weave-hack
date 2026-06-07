import { randomUUID } from "node:crypto";
import type {
  CopilotServiceAdapter,
  CopilotRuntimeChatCompletionRequest,
  CopilotRuntimeChatCompletionResponse,
} from "@copilotkit/runtime";
import { InferenceApiClient, type FetchLike } from "../frontend/chat.js";

export interface WeaveSelfAdapterOptions {
  /** Inference_API base URL (e.g. http://127.0.0.1:8000). */
  inferenceApiUrl: string;
  /** adapter_id of the selected Unit, or null for the Base_Model baseline. */
  adapterId: string | null;
  /** Default generation budget when the client does not forward maxTokens. */
  maxNewTokens?: number;
  /** Injectable fetch (tests); defaults to the global fetch. */
  fetchImpl?: FetchLike;
}

/**
 * A REAL CopilotKit `CopilotServiceAdapter` that routes chat completions to the
 * WeaveSelf Python Inference_API instead of an LLM vendor.
 *
 * For each request it takes the latest user message as the `prompt`, POSTs
 * `{ prompt, adapter_id, max_new_tokens }` to `${inferenceApiUrl}/generate`, and
 * streams the returned text back to the CopilotKit client through the runtime
 * event source. The `adapter_id` is the one selected in the Unit dropdown
 * (Requirement 17.1), so the selected Unit genuinely influences generation.
 */
export class WeaveSelfServiceAdapter implements CopilotServiceAdapter {
  public readonly provider = "weaveself-inference-api";

  private readonly client: InferenceApiClient;
  private readonly adapterId: string | null;
  private readonly maxNewTokens: number;

  constructor(options: WeaveSelfAdapterOptions) {
    this.client = new InferenceApiClient(options.inferenceApiUrl, options.fetchImpl);
    this.adapterId = options.adapterId;
    this.maxNewTokens = options.maxNewTokens ?? 256;
  }

  async process(
    request: CopilotRuntimeChatCompletionRequest,
  ): Promise<CopilotRuntimeChatCompletionResponse> {
    const threadId = request.threadId ?? randomUUID();
    const prompt = extractLatestUserPrompt(request.messages);
    const maxNewTokens = request.forwardedParameters?.maxTokens ?? this.maxNewTokens;

    await request.eventSource.stream(async (eventStream$) => {
      const messageId = randomUUID();
      try {
        const result = await this.client.generate({
          prompt,
          adapter_id: this.adapterId,
          max_new_tokens: maxNewTokens,
        });
        eventStream$.sendTextMessage(messageId, result.text);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        eventStream$.sendTextMessage(
          messageId,
          `WeaveSelf Inference_API request failed: ${detail}`,
        );
      }
      eventStream$.complete();
    });

    return { threadId };
  }
}

/**
 * Extract the prompt from the conversation: the most recent text message, which
 * is the user's newly submitted turn.
 */
function extractLatestUserPrompt(
  messages: CopilotRuntimeChatCompletionRequest["messages"],
): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.isTextMessage()) {
      return message.content;
    }
  }
  return "";
}
