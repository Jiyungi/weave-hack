export interface GenerateRequest {
  prompt: string;
  adapter_id: string | null;
  max_new_tokens: number;
}

export interface GenerateResponse {
  text: string;
  tokens: number;
  latency_ms: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface UnitOption {
  unitLabel: string;
  adapterId: string | null;
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export class InferenceApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly fetchImpl: FetchLike = fetch,
  ) {}

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await this.fetchImpl(`${this.baseUrl}/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Inference_API /generate failed with HTTP ${response.status}`);
    }

    return (await response.json()) as GenerateResponse;
  }
}

export class ChatController {
  private selectedUnit: UnitOption | undefined;
  private readonly messages: ChatMessage[] = [];

  constructor(private readonly inferenceApi: InferenceApiClient) {}

  selectUnit(unit: UnitOption): void {
    this.selectedUnit = { ...unit };
  }

  getMessages(): ChatMessage[] {
    return this.messages.map((message) => ({ ...message }));
  }

  async sendMessage(content: string, maxNewTokens = 160): Promise<ChatMessage> {
    if (this.selectedUnit === undefined) {
      throw new Error("A Unit must be selected before sending chat messages");
    }

    this.messages.push({ role: "user", content });
    const response = await this.inferenceApi.generate({
      prompt: content,
      adapter_id: this.selectedUnit.adapterId,
      max_new_tokens: maxNewTokens,
    });

    const assistantMessage: ChatMessage = { role: "assistant", content: response.text };
    this.messages.push(assistantMessage);
    return { ...assistantMessage };
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function renderChatHtml(messages: readonly ChatMessage[]): string {
  const rows = messages
    .map(
      (message) =>
        `<div data-role="${message.role}"><strong>${message.role}</strong><p>${escapeHtml(
          message.content,
        )}</p></div>`,
    )
    .join("");

  return `<section data-view="chat">${rows}</section>`;
}
