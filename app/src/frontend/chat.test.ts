import { describe, expect, it } from "vitest";
import { ChatController, InferenceApiClient, renderChatHtml, type GenerateRequest } from "./chat.js";

describe("CopilotKit chat behavior boundary (Requirement 17.1)", () => {
  it("calls the Inference_API with the selected Unit adapter and displays the generated response", async () => {
    let capturedRequest: GenerateRequest | undefined;
    const fetchImpl = async (_input: string, init?: RequestInit): Promise<Response> => {
      capturedRequest = JSON.parse(String(init?.body)) as GenerateRequest;
      return Response.json({ text: "adapter answer", tokens: 12, latency_ms: 30 });
    };

    const controller = new ChatController(new InferenceApiClient("http://localhost:8000", fetchImpl));
    controller.selectUnit({ unitLabel: "cooking", adapterId: "stackexchange_cooking_v3" });

    const message = await controller.sendMessage("How do I fix soup?");
    expect(capturedRequest).toEqual({
      prompt: "How do I fix soup?",
      adapter_id: "stackexchange_cooking_v3",
      max_new_tokens: 160,
    });
    expect(message).toEqual({ role: "assistant", content: "adapter answer" });
    expect(controller.getMessages()).toEqual([
      { role: "user", content: "How do I fix soup?" },
      { role: "assistant", content: "adapter answer" },
    ]);
    expect(renderChatHtml(controller.getMessages())).toContain('data-view="chat"');
    expect(renderChatHtml(controller.getMessages())).toContain("adapter answer");
  });
});
