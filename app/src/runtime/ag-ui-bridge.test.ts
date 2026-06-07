import { describe, expect, it } from "vitest";
import { AgUiBridge, parseSseStream } from "./ag-ui-bridge.js";

function sseResponse(payload: string): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(payload));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("AG-UI bridge (Requirement 18)", () => {
  it("parses AG-UI server-sent events from the LangGraph agent stream", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("event: token\ndata: hello\n\n"));
        controller.enqueue(new TextEncoder().encode("event: token\ndata: world\n\n"));
        controller.close();
      },
    });

    const events = [];
    for await (const event of parseSseStream(body)) {
      events.push(event);
    }

    expect(events).toEqual([
      { event: "token", data: "hello" },
      { event: "token", data: "world" },
    ]);
  });

  it("streams responses and clears connection errors after restore", async () => {
    const bridge = new AgUiBridge("http://localhost:9000/ag-ui", async () =>
      sseResponse("event: message\ndata: restored\n\n"),
    );

    const events = [];
    for await (const event of bridge.stream({ prompt: "status" })) {
      events.push(event);
    }

    expect(events).toEqual([{ event: "message", data: "restored" }]);
    expect(bridge.getState()).toEqual({ connected: true, connectionError: null });
  });

  it("records a connection error when the AG-UI bridge is unavailable", async () => {
    const bridge = new AgUiBridge("http://localhost:9000/ag-ui", async () => new Response(null, { status: 503 }));

    await expect(async () => {
      for await (const _event of bridge.stream({ prompt: "status" })) {
        throw new Error("unexpected event");
      }
    }).rejects.toThrow("AG-UI bridge unavailable: HTTP 503");
    expect(bridge.getState()).toEqual({
      connected: false,
      connectionError: "AG-UI bridge unavailable: HTTP 503",
    });
  });
});
