import { describe, expect, it } from "vitest";
import { InferenceApiClient, type GenerateRequest } from "./chat.js";
import { DemoController, type DemoRouteCatalog } from "./demo-wiring.js";
import { MOCK_ADAPTERS, MOCK_EVAL_RESULTS } from "./mock-eval-results.js";

function makeFetch(captured: { request?: GenerateRequest }) {
  return async (_input: string, init?: RequestInit): Promise<Response> => {
    captured.request = JSON.parse(String(init?.body)) as GenerateRequest;
    return Response.json({
      text: `adapter answer for ${captured.request.adapter_id}`,
      tokens: 9,
      latency_ms: 21,
    });
  };
}

const CATALOG: DemoRouteCatalog = {
  dayIndex: 3,
  routes: {
    cooking: "stackexchange_cooking_v3",
    diy: "stackexchange_diy_v3",
  },
};

describe("Demo Unit-selection routing to proof visuals (Requirement 21.4)", () => {
  it("routes a selected Unit to its pre-baked adapter via the route catalog", () => {
    const controller = new DemoController(
      new InferenceApiClient("http://localhost:8000", makeFetch({})),
      CATALOG,
      MOCK_ADAPTERS,
      MOCK_EVAL_RESULTS,
    );
    expect(controller.route("cooking")).toBe("stackexchange_cooking_v3");
    expect(controller.route("diy")).toBe("stackexchange_diy_v3");
    expect(() => controller.route("unknown")).toThrow(/No pre-baked adapter/);
  });

  it("selects a Unit, generates through the Inference_API, and yields proof visuals", async () => {
    const captured: { request?: GenerateRequest } = {};
    const controller = new DemoController(
      new InferenceApiClient("http://localhost:8000", makeFetch(captured)),
      CATALOG,
      MOCK_ADAPTERS,
      MOCK_EVAL_RESULTS,
    );

    const result = await controller.selectUnit("cooking", "How do I fix soup?");

    // Routed to the correct adapter and generated through the Inference_API.
    expect(captured.request).toEqual({
      prompt: "How do I fix soup?",
      adapter_id: "stackexchange_cooking_v3",
      max_new_tokens: 160,
    });
    expect(result.adapterId).toBe("stackexchange_cooking_v3");
    expect(result.message.role).toBe("assistant");
    expect(result.message.content).toContain("stackexchange_cooking_v3");

    // Surfaced the corresponding eval_results.json proof visuals.
    expect(result.dashboard.heatmap.length).toBe(
      MOCK_EVAL_RESULTS.confusion_matrix.labels.length,
    );
    // Zero-size adapters are hidden in the library (Req 17.2).
    expect(result.dashboard.adapterLibrary.every((a) => a.size_bytes > 0)).toBe(true);
    expect(result.dashboard.sizeChart).toEqual([
      { label: "NKT-Mirror", bytes: MOCK_EVAL_RESULTS.size_bytes.nktmirror },
      { label: "LoRA", bytes: MOCK_EVAL_RESULTS.size_bytes.lora },
    ]);
  });
});
