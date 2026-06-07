import { describe, expect, it } from "vitest";
import fc from "fast-check";
import type { AdapterMeta } from "../contracts/index.js";
import {
  buildDashboardViewModel,
  renderDashboardHtml,
  visibleAdapters,
  type EvalResults,
} from "./eval-results.js";

function meta(adapterId: string, sizeBytes: number): AdapterMeta {
  return {
    adapter_id: adapterId,
    base_model: "Qwen/Qwen2.5-1.5B-Instruct",
    unit_type: "category",
    unit_label: adapterId,
    train_rows: 1,
    trained_at: "2026-06-06T21:00:00Z",
    day_index: 1,
    size_bytes: sizeBytes,
  };
}

// Inline fixture (NOT a shipped mock): used only to exercise the pure
// view-model builders in this unit test.
const FIXTURE_EVAL: EvalResults = {
  perplexity: { base: 97.2, adapter: 68.9, context_memory: 56.5 },
  confusion_matrix: {
    labels: ["cooking", "fitness", "finance"],
    matrix: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  },
  size_bytes: { nktmirror: 37392, lora: 18464768 },
  examples: [
    { prompt: "How do I improve my sauce?", base: "b", adapter: "a", reference: "r" },
  ],
};

describe("Dashboard view model (Requirement 17)", () => {
  it("Feature: weaveself, Property 29: Adapter library filters zero-size adapters", () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 0, max: 200000 }), { minLength: 0, maxLength: 50 }), (sizes) => {
        const adapters = sizes.map((sizeBytes, index) => meta(`adapter_${index}`, sizeBytes));
        expect(visibleAdapters(adapters).map((adapter) => adapter.size_bytes)).toEqual(
          sizes.filter((sizeBytes) => sizeBytes > 0),
        );
      }),
      { numRuns: 100 },
    );
  });

  it("builds and renders the dashboard from an eval_results.json payload", () => {
    const adapters = [meta("cooking-d0", 37392), meta("fitness-d0", 37392), meta("empty_pending", 0)];
    const viewModel = buildDashboardViewModel(adapters, FIXTURE_EVAL);
    const html = renderDashboardHtml(viewModel);

    expect(viewModel.adapterLibrary.map((adapter) => adapter.adapter_id)).not.toContain("empty_pending");
    expect(viewModel.heatmap).toHaveLength(FIXTURE_EVAL.confusion_matrix.labels.length);
    expect(viewModel.examples).toHaveLength(FIXTURE_EVAL.examples.length);
    expect(viewModel.sizeChart.map((item) => item.label)).toEqual(["NKT-Mirror", "LoRA"]);
    expect(html).toContain('data-view="adapter-library"');
    expect(html).toContain('data-view="confusion-matrix"');
    expect(html).toContain('data-view="examples"');
    expect(html).toContain('data-view="size-chart"');
  });
});
