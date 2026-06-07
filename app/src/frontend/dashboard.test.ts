import { describe, expect, it } from "vitest";
import fc from "fast-check";
import type { AdapterMeta } from "../contracts/index.js";
import {
  MOCK_ADAPTERS,
  MOCK_EVAL_RESULTS,
} from "./mock-eval-results.js";
import {
  buildDashboardViewModel,
  renderDashboardHtml,
  visibleAdapters,
} from "./eval-results.js";

function meta(adapterId: string, sizeBytes: number): AdapterMeta {
  return {
    adapter_id: adapterId,
    base_model: "Qwen/Qwen2.5-7B-Instruct",
    unit_type: "category",
    unit_label: adapterId,
    train_rows: 1,
    trained_at: "2026-06-06T21:00:00Z",
    day_index: 1,
    size_bytes: sizeBytes,
  };
}

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

  it("renders the Track C standalone fallback dashboard from mock eval_results.json", () => {
    const viewModel = buildDashboardViewModel(MOCK_ADAPTERS, MOCK_EVAL_RESULTS);
    const html = renderDashboardHtml(viewModel);

    expect(viewModel.adapterLibrary.map((adapter) => adapter.adapter_id)).not.toContain("empty_pending_adapter");
    expect(viewModel.heatmap).toHaveLength(MOCK_EVAL_RESULTS.confusion_matrix.labels.length);
    expect(viewModel.examples).toHaveLength(MOCK_EVAL_RESULTS.examples.length);
    expect(viewModel.sizeChart.map((item) => item.label)).toEqual(["NKT-Mirror", "LoRA"]);
    expect(html).toContain('data-view="adapter-library"');
    expect(html).toContain('data-view="confusion-matrix"');
    expect(html).toContain('data-view="examples"');
    expect(html).toContain('data-view="size-chart"');
  });
});
