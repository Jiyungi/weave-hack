import type { AdapterMeta } from "../contracts/index.js";

export interface EvalResults {
  perplexity: {
    base: number;
    adapter: number;
    context_memory: number;
  };
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  size_bytes: {
    nktmirror: number;
    lora: number;
  };
  examples: Array<{
    prompt: string;
    base: string;
    adapter: string;
    reference: string;
  }>;
}

export interface HeatmapCell {
  rowLabel: string;
  columnLabel: string;
  value: number;
  intensity: number;
}

export interface DashboardViewModel {
  adapterLibrary: AdapterMeta[];
  heatmap: HeatmapCell[][];
  examples: EvalResults["examples"];
  sizeChart: Array<{ label: string; bytes: number }>;
  perplexity: EvalResults["perplexity"];
}

export function visibleAdapters(adapters: readonly AdapterMeta[]): AdapterMeta[] {
  return adapters
    .filter((adapter) => adapter.size_bytes > 0)
    .map((adapter) => ({ ...adapter }))
    .sort((left, right) => left.adapter_id.localeCompare(right.adapter_id));
}

export function buildHeatmap(results: EvalResults): HeatmapCell[][] {
  const labels = results.confusion_matrix.labels;
  const maxValue = Math.max(1, ...results.confusion_matrix.matrix.flat());

  return results.confusion_matrix.matrix.map((row, rowIndex) =>
    row.map((value, columnIndex) => ({
      rowLabel: labels[rowIndex] ?? `row_${rowIndex}`,
      columnLabel: labels[columnIndex] ?? `column_${columnIndex}`,
      value,
      intensity: value / maxValue,
    })),
  );
}

export function buildDashboardViewModel(
  adapters: readonly AdapterMeta[],
  results: EvalResults,
): DashboardViewModel {
  return {
    adapterLibrary: visibleAdapters(adapters),
    heatmap: buildHeatmap(results),
    examples: results.examples.map((example) => ({ ...example })),
    sizeChart: [
      { label: "NKT-Mirror", bytes: results.size_bytes.nktmirror },
      { label: "LoRA", bytes: results.size_bytes.lora },
    ],
    perplexity: { ...results.perplexity },
  };
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function renderDashboardHtml(viewModel: DashboardViewModel): string {
  const adapterRows = viewModel.adapterLibrary
    .map(
      (adapter) =>
        `<tr><td>${escapeHtml(adapter.adapter_id)}</td><td>${escapeHtml(
          adapter.unit_label,
        )}</td><td>${adapter.size_bytes}</td></tr>`,
    )
    .join("");

  const heatmapRows = viewModel.heatmap
    .map(
      (row) =>
        `<tr>${row
          .map(
            (cell) =>
              `<td data-row="${escapeHtml(cell.rowLabel)}" data-column="${escapeHtml(
                cell.columnLabel,
              )}" style="opacity:${cell.intensity.toFixed(3)}">${cell.value}</td>`,
          )
          .join("")}</tr>`,
    )
    .join("");

  const examples = viewModel.examples
    .map(
      (example) =>
        `<article><h3>${escapeHtml(example.prompt)}</h3><p>${escapeHtml(
          example.base,
        )}</p><p>${escapeHtml(example.adapter)}</p><p>${escapeHtml(example.reference)}</p></article>`,
    )
    .join("");

  const sizes = viewModel.sizeChart
    .map((item) => `<li>${escapeHtml(item.label)}: ${item.bytes}</li>`)
    .join("");

  return `<section data-view="dashboard"><table data-view="adapter-library">${adapterRows}</table><table data-view="confusion-matrix">${heatmapRows}</table><section data-view="examples">${examples}</section><ul data-view="size-chart">${sizes}</ul></section>`;
}
