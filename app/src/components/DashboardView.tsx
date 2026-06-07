import { useEffect, useMemo, useState } from "react";
import { buildDashboardViewModel } from "../frontend/eval-results.js";
import type { EvalResults } from "../frontend/eval-results.js";
import { MOCK_ADAPTERS, MOCK_EVAL_RESULTS } from "../frontend/mock-eval-results.js";
import { EVAL_RESULTS_URL } from "../config.js";

type LoadState = "mock" | "loading" | "loaded" | "error";

/**
 * DASHBOARD VIEW (Requirement 17.2–17.5).
 *
 * Renders, via the existing view-model builders in frontend/eval-results.ts:
 *  - the adapter library, hiding zero-size adapters (Req 17.2)
 *  - the confusion-matrix heatmap (Req 17.3)
 *  - base-vs-adapter example pairs with reference text (Req 17.4)
 *  - the NKT-Mirror vs LoRA size chart (Req 17.5)
 *
 * Data source: a real eval_results.json from VITE_EVAL_RESULTS_URL when set,
 * otherwise the bundled MOCK_EVAL_RESULTS so the dashboard renders before the
 * real eval runs.
 */
export function DashboardView(): JSX.Element {
  const [results, setResults] = useState<EvalResults>(MOCK_EVAL_RESULTS);
  const [state, setState] = useState<LoadState>("loading");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch(EVAL_RESULTS_URL, { headers: { accept: "application/json" } });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = (await response.json()) as EvalResults;
        if (!cancelled) {
          setResults(payload);
          setState("loaded");
        }
      } catch {
        if (!cancelled) {
          setResults(MOCK_EVAL_RESULTS);
          setState("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const viewModel = useMemo(() => buildDashboardViewModel(MOCK_ADAPTERS, results), [results]);
  const maxSize = Math.max(1, ...viewModel.sizeChart.map((item) => item.bytes));

  return (
    <section className="ws-dashboard" data-view="dashboard">
      <p className="ws-source">
        Source:{" "}
        {state === "loaded"
          ? `live eval_results.json (${EVAL_RESULTS_URL})`
          : state === "error"
            ? "mock (failed to load configured eval_results.json)"
            : state === "loading"
              ? "loading…"
              : "bundled mock eval_results.json"}
      </p>

      <div className="ws-grid">
        <article className="ws-card">
          <h2>Adapter library</h2>
          <table data-view="adapter-library">
            <thead>
              <tr>
                <th>adapter_id</th>
                <th>unit_label</th>
                <th>size_bytes</th>
              </tr>
            </thead>
            <tbody>
              {viewModel.adapterLibrary.map((adapter) => (
                <tr key={adapter.adapter_id}>
                  <td>{adapter.adapter_id}</td>
                  <td>{adapter.unit_label}</td>
                  <td>{adapter.size_bytes.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>

        <article className="ws-card">
          <h2>Confusion matrix</h2>
          <table data-view="confusion-matrix" className="ws-heatmap">
            <thead>
              <tr>
                <th />
                {results.confusion_matrix.labels.map((label) => (
                  <th key={`col-${label}`}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {viewModel.heatmap.map((row, rowIndex) => (
                <tr key={`row-${results.confusion_matrix.labels[rowIndex] ?? rowIndex}`}>
                  <th>{results.confusion_matrix.labels[rowIndex] ?? `row_${rowIndex}`}</th>
                  {row.map((cell) => (
                    <td
                      key={`${cell.rowLabel}:${cell.columnLabel}`}
                      data-row={cell.rowLabel}
                      data-column={cell.columnLabel}
                      style={{
                        backgroundColor: `rgba(56, 132, 255, ${cell.intensity.toFixed(3)})`,
                      }}
                    >
                      {cell.value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </article>
      </div>

      <article className="ws-card">
        <h2>Size: NKT-Mirror vs LoRA</h2>
        <ul data-view="size-chart" className="ws-bars">
          {viewModel.sizeChart.map((item) => (
            <li key={item.label}>
              <span className="ws-bar-label">{item.label}</span>
              <span className="ws-bar-track">
                <span
                  className="ws-bar-fill"
                  style={{ width: `${Math.max(2, (item.bytes / maxSize) * 100)}%` }}
                />
              </span>
              <span className="ws-bar-value">{item.bytes.toLocaleString()} bytes</span>
            </li>
          ))}
        </ul>
      </article>

      <article className="ws-card">
        <h2>Base vs adapter examples</h2>
        <section data-view="examples" className="ws-examples">
          {viewModel.examples.map((example, index) => (
            <div className="ws-example" key={`example-${index}`}>
              <h3>{example.prompt}</h3>
              <div className="ws-example-cols">
                <div>
                  <strong>Base</strong>
                  <p>{example.base}</p>
                </div>
                <div>
                  <strong>Adapter</strong>
                  <p>{example.adapter}</p>
                </div>
                <div>
                  <strong>Reference</strong>
                  <p>{example.reference}</p>
                </div>
              </div>
            </div>
          ))}
        </section>
      </article>
    </section>
  );
}
