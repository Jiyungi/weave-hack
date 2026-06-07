import { useEffect, useMemo, useState } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import { ChatView } from "./components/ChatView.js";
import { DashboardView } from "./components/DashboardView.js";
import { buildUnitOptions, BASE_MODEL_OPTION } from "./units.js";
import type { AdapterMeta } from "./contracts/index.js";
import {
  COPILOTKIT_RUNTIME_URL,
  INFERENCE_API_URL,
  ADAPTER_ID_HEADER,
  UNIT_LABEL_HEADER,
} from "./config.js";
import type { UnitOption } from "./frontend/chat.js";

type Tab = "chat" | "dashboard";

export function App(): JSX.Element {
  // Load the REAL adapters served by the Inference_API. No mock fallback: if
  // the API is unavailable the Unit list is just the Base_Model option.
  const [adapters, setAdapters] = useState<readonly AdapterMeta[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${INFERENCE_API_URL}/adapters/meta`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data: AdapterMeta[]) => {
        if (!cancelled && Array.isArray(data)) {
          setAdapters(data);
        }
      })
      .catch(() => {
        /* API down: leave the list empty (Base_Model only) — no mock data */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const unitOptions = useMemo<UnitOption[]>(
    () => [BASE_MODEL_OPTION, ...buildUnitOptions(adapters)],
    [adapters],
  );
  const [tab, setTab] = useState<Tab>("chat");
  const [selectedUnit, setSelectedUnit] = useState<UnitOption>(BASE_MODEL_OPTION);

  // Once real adapters load, default the selection to the first real Unit.
  useEffect(() => {
    if (selectedUnit.adapterId === null && unitOptions.length > 1) {
      setSelectedUnit(unitOptions[1]);
    }
  }, [unitOptions, selectedUnit]);

  // The selected Unit's adapter_id is forwarded to the CopilotKit runtime on
  // every chat request, so the chosen Unit genuinely influences generation.
  const headers = useMemo<Record<string, string>>(
    () => ({
      [ADAPTER_ID_HEADER]: selectedUnit.adapterId ?? "",
      [UNIT_LABEL_HEADER]: selectedUnit.unitLabel,
    }),
    [selectedUnit],
  );

  return (
    <div className="ws-app">
      <header className="ws-header">
        <h1>WeaveSelf</h1>
        <nav className="ws-tabs">
          <button
            type="button"
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            Chat
          </button>
          <button
            type="button"
            className={tab === "dashboard" ? "active" : ""}
            onClick={() => setTab("dashboard")}
          >
            Dashboard
          </button>
        </nav>
      </header>

      <main className="ws-main">
        {tab === "chat" ? (
          <CopilotKit runtimeUrl={COPILOTKIT_RUNTIME_URL} headers={headers}>
            <ChatView
              unitOptions={unitOptions}
              selectedUnit={selectedUnit}
              onSelectUnit={setSelectedUnit}
            />
          </CopilotKit>
        ) : (
          <DashboardView />
        )}
      </main>
    </div>
  );
}
