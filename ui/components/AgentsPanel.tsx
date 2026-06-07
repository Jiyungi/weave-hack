"use client";

import { useState } from "react";
import { ag, OrchestratorResult } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Pill, Status } from "./ui";
import { DelegationCard } from "./DelegationTree";

export function AgentsPanel() {
  const { health, refresh } = useDashboard();
  const [task, setTask] = useState(
    "What's the weather in Berlin? Also tell me about Alan Turing.",
  );
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<OrchestratorResult | null>(null);
  const [busy, setBusy] = useState(false);

  const agOk = !health.agError;

  async function run() {
    if (!task.trim()) {
      setStatus("type a task");
      return;
    }
    setBusy(true);
    setResult(null);
    setStatus("running orchestrator…");
    try {
      const r = await ag.run(task.trim());
      setResult(r);
      setStatus(
        `done · ${r.stopped_reason} · ${r.delegations.length} delegation(s)`,
      );
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="6 · Agents — orchestrator + governed workers"
      badge={
        agOk ? (
          <Pill variant="good">connected</Pill>
        ) : (
          <Pill variant="bad">disconnected</Pill>
        )
      }
      className="col-span-2"
    >
      <p className="mb-2 text-[11.5px] text-muted">
        Task for the orchestrator (delegates to{" "}
        <code>exec-assistant</code> &amp; <code>support-bot</code>)
      </p>
      <Input value={task} onChange={setTask} />
      <div className="mt-1 flex flex-wrap gap-1.5">
        <Btn variant="ghost" onClick={() => setTask("What's the weather in Berlin?")}>
          weather only
        </Btn>
        <Btn variant="ghost" onClick={() => setTask("Search the web for Alan Turing.")}>
          search only
        </Btn>
        <Btn
          variant="ghost"
          onClick={() =>
            setTask(
              "What's the weather in Berlin? Also tell me about Alan Turing.",
            )
          }
        >
          both
        </Btn>
        <Btn variant="ghost" onClick={() => setTask("What is (3+5)*2 ?")}>
          calculator
        </Btn>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <Btn onClick={run} disabled={busy || !agOk}>
          Run orchestrator
        </Btn>
        <Status>{status}</Status>
      </div>
      {result && (
        <div className="mt-3">
          {result.delegations.map((d, i) => (
            <DelegationCard key={i} d={d} />
          ))}
          {result.final_answer ? (
            <div className="mt-3 rounded-lg border border-line bg-panel2 p-3">
              <b>orchestrator FINAL</b>
              <div className="mt-1">{result.final_answer}</div>
            </div>
          ) : (
            <div className="mt-2 text-[11.5px] text-muted">
              no FINAL emitted (stopped: {result.stopped_reason})
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
