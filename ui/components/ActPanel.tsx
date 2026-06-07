"use client";

import { useEffect, useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Label, Pill, Pre, Select, Status } from "./ui";

export function ActPanel() {
  const { state, refresh } = useDashboard();
  const [sessionId, setSessionId] = useState("");
  const [prompt, setPrompt] = useState(
    "User: any events on 2026-05-05?\nAssistant:",
  );
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<Awaited<ReturnType<typeof cp.act>> | null>(
    null,
  );

  const sessionOpts = Object.entries(state.sessions).map(([sid, v]) => ({
    value: sid,
    label: `${sid} (${v.principal}) auth=[${v.authorized.join(",")}]`,
  }));

  useEffect(() => {
    if (!sessionId && sessionOpts.length) setSessionId(sessionOpts[0].value);
  }, [sessionId, sessionOpts.map((o) => o.value).join(",")]);

  async function act() {
    if (!sessionId) {
      setStatus("open a session first");
      return;
    }
    setStatus("running…");
    setResult(null);
    try {
      const r = await cp.act(sessionId, prompt);
      setResult(r);
      setStatus("");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="Act console">
      <Label>Session</Label>
      <Select value={sessionId} onChange={setSessionId} options={sessionOpts} />
      <Label>Prompt</Label>
      <Input value={prompt} onChange={setPrompt} />
      <div className="mt-1 flex flex-wrap gap-1.5">
        <Btn
          variant="ghost"
          onClick={() =>
            setPrompt("User: what's the weather in Berlin?\nAssistant:")
          }
        >
          weather: Berlin
        </Btn>
        <Btn
          variant="ghost"
          onClick={() =>
            setPrompt("User: any events on 2026-05-05?\nAssistant:")
          }
        >
          calendar: 2026-05-05
        </Btn>
      </div>
      <div className="mt-2">
        <Btn onClick={act}>Send</Btn>
      </div>
      <Status>{status}</Status>
      {result && (
        <div className="mt-2">
          {result.permitted ? (
            <Pill variant="good">permitted</Pill>
          ) : (
            <Pill variant="bad">BLOCKED by runtime guard</Pill>
          )}
          <Pre>{result.completion}</Pre>
          <div className="mt-1 font-mono text-[12px]">
            tool_calls: {result.tool_calls.join(", ") || "—"} · allowed:{" "}
            <span className="text-good">{result.allowed_calls.join(", ") || "—"}</span>{" "}
            · blocked:{" "}
            <span className="text-bad">{result.blocked_calls.join(", ") || "—"}</span>
          </div>
        </div>
      )}
    </Card>
  );
}
