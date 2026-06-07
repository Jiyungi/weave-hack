"use client";

import { useEffect, useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Label, Pill, Pre, Select, Status } from "./ui";

export function ActPanel() {
  const { state, refresh } = useDashboard();
  const [sessionId, setSessionId] = useState("");
  const [styleUserId, setStyleUserId] = useState("alice");
  const [prompt, setPrompt] = useState(
    "User: explain photosynthesis.\nAssistant:",
  );
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<Awaited<ReturnType<typeof cp.act>> | null>(
    null,
  );
  const [styleResult, setStyleResult] = useState<
    Awaited<ReturnType<typeof cp.actStyle>> | null
  >(null);

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
    setStyleResult(null);
    try {
      const r = await cp.act(sessionId, prompt);
      setResult(r);
      setStatus("");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function actStyle() {
    const uid = styleUserId.trim();
    if (!uid) {
      setStatus("user id required for style act");
      return;
    }
    setStatus("style act…");
    setResult(null);
    setStyleResult(null);
    try {
      const r = await cp.actStyle(uid, prompt);
      setStyleResult(r);
      setStatus("");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="Act console">
      <p className="mb-2 text-[11px] leading-relaxed text-muted">
        <b>Session act</b> — composed session controller (tools ± style).{" "}
        <b>Style act</b> — <code>user_style-{"{user}"}</code> solo (HOW only,
        no tool compose).
      </p>
      <Label>Session (tool / compose demo)</Label>
      <Select value={sessionId} onChange={setSessionId} options={sessionOpts} />
      <Label className="mt-2">User ID (style act)</Label>
      <Input value={styleUserId} onChange={setStyleUserId} placeholder="alice" />
      <Label className="mt-2">Prompt</Label>
      <Input value={prompt} onChange={setPrompt} />
      <div className="mt-1 flex flex-wrap gap-1.5">
        <Btn
          variant="ghost"
          onClick={() =>
            setPrompt("User: explain photosynthesis.\nAssistant:")
          }
        >
          style: photosynthesis
        </Btn>
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
      <div className="mt-2 flex flex-wrap gap-2">
        <Btn onClick={act}>Session act</Btn>
        <Btn variant="primary" onClick={actStyle}>
          Style act
        </Btn>
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
      {styleResult && (
        <div className="mt-2">
          <Pill variant="good">user_style</Pill>
          <Pre>{styleResult.completion}</Pre>
          <div className="mt-1 font-mono text-[12px] text-muted">
            {styleResult.controller_id}
          </div>
        </div>
      )}
    </Card>
  );
}
