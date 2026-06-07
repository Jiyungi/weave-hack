"use client";

import { useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Label, Pill } from "./ui";

export function MemoryPanel() {
  const { state, refresh } = useDashboard();
  const [userId, setUserId] = useState("alice");
  const [userMsg, setUserMsg] = useState("Summarize this PR in bullet points.");
  const [assistantMsg, setAssistantMsg] = useState(
    "TL;DR: three changes.\n- Fixed auth bug\n- Added tests\n- Updated docs",
  );
  const [busy, setBusy] = useState("");
  const [status, setStatus] = useState("");

  const pending = state.memory?.pending ?? {};
  const personalized = state.personalization ?? {};
  const pendingCount = pending[userId] ?? 0;
  const hasStyle = Boolean(personalized[userId]);

  async function logTurn() {
    setBusy("log");
    setStatus("");
    try {
      const r = await cp.logInteraction(userId, userMsg, assistantMsg);
      setStatus(`logged · ${r.pending_interactions} pending for ${userId}`);
      await refresh();
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function consolidate() {
    setBusy("consolidate");
    setStatus("consolidating (~36s mint on Track A)…");
    try {
      const r = await cp.consolidateUser(userId);
      setStatus(
        `minted ${r.controller_id} · ${r.curated_pairs} pairs · deleted ${r.logs_deleted} logs`,
      );
      await refresh();
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  return (
    <Card
      title="Weight memory — personalization"
      badge={
        hasStyle ? (
          <Pill variant="good">user_style minted</Pill>
        ) : (
          <Pill variant="muted">1 adapter per user</Pill>
        )
      }
    >
      <p className="mb-3 text-[12px] leading-relaxed text-muted">
        Log how a user likes answers, then <b>Consolidate</b> to mint{" "}
        <code>user_style-{"{user}"}</code> via Track A. Raw logs are deleted;
        memory lives in the adapter. Compose with tool skills at session open.
      </p>

      <Label>User ID</Label>
      <Input value={userId} onChange={setUserId} placeholder="alice" />

      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
        <span>pending chats: {pendingCount}</span>
        {hasStyle && (
          <span>
            controller: <code className="text-text">{personalized[userId]}</code>
          </span>
        )}
      </div>

      <Label className="mt-3">Example user message</Label>
      <Input value={userMsg} onChange={setUserMsg} />
      <Label className="mt-2">Example assistant reply (their style)</Label>
      <Input value={assistantMsg} onChange={setAssistantMsg} />

      <div className="mt-3 flex flex-wrap gap-2">
        <Btn variant="ghost" disabled={busy === "log"} onClick={logTurn}>
          Log turn
        </Btn>
        <Btn
          variant="primary"
          disabled={busy === "consolidate" || pendingCount === 0}
          onClick={consolidate}
        >
          {busy === "consolidate" ? "Consolidating…" : "Consolidate → mint style"}
        </Btn>
      </div>

      {status && (
        <div className="mt-3 rounded-md border border-line bg-panel2/60 px-3 py-2 text-[12px] text-muted">
          {status}
        </div>
      )}
    </Card>
  );
}
