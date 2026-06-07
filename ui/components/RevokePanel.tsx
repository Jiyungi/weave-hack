"use client";

import { useEffect, useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Label, Select, Status } from "./ui";

export function RevokePanel() {
  const { state, refresh } = useDashboard();
  const [sessionId, setSessionId] = useState("");
  const [skill, setSkill] = useState("");
  const [status, setStatus] = useState("");

  const sessions = Object.entries(state.sessions);
  const sessionOpts = sessions.map(([sid, v]) => ({
    value: sid,
    label: `${sid} (${v.principal}) auth=[${v.authorized.join(",")}]`,
  }));

  const current = state.sessions[sessionId];
  const skillOpts = (current?.authorized ?? []).map((s) => ({
    value: s,
    label: s,
  }));

  useEffect(() => {
    if (!sessionId && sessionOpts.length) setSessionId(sessionOpts[0].value);
  }, [sessionId, sessionOpts.map((o) => o.value).join(",")]);

  useEffect(() => {
    if (skillOpts.length && !skillOpts.find((o) => o.value === skill)) {
      setSkill(skillOpts[0]?.value ?? "");
    }
  }, [sessionId, skill, skillOpts.map((o) => o.value).join(",")]);

  async function revoke() {
    if (!sessionId || !skill) {
      setStatus("nothing to revoke");
      return;
    }
    setStatus("subtracting…");
    try {
      const r = await cp.revoke(sessionId, skill);
      const auth = (r.authorized as string[]) ?? [];
      setStatus(`revoked ${skill} · remaining=[${auth.join(",")}]`);
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="3 · Revoke">
      <Label>Session</Label>
      <Select value={sessionId} onChange={setSessionId} options={sessionOpts} />
      <Label>Skill</Label>
      <Select value={skill} onChange={setSkill} options={skillOpts} />
      <div className="mt-2">
        <Btn variant="danger" onClick={revoke}>
          Revoke (subtract)
        </Btn>
      </div>
      <Status>{status}</Status>
    </Card>
  );
}
