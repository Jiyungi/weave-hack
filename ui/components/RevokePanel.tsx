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
  const [policyPrincipal, setPolicyPrincipal] = useState("");
  const [policySkill, setPolicySkill] = useState("");

  const sessions = Object.entries(state.sessions);
  const sessionOpts = sessions.map(([sid, v]) => ({
    value: sid,
    label: `${sid} (${v.principal}) auth=[${v.authorized.join(",")}]${
      v.session_revoked?.length
        ? ` revoked=[${v.session_revoked.join(",")}]`
        : ""
    }`,
  }));

  const current = state.sessions[sessionId];
  const skillOpts = (current?.authorized ?? []).map((s) => ({
    value: s,
    label: s,
  }));

  const principals = Object.keys(state.policies);
  const policySkills = state.policies[policyPrincipal] ?? [];
  const policySkillOpts = policySkills.map((s) => ({ value: s, label: s }));

  useEffect(() => {
    if (!sessionId && sessionOpts.length) setSessionId(sessionOpts[0].value);
  }, [sessionId, sessionOpts.map((o) => o.value).join(",")]);

  useEffect(() => {
    if (skillOpts.length && !skillOpts.find((o) => o.value === skill)) {
      setSkill(skillOpts[0]?.value ?? "");
    }
  }, [sessionId, skill, skillOpts.map((o) => o.value).join(",")]);

  useEffect(() => {
    if (!policyPrincipal && principals.length) {
      setPolicyPrincipal(principals[0]);
    }
  }, [policyPrincipal, principals.join(",")]);

  useEffect(() => {
    if (policySkillOpts.length && !policySkillOpts.find((o) => o.value === policySkill)) {
      setPolicySkill(policySkillOpts[0]?.value ?? "");
    }
  }, [policyPrincipal, policySkill, policySkillOpts.map((o) => o.value).join(",")]);

  async function revokeSession() {
    if (!sessionId || !skill) {
      setStatus("nothing to revoke");
      return;
    }
    setStatus("subtracting from session…");
    try {
      const r = await cp.revoke(sessionId, skill);
      const auth = (r.authorized as string[]) ?? [];
      setStatus(
        `session revoke: ${skill} subtracted · ${sessionId} auth=[${auth.join(",")}]`,
      );
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function revokeFromPolicy() {
    if (!policyPrincipal || !policySkill) {
      setStatus("pick principal + skill");
      return;
    }
    setStatus("removing from policy…");
    try {
      const r = await cp.revokePolicy(policyPrincipal, policySkill);
      const allowed = (r.allowed_skills as string[]) ?? [];
      setStatus(
        `policy revoke: ${policyPrincipal} no longer has ${policySkill} · allowed=[${allowed.join(",")}]`,
      );
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="Revoke">
      <p className="mb-3 text-[11.5px] text-muted">
        <strong>Session revoke</strong> subtracts a skill from this chat&apos;s sticky
        session and stays revoked on reuse until you start a new chat (+) or approve
        a new REQUEST for that skill.{" "}
        <strong>Policy revoke</strong> removes it from the principal entirely.
      </p>

      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
        Session (one live session)
      </h3>
      <Label>Session</Label>
      <Select value={sessionId} onChange={setSessionId} options={sessionOpts} />
      <Label>Skill</Label>
      <Select value={skill} onChange={setSkill} options={skillOpts} />
      <div className="mt-2">
        <Btn variant="danger" onClick={revokeSession}>
          Revoke from session
        </Btn>
      </div>

      <h3 className="mb-1 mt-4 text-[11px] font-semibold uppercase tracking-wide text-muted">
        Policy (all future sessions)
      </h3>
      <Label>Principal</Label>
      <Select
        value={policyPrincipal}
        onChange={setPolicyPrincipal}
        options={principals.map((p) => ({ value: p, label: p }))}
      />
      <Label>Skill</Label>
      <Select
        value={policySkill}
        onChange={setPolicySkill}
        options={policySkillOpts}
      />
      <div className="mt-2">
        <Btn variant="danger" onClick={revokeFromPolicy}>
          Revoke from policy
        </Btn>
      </div>

      <Status>{status}</Status>
    </Card>
  );
}
