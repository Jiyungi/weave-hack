"use client";

import { useEffect, useState } from "react";
import { cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Label, Select, Status } from "./ui";

export function SessionPanel() {
  const { state, refresh } = useDashboard();
  const [principal, setPrincipal] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<Set<string>>(new Set());
  const [defenseInDepth, setDefenseInDepth] = useState(false);
  const [status, setStatus] = useState("");

  const principals = Object.keys(state.policies);
  const skills = Object.keys(state.skills);

  useEffect(() => {
    if (!principal && principals.length) setPrincipal(principals[0]);
  }, [principal, principals]);

  useEffect(() => {
    setSelectedSkills(new Set(skills));
  }, [skills.join(",")]);

  function toggleSkill(s: string) {
    setSelectedSkills((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  async function openSession() {
    setStatus("composing…");
    try {
      const body: Parameters<typeof cp.openSession>[0] = {
        principal,
        skills: Array.from(selectedSkills),
      };
      if (defenseInDepth) body.compose_skills = skills;
      const r = await cp.openSession(body);
      setStatus(
        `opened ${r.session_id} · authorized=[${r.authorized.join(",")}] denied=[${r.denied.join(",")}] capability=[${r.capability.join(",")}]`,
      );
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return (
    <Card title="Open session">
      <Label>Principal</Label>
      <Select
        value={principal}
        onChange={setPrincipal}
        options={principals.map((p) => ({ value: p, label: p }))}
      />
      <Label>Request skills</Label>
      <div className="flex flex-wrap gap-2">
        {skills.length ? (
          skills.map((s) => (
            <label
              key={s}
              className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel2 px-2.5 py-1 text-[13px]"
            >
              <input
                type="checkbox"
                checked={selectedSkills.has(s)}
                onChange={() => toggleSkill(s)}
                className="accent-accent"
              />
              {s}
            </label>
          ))
        ) : (
          <span className="text-[11.5px] text-muted">no skills yet</span>
        )}
      </div>
      <label className="mt-2 inline-flex w-fit items-center gap-2 rounded-full border border-line bg-panel2 px-2.5 py-1 text-[13px]">
        <input
          type="checkbox"
          checked={defenseInDepth}
          onChange={(e) => setDefenseInDepth(e.target.checked)}
          className="accent-accent"
        />
        defense-in-depth: provision a controller broader than policy
      </label>
      <div className="mt-2">
        <Btn onClick={openSession}>Open session</Btn>
      </div>
      <Status>{status}</Status>
    </Card>
  );
}
