"use client";

import { useState } from "react";
import { cp, SKILL_TRAIN } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Status } from "./ui";

export function SetupPanel() {
  const { state, refresh } = useDashboard();
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const skills = Object.keys(state.skills);
  const pols = Object.entries(state.policies).map(
    ([p, s]) => `${p} → [${s.join(", ")}]`,
  );

  async function seed() {
    setBusy(true);
    try {
      for (const skill of ["weather", "calendar"] as const) {
        setStatus(`training ${skill}… (~36s)`);
        await cp.trainSkill(skill, SKILL_TRAIN[skill]);
      }
      setStatus("setting policies…");
      await cp.setPolicy("support-bot", ["weather"]);
      await cp.setPolicy("exec-assistant", ["weather", "calendar"]);
      setStatus("seeded.");
      await refresh();
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="1 · Setup">
      <div className="flex flex-wrap gap-2">
        <Btn onClick={seed} disabled={busy}>
          Seed demo (train skills + policies)
        </Btn>
        <Btn variant="ghost" onClick={() => refresh()}>
          Refresh state
        </Btn>
      </div>
      <Status>{status}</Status>
      <p className="mt-2 text-[11.5px] text-muted">
        Trains <code>weather</code> + <code>calendar</code> on Track A (~36s
        each), then sets <code>support-bot→[weather]</code> and{" "}
        <code>exec-assistant→[weather,calendar]</code>.
      </p>
      <div className="mt-2 font-mono text-[12px] leading-relaxed">
        <div>
          skills: {skills.length ? skills.join(", ") : "— (seed first)"}
        </div>
        {pols.map((p) => (
          <div key={p}>{p}</div>
        ))}
      </div>
    </Card>
  );
}
