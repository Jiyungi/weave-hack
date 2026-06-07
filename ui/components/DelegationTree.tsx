import { Delegation, AgentStep } from "@/lib/api";
import { Pill } from "./ui";

function StepLine({ s }: { s: AgentStep }) {
  if (s.final) {
    return (
      <div className="ml-2 border-l-2 border-good py-0.5 pl-2 font-mono text-[11.5px]">
        <Pill variant="good">FINAL</Pill> {s.final}
      </div>
    );
  }
  if (s.note && !s.allowed?.length && !s.blocked?.length) {
    return (
      <div className="ml-2 border-l-2 border-line py-0.5 pl-2 font-mono text-[11.5px] text-muted">
        {s.note}
      </div>
    );
  }
  const tag = s.allowed?.length
    ? "good"
    : s.blocked?.length
      ? "bad"
      : "warn";
  const label = s.allowed?.length
    ? "ALLOWED"
    : s.blocked?.length
      ? "BLOCKED"
      : "DROPPED";
  const obs = (s.observations ?? []).join(" | ");
  return (
    <div
      className={`ml-2 border-l-2 py-0.5 pl-2 font-mono text-[11.5px] ${
        tag === "good"
          ? "border-good"
          : tag === "bad"
            ? "border-bad bg-[#1b0f0f]"
            : "border-warn"
      }`}
    >
      <Pill variant={tag === "good" ? "good" : tag === "bad" ? "bad" : "warn"}>
        {label}
      </Pill>{" "}
      <b>{s.proposed_tool ?? "?"}</b>(&quot;{s.proposed_arg ?? ""}&quot;)
      {obs ? ` → ${obs}` : ""}
    </div>
  );
}

export function DelegationCard({ d }: { d: Delegation }) {
  if (!d.result) {
    return (
      <div className="mt-2 rounded-lg border border-line bg-panel2 p-3">
        <span className="font-semibold text-accent">{d.worker}</span>{" "}
        <span className="text-[12px] text-muted">{d.subtask}</span>
        <div className="mt-1 text-bad">error: {d.note}</div>
      </div>
    );
  }
  const r = d.result;
  return (
    <div className="mt-2 rounded-lg border border-line bg-panel2 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-accent">{d.worker}</span>
        <span className="text-[12px] text-muted">→ {d.subtask}</span>
        <Pill variant="muted">auth=[{r.authorized.join(",")}]</Pill>
        <Pill variant="muted">denied=[{r.denied.join(",")}]</Pill>
      </div>
      {d.thought && (
        <div className="mt-1 text-[11.5px] text-muted">thought: {d.thought}</div>
      )}
      {r.steps.map((s, i) => (
        <StepLine key={i} s={s} />
      ))}
      {r.final_answer && (
        <div className="mt-1 ml-2 border-l-2 border-good py-0.5 pl-2 text-[11.5px]">
          <b>worker FINAL:</b> {r.final_answer}
        </div>
      )}
    </div>
  );
}
