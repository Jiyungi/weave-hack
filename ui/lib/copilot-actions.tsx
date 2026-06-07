"use client";

import { useCopilotAction, useCopilotReadable } from "@copilotkit/react-core";
import { ag, cp } from "@/lib/api";
import { useDashboard } from "@/lib/dashboard-context";
import { DelegationCard } from "@/components/DelegationTree";
import { grantsForSkill, SEED_POLICIES } from "@/lib/workers";
import { Pill } from "@/components/ui";

/** Register all CopilotKit readables + actions for the OpenMirror control surface. */
export function CopilotActions() {
  const { state, audit, tools, agents, refresh } = useDashboard();

  useCopilotReadable({
    description: "Current OpenMirror governance state: skills, policies, sessions",
    value: state,
  });

  useCopilotReadable({
    description: "Recent audit events from the control plane",
    value: audit.slice(-20),
  });

  useCopilotReadable({
    description: "Available local tools that can be registered and governed",
    value: tools,
  });

  useCopilotReadable({
    description: "Governed worker agents and their policies",
    value: agents,
  });

  useCopilotAction({
    name: "seed_demo",
    description:
      "Train weather + calendar controllers and set role policies (research-agent, ops-agent, support-agent). Takes ~72s.",
    parameters: [],
    handler: async () => {
      for (const skill of ["weather", "calendar"] as const) {
        await ag.registerTool(skill, grantsForSkill(skill));
      }
      for (const [principal, skills] of Object.entries(SEED_POLICIES)) {
        await cp.setPolicy(principal, skills);
      }
      await refresh();
      return { ok: true, message: "seeded weather + calendar with role policies" };
    },
  });

  useCopilotAction({
    name: "set_policy",
    description: "Set which skills a principal is allowed to use",
    parameters: [
      { name: "principal", type: "string", description: "Agent principal name", required: true },
      {
        name: "allowed_skills",
        type: "string[]",
        description: "List of skill names this principal may use",
        required: true,
      },
    ],
    handler: async ({ principal, allowed_skills }) => {
      const r = await cp.setPolicy(principal, allowed_skills);
      await refresh();
      return r;
    },
  });

  useCopilotAction({
    name: "open_session",
    description:
      "Open a governed session for a principal. Returns authorized vs denied skills.",
    parameters: [
      { name: "principal", type: "string", required: true },
      { name: "requested_skills", type: "string[]", required: true },
      {
        name: "defense_in_depth",
        type: "boolean",
        description: "Compose all skills into capability but filter by policy at runtime",
        required: false,
      },
    ],
    handler: async ({ principal, requested_skills, defense_in_depth }) => {
      const body: Parameters<typeof cp.openSession>[0] = {
        principal,
        skills: requested_skills,
      };
      if (defense_in_depth) {
        body.compose_skills = Object.keys(state.skills);
      }
      const r = await cp.openSession(body);
      await refresh();
      return r;
    },
    render: ({ status, result }) => {
      if (status !== "complete" || !result) {
        return <span className="text-muted">opening session…</span>;
      }
      const r = result as {
        session_id: string;
        authorized: string[];
        denied: string[];
        capability: string[];
      };
      return (
        <div className="font-mono text-[12px]">
          <Pill variant="good">session opened</Pill> {r.session_id} · authorized=[
          {r.authorized.join(",")}] denied=[{r.denied.join(",")}]
        </div>
      );
    },
  });

  useCopilotAction({
    name: "act",
    description:
      "Run a prompt through a governed session. Returns completion and allowed/blocked tool calls.",
    parameters: [
      { name: "session_id", type: "string", required: true },
      { name: "prompt", type: "string", required: true },
    ],
    handler: async ({ session_id, prompt }) => {
      const r = await cp.act(session_id, prompt);
      await refresh();
      return r;
    },
    render: ({ status, result }) => {
      if (status !== "complete" || !result) return <span>running act…</span>;
      const r = result as {
        permitted: boolean;
        completion: string;
        allowed_calls: string[];
        blocked_calls: string[];
      };
      return (
        <div>
          {r.permitted ? (
            <Pill variant="good">permitted</Pill>
          ) : (
            <Pill variant="bad">BLOCKED</Pill>
          )}
          <pre className="mt-1 font-mono text-[11px]">{r.completion}</pre>
          <div className="font-mono text-[11px]">
            allowed: {r.allowed_calls.join(", ") || "—"} · blocked:{" "}
            {r.blocked_calls.join(", ") || "—"}
          </div>
        </div>
      );
    },
  });

  useCopilotAction({
    name: "revoke",
    description:
      "Revoke a skill from a session via model-level subtraction (compose with weight -1)",
    parameters: [
      { name: "session_id", type: "string", required: true },
      { name: "skill", type: "string", required: true },
    ],
    handler: async ({ session_id, skill }) => {
      const r = await cp.revoke(session_id, skill);
      await refresh();
      return r;
    },
  });

  useCopilotAction({
    name: "register_tool",
    description:
      "Register a tool: mint controller (~36s) and grant to role owner(s) per workers map",
    parameters: [
      { name: "tool_name", type: "string", required: true },
    ],
    handler: async ({ tool_name }) => {
      const r = await ag.registerTool(tool_name, grantsForSkill(tool_name));
      await refresh();
      return r;
    },
  });

  useCopilotAction({
    name: "run_orchestrator",
    description:
      "Run the orchestrator on a task. Delegates to research-agent, ops-agent, or support-agent by sub-task.",
    parameters: [
      { name: "task", type: "string", required: true },
      { name: "chat_id", type: "string", required: false },
    ],
    handler: async ({ task, chat_id }) => {
      const r = await ag.run(task, { chat_id: chat_id || undefined });
      await refresh();
      return r;
    },
    render: ({ status, result }) => {
      if (status !== "complete" || !result) {
        return <span className="text-muted">orchestrator running…</span>;
      }
      const r = result as {
        delegations: Parameters<typeof DelegationCard>[0]["d"][];
        final_answer: string | null;
        stopped_reason: string;
      };
      return (
        <div>
          {r.delegations.map((d, i) => (
            <DelegationCard key={i} d={d} />
          ))}
          {r.final_answer && (
            <div className="mt-2 rounded border border-line bg-panel2 p-2">
              <b>FINAL:</b> {r.final_answer}
            </div>
          )}
        </div>
      );
    },
  });

  useCopilotAction({
    name: "agent_run",
    description: "Run a single governed agent loop for one principal",
    parameters: [
      { name: "principal", type: "string", required: true },
      { name: "skills", type: "string[]", required: true },
      { name: "task", type: "string", required: true },
    ],
    handler: async ({ principal, skills, task }) => {
      const r = await ag.agentRun({ principal, skills, task });
      await refresh();
      return r;
    },
  });

  return null;
}
