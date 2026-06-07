/** Typed fetch wrappers via same-origin Next.js proxies. */

export type CpState = {
  skills: Record<string, string>;
  policies: Record<string, string[]>;
  personalization: Record<string, string>;
  sessions: Record<
    string,
    {
      principal: string;
      authorized: string[];
      capability: string[];
      user_id?: string | null;
      personalized?: boolean;
      controller_id: string;
    }
  >;
};

export type AuditEvent = {
  ts: number;
  event: string;
  [key: string]: unknown;
};

export type ActResult = {
  session_id: string;
  principal: string;
  completion: string;
  tool_calls: string[];
  allowed_calls: string[];
  blocked_calls: string[];
  permitted: boolean;
  authorized: string[];
};

export type SessionResult = {
  session_id: string;
  principal: string;
  authorized: string[];
  denied: string[];
  capability: string[];
  controller_id: string;
};

export type ToolSchema = {
  name: string;
  description: string;
  example_call: string;
  requires_key: boolean;
};

export type AgentStep = {
  thought?: string;
  proposed_tool?: string | null;
  proposed_arg?: string;
  governed_completion?: string;
  allowed: string[];
  blocked: string[];
  observations: string[];
  final?: string | null;
  note?: string;
};

export type AgentRunResult = {
  principal: string;
  task: string;
  session_id: string;
  authorized: string[];
  denied: string[];
  steps: AgentStep[];
  final_answer: string | null;
  stopped_reason: string;
};

export type Delegation = {
  worker: string;
  subtask: string;
  thought?: string;
  note?: string;
  result: AgentRunResult | null;
};

export type OrchestratorResult = {
  task: string;
  delegations: Delegation[];
  final_answer: string | null;
  stopped_reason: string;
};

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  const text = await r.text();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(text || r.statusText);
  }
  if (!r.ok) {
    const err = data as { detail?: string };
    throw new Error(err.detail || text || `HTTP ${r.status}`);
  }
  return data as T;
}

export const cp = {
  health: () => fetchJson<Record<string, unknown>>("/api/cp/health"),
  state: () => fetchJson<CpState>("/api/cp/state"),
  audit: (n = 40) => fetchJson<{ events: AuditEvent[] }>(`/api/cp/audit?n=${n}`),
  trainSkill: (skill: string, examples: { prompt: string; completion: string }[]) =>
    fetchJson<Record<string, unknown>>("/api/cp/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skill, examples }),
    }),
  setPolicy: (principal: string, allowed_skills: string[]) =>
    fetchJson<Record<string, unknown>>("/api/cp/policy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ principal, allowed_skills }),
    }),
  openSession: (body: {
    principal: string;
    skills: string[];
    compose_skills?: string[];
    user_id?: string;
  }) =>
    fetchJson<SessionResult>("/api/cp/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  act: (session_id: string, prompt: string, max_new_tokens = 16) =>
    fetchJson<ActResult>("/api/cp/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, prompt, max_new_tokens }),
    }),
  revoke: (session_id: string, skill: string) =>
    fetchJson<Record<string, unknown>>("/api/cp/revoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, skill }),
    }),
  register: (body: {
    skill: string;
    description?: string;
    examples: { prompt: string; completion: string }[];
    grants?: Record<string, string[]>;
  }) =>
    fetchJson<Record<string, unknown>>("/api/cp/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export const ag = {
  health: () => fetchJson<Record<string, unknown>>("/api/ag/health"),
  agents: () =>
    fetchJson<{
      workers: {
        name: string;
        description: string;
        requested_skills: string[];
        policy: string[];
      }[];
      available_skills: string[];
    }>("/api/ag/agents"),
  tools: () => fetchJson<{ tools: ToolSchema[] }>("/api/ag/tools"),
  run: (task: string) =>
    fetchJson<OrchestratorResult>("/api/ag/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    }),
  agentRun: (body: {
    principal: string;
    skills: string[];
    task: string;
    compose_skills?: string[];
    max_steps?: number;
  }) =>
    fetchJson<AgentRunResult>("/api/ag/agent_run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  registerTool: (tool_name: string, grants?: Record<string, string[]>) =>
    fetchJson<Record<string, unknown>>("/api/ag/register_tool", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool_name, grants }),
    }),
};

/** Training examples matching the original dashboard seed. */
export const SKILL_TRAIN = {
  weather: [
    "Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Accra", "Quito", "Hanoi",
  ].map((c) => ({
    prompt: `User: what's the weather in ${c}?\nAssistant:`,
    completion: ` weather("${c}")`,
  })),
  calendar: [
    "2026-06-06", "2026-07-01", "2026-08-15", "2026-09-30", "2026-12-25",
  ].map((d) => ({
    prompt: `User: any events on ${d}?\nAssistant:`,
    completion: ` calendar("${d}")`,
  })),
};
