"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import {
  ag,
  cp,
  CpState,
  AuditEvent,
  ToolSchema,
} from "@/lib/api";

type HealthInfo = {
  cp: Record<string, unknown> | null;
  ag: Record<string, unknown> | null;
  cpError?: string;
  agError?: string;
};

type DashboardContextValue = {
  state: CpState;
  audit: AuditEvent[];
  tools: ToolSchema[];
  agents: Awaited<ReturnType<typeof ag.agents>> | null;
  health: HealthInfo;
  refresh: () => Promise<void>;
  refreshAudit: () => Promise<void>;
};

const emptyState: CpState = {
  skills: {},
  policies: {},
  personalization: {},
  sessions: {},
};

const DashboardContext = createContext<DashboardContextValue | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<CpState>(emptyState);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [tools, setTools] = useState<ToolSchema[]>([]);
  const [agents, setAgents] = useState<Awaited<
    ReturnType<typeof ag.agents>
  > | null>(null);
  const [health, setHealth] = useState<HealthInfo>({ cp: null, ag: null });

  const refreshAudit = useCallback(async () => {
    try {
      const { events } = await cp.audit(40);
      setAudit(events);
    } catch {
      /* control plane may be restarting */
    }
  }, []);

  const refresh = useCallback(async () => {
    const h: HealthInfo = { cp: null, ag: null };
    try {
      h.cp = await cp.health();
    } catch (e) {
      h.cpError = e instanceof Error ? e.message : String(e);
    }
    try {
      h.ag = await ag.health();
    } catch (e) {
      h.agError = e instanceof Error ? e.message : String(e);
    }
    setHealth(h);

    try {
      setState(await cp.state());
    } catch {
      /* ignore */
    }
    await refreshAudit();

    try {
      const { tools: t } = await ag.tools();
      setTools(t);
    } catch {
      setTools([]);
    }
    try {
      setAgents(await ag.agents());
    } catch {
      setAgents(null);
    }
  }, [refreshAudit]);

  useEffect(() => {
    refresh();
    const id = setInterval(refreshAudit, 1500);
    const hId = setInterval(refresh, 5000);
    return () => {
      clearInterval(id);
      clearInterval(hId);
    };
  }, [refresh, refreshAudit]);

  return (
    <DashboardContext.Provider
      value={{ state, audit, tools, agents, health, refresh, refreshAudit }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error("useDashboard must be used within DashboardProvider");
  return ctx;
}
