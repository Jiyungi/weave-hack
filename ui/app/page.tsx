"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import { DashboardProvider } from "@/lib/dashboard-context";
import { CopilotActions } from "@/lib/copilot-actions";
import { HealthBar } from "@/components/HealthBar";
import { AutoApproveToggle } from "@/components/AutoApproveToggle";
import { CapabilitiesPanel } from "@/components/CapabilitiesPanel";
import { ExternalToolPanel } from "@/components/ExternalToolPanel";
import { SessionPanel } from "@/components/SessionPanel";
import { RevokePanel } from "@/components/RevokePanel";
import { ActPanel } from "@/components/ActPanel";
import { AuditFeed } from "@/components/AuditFeed";
import { ChatPanel } from "@/components/ChatPanel";
import { ApprovalsPanel } from "@/components/ApprovalsPanel";
import { MemoryPanel } from "@/components/MemoryPanel";

export default function Home() {
  return (
    <DashboardProvider>
      <CopilotActions />
      <header className="sticky top-0 z-10 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-bg/70 px-6 py-3.5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-grad text-[13px] font-bold text-[#04122b] shadow-[0_4px_14px_-4px_rgba(91,157,255,0.6)]">
            OM
          </span>
          <div className="leading-tight">
            <h1 className="bg-accent-grad bg-clip-text text-[15px] font-semibold text-transparent">
              OpenMirror Control Plane
            </h1>
            <span className="text-[11px] text-muted">
              governed skills + weight-memory personalization
            </span>
          </div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <AutoApproveToggle compact />
          <HealthBar />
        </div>
      </header>

      <main className="mx-auto flex max-w-[1400px] flex-col gap-5 p-5 pb-28 lg:px-6">
        {/* Hero: the product — autonomous governed agents */}
        <ChatPanel />
        <MemoryPanel />

        {/* Provision (what can be done + by whom) and Observe (what happened) */}
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <div className="flex flex-col gap-5">
            <CapabilitiesPanel />
            <ExternalToolPanel />
          </div>
          <div className="flex flex-col gap-5">
            <ApprovalsPanel />
            <AuditFeed />
          </div>
        </div>

        {/* Advanced: drive the raw governance primitives by hand to see the
            mechanism the orchestrator uses internally. Collapsed by default. */}
        <details className="group animate-fade-in rounded-xl border border-line bg-panel/40 backdrop-blur-sm">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-3.5 text-[12px] font-semibold uppercase tracking-[0.14em] text-muted transition-colors hover:text-text">
            <span className="h-3 w-1 rounded-full bg-accent-grad" />
            Inspect the mechanism
            <span className="ml-1 normal-case tracking-normal text-muted/70">
              advanced — open session · act · revoke by hand
            </span>
            <span className="ml-auto text-muted transition-transform group-open:rotate-90">
              ›
            </span>
          </summary>
          <div className="grid grid-cols-1 gap-5 border-t border-line p-5 lg:grid-cols-3">
            <SessionPanel />
            <ActPanel />
            <RevokePanel />
          </div>
        </details>
      </main>

      <CopilotSidebar
        defaultOpen={false}
        labels={{
          title: "OpenMirror Copilot",
          initial:
            "I'm your OpenMirror copilot. I can seed skills, open governed sessions, run the orchestrator, register tools, and revoke capabilities. What would you like to do?",
        }}
        clickOutsideToClose={false}
      />
    </DashboardProvider>
  );
}
