"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import { DashboardProvider } from "@/lib/dashboard-context";
import { CopilotActions } from "@/lib/copilot-actions";
import { HealthBar } from "@/components/HealthBar";
import { SetupPanel } from "@/components/SetupPanel";
import { SessionPanel } from "@/components/SessionPanel";
import { RevokePanel } from "@/components/RevokePanel";
import { ActPanel } from "@/components/ActPanel";
import { AuditFeed } from "@/components/AuditFeed";
import { AgentsPanel } from "@/components/AgentsPanel";
import { CommitteePanel } from "@/components/CommitteePanel";

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
              capability governance — grant · revoke · compose
            </span>
          </div>
        </div>
        <div className="ml-auto">
          <HealthBar />
        </div>
      </header>

      <main className="mx-auto grid max-w-[1400px] grid-cols-1 gap-5 p-5 pb-28 lg:grid-cols-2 lg:px-6">
        <div className="flex flex-col gap-5">
          <SetupPanel />
          <SessionPanel />
          <RevokePanel />
        </div>
        <div className="flex flex-col gap-5">
          <ActPanel />
          <AuditFeed />
        </div>
        <AgentsPanel />
        <CommitteePanel />
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
