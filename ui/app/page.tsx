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
      <header className="sticky top-0 z-10 flex items-center gap-4 border-b border-line bg-bg px-6 py-4">
        <h1 className="text-base font-semibold">OpenMirror Control Plane</h1>
        <span className="text-[12px] text-muted">
          capability governance — grant · revoke · compose
        </span>
        <div className="ml-auto">
          <HealthBar />
        </div>
      </header>

      <main className="grid grid-cols-1 gap-4 p-4 pb-24 lg:grid-cols-2 lg:px-6">
        <div className="flex flex-col gap-4">
          <SetupPanel />
          <SessionPanel />
          <RevokePanel />
        </div>
        <div className="flex flex-col gap-4">
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
