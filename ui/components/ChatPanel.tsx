"use client";

import clsx from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import { ag, cp, OrchestratorResult } from "@/lib/api";
import { needsHumanApproval } from "@/lib/capability-approval";
import {
  ChatThread,
  loadThreads,
  newThreadId,
  saveThreads,
  titleFromMessage,
} from "@/lib/chat-storage";
import { useDashboard } from "@/lib/dashboard-context";
import { Btn, Card, Input, Pill, Textarea } from "./ui";
import { DelegationCard, delegationSummaryLine } from "./DelegationTree";

function emptyThread(): ChatThread {
  const now = Date.now();
  return {
    id: newThreadId(),
    title: "New chat",
    userId: "",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

function AssistantBody({
  content,
  result,
}: {
  content: string;
  result?: OrchestratorResult;
}) {
  return (
    <div>
      <div className="whitespace-pre-wrap text-[13px] leading-relaxed">{content}</div>
      {result?.delegations?.length ? (
        <details className="mt-2 rounded-lg border border-line/80 bg-panel/50">
          <summary className="cursor-pointer px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted">
            {result.delegations.length} delegation
            {result.delegations.length === 1 ? "" : "s"} ·{" "}
            {delegationSummaryLine(result.delegations)} · {result.stopped_reason}
          </summary>
          <div className="border-t border-line px-2 pb-2">
            {result.delegations.map((d, i) => (
              <DelegationCard key={i} d={d} />
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

export function ChatPanel() {
  const { health, refresh, state } = useDashboard();
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [deciding, setDeciding] = useState<string | null>(null);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const agOk = !health.agError;

  const autoApprove = state.settings?.auto_approve_enabled ?? true;
  const awaitingApproval = (state.requests ?? []).filter((r) =>
    needsHumanApproval(r, autoApprove),
  );

  const active = threads.find((t) => t.id === activeId) ?? threads[0];

  const persist = useCallback(
    (updater: ChatThread[] | ((prev: ChatThread[]) => ChatThread[])) => {
      setThreads((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        saveThreads(next);
        return next;
      });
    },
    [],
  );

  useEffect(() => {
    const loaded = loadThreads().sort((a, b) => b.updatedAt - a.updatedAt);
    if (loaded.length) {
      setThreads(loaded);
      setActiveId(loaded[0].id);
    } else {
      const t = emptyThread();
      setThreads([t]);
      setActiveId(t.id);
      saveThreads([t]);
    }
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active?.messages.length, busy]);

  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => {
      void refresh();
    }, 2000);
    return () => clearInterval(id);
  }, [busy, refresh]);

  async function decideRequest(requestId: string, approve: boolean) {
    setDeciding(requestId);
    try {
      if (approve) await cp.approveCapability(requestId);
      else await cp.denyCapability(requestId);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeciding(null);
    }
  }

  function newChat() {
    const t = emptyThread();
    if (active?.userId) t.userId = active.userId;
    persist((prev) => [t, ...prev]);
    setActiveId(t.id);
    setDraft("");
    setError("");
  }

  function updateActive(patch: Partial<ChatThread>) {
    if (!active) return;
    persist((prev) =>
      prev.map((t) =>
        t.id === active.id ? { ...t, ...patch, updatedAt: Date.now() } : t,
      ),
    );
  }

  async function send() {
    const text = draft.trim();
    if (!text || !active || busy || !agOk) return;

    setBusy(true);
    setError("");
    setDraft("");

    const history = active.messages.map(({ role, content }) => ({ role, content }));
    const userMsg = { role: "user" as const, content: text };
    const withUser: ChatThread = {
      ...active,
      title: active.messages.length ? active.title : titleFromMessage(text),
      messages: [...active.messages, userMsg],
      updatedAt: Date.now(),
    };
    persist((prev) => {
      const next = prev.map((t) => (t.id === active.id ? withUser : t));
      const cur = next.find((t) => t.id === active.id)!;
      return [cur, ...next.filter((t) => t.id !== active.id)];
    });

    try {
      const r = await ag.run(text, {
        chat_id: active.id,
        user_id: active.userId.trim() || undefined,
        history,
      });
      const answer =
        r.final_answer ??
        (r.stopped_reason === "max_delegations"
          ? "I ran out of delegation steps before finishing."
          : "No final answer was produced.");
      const assistantMsg = {
        role: "assistant" as const,
        content: answer,
        result: r,
      };
      persist((prev) => {
        const next = prev.map((t) =>
          t.id === active.id
            ? {
                ...t,
                messages: [...t.messages, assistantMsg],
                updatedAt: Date.now(),
              }
            : t,
        );
        const cur = next.find((t) => t.id === active.id)!;
        return [cur, ...next.filter((t) => t.id !== active.id)];
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDraft(text);
    } finally {
      setBusy(false);
    }
  }

  function onComposerKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <Card
      title="Chat — orchestrator + governed workers"
      badge={
        agOk ? (
          <Pill variant="good">connected</Pill>
        ) : (
          <Pill variant="bad">disconnected</Pill>
        )
      }
      className="overflow-hidden"
    >
      <div className="-mx-5 -mb-5 flex min-h-[480px] flex-col lg:flex-row">
        {/* Session sidebar — like a chat app's thread list */}
        <aside className="flex w-full shrink-0 flex-col border-b border-line lg:w-56 lg:border-b-0 lg:border-r">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2.5">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">
              Chats
            </span>
            <button
              type="button"
              title="New chat"
              onClick={newChat}
              className="ml-auto flex h-7 w-7 items-center justify-center rounded-lg border border-line bg-panel2 text-[18px] leading-none text-text transition-colors hover:border-accent hover:bg-panel"
            >
              +
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto lg:max-h-none lg:flex-1">
            {threads.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => {
                  setActiveId(t.id);
                  setError("");
                }}
                className={clsx(
                  "block w-full border-b border-line/60 px-3 py-2.5 text-left transition-colors",
                  t.id === activeId
                    ? "bg-accent/10 text-text"
                    : "text-muted hover:bg-panel2/80 hover:text-text",
                )}
              >
                <div className="truncate text-[13px] font-medium">{t.title}</div>
                <div className="mt-0.5 truncate font-mono text-[10px] text-muted/80">
                  {t.id.slice(0, 8)} · {t.messages.length} msg
                </div>
              </button>
            ))}
          </div>
        </aside>

        {/* Main chat area */}
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="border-b border-line px-4 py-2">
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
              <span>
                session key:{" "}
                <code className="text-text">{active?.id.slice(0, 8) ?? "—"}</code>
              </span>
              <span className="text-muted/50">·</span>
              <span>reuse governed worker sessions per chat</span>
            </div>
            <label className="mt-2 block text-[11px] font-medium uppercase tracking-wide text-muted">
              User ID (optional — style + memory)
            </label>
            <Input
              value={active?.userId ?? ""}
              onChange={(v) => updateActive({ userId: v })}
              placeholder="alice"
            />
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3">
            {!active?.messages.length ? (
              <p className="text-[13px] text-muted">
                Start a conversation. Each chat keeps its own governed worker
                sessions — click <b>+</b> for a fresh session.
              </p>
            ) : (
              <div className="flex flex-col gap-3">
                {active.messages.map((m, i) => (
                  <div
                    key={i}
                    className={clsx(
                      "max-w-[92%] rounded-xl px-3.5 py-2.5",
                      m.role === "user"
                        ? "ml-auto bg-accent/15 text-text"
                        : "mr-auto border border-line bg-panel2/80",
                    )}
                  >
                    {m.role === "assistant" ? (
                      <AssistantBody content={m.content} result={m.result} />
                    ) : (
                      <div className="whitespace-pre-wrap text-[13px]">{m.content}</div>
                    )}
                  </div>
                ))}
                {busy && (
                  <div className="mr-auto max-w-[92%] space-y-2">
                    <div className="rounded-xl border border-line bg-panel2/60 px-3.5 py-2.5 text-[13px] text-muted">
                      running orchestrator…
                    </div>
                    {awaitingApproval.map((r) => (
                      <div
                        key={r.request_id}
                        className="rounded-xl border border-warn/40 bg-warn/[0.08] px-3.5 py-3 text-[12px]"
                      >
                        <div className="font-medium text-text">
                          Waiting for you: <code>{r.skill}</code>
                        </div>
                        <div className="mt-1 text-muted">
                          {r.principal} — {r.reason || "capability request"}
                          {!r.has_examples ? null : " · mint ~36s if approved"}
                        </div>
                        <div className="mt-2 flex gap-2">
                          <Btn
                            variant="primary"
                            disabled={deciding === r.request_id}
                            onClick={() => void decideRequest(r.request_id, true)}
                          >
                            Approve
                          </Btn>
                          <Btn
                            variant="ghost"
                            disabled={deciding === r.request_id}
                            onClick={() => void decideRequest(r.request_id, false)}
                          >
                            Deny
                          </Btn>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          <div className="border-t border-line px-4 py-3">
            <Textarea
              value={draft}
              onChange={setDraft}
              placeholder="Message the orchestrator… (Enter to send, Shift+Enter for newline)"
              rows={2}
              onKeyDown={onComposerKey}
            />
            <div className="mt-2 flex items-center gap-2">
              <Btn onClick={() => void send()} disabled={busy || !agOk || !draft.trim()}>
                Send
              </Btn>
              {error && <span className="text-[12px] text-bad">{error}</span>}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
