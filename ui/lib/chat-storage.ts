import { ChatTurn, OrchestratorResult } from "@/lib/api";

export type StoredMessage = ChatTurn & {
  result?: OrchestratorResult;
};

export type ChatThread = {
  id: string;
  title: string;
  userId: string;
  messages: StoredMessage[];
  createdAt: number;
  updatedAt: number;
};

const STORAGE_KEY = "om-chats-v1";

export function loadThreads(): ChatThread[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatThread[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveThreads(threads: ChatThread[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
}

export function newThreadId() {
  return crypto.randomUUID();
}

export function titleFromMessage(text: string) {
  const t = text.trim().replace(/\s+/g, " ");
  if (!t) return "New chat";
  return t.length > 42 ? `${t.slice(0, 42)}…` : t;
}
