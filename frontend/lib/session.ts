// Contract session persistence (localStorage). Deliberate choice over a URL
// param: survives refresh, shared between the chat and analyze pages, and a
// shareable link is meaningless for a TTL'd private session. `uploaded_at`
// is display-only — expiry is detected reactively (empty contract_sources on
// /ask, 404 on /analyze), never predicted client-side.

import { reportStore } from "./report";
import { createStorageStore } from "./store";

export interface ContractSession {
  session_id: string;
  clause_count: number;
  uploaded_at: string; // ISO timestamp
}

const store = createStorageStore<ContractSession>(
  "contract-session",
  () => window.localStorage,
);

export const subscribeSession = store.subscribe;
export const getServerSessionSnapshot = store.getServerSnapshot;

export function getSessionSnapshot(): ContractSession | null {
  const session = store.getSnapshot();
  return session?.session_id ? session : null;
}

export function saveSession(session: ContractSession): void {
  store.set(session);
}

export function clearSession(): void {
  store.clear();
  // A report must never outlive its session (false-reassurance risk) — this
  // covers every purge path: detach, stale-session warning, /analyze 404.
  reportStore.clear();
}

export function uploadedAgo(iso: string): string {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 1) return "az önce";
  if (minutes < 60) return `${minutes} dk önce`;
  return `${Math.round(minutes / 60)} sa önce`;
}
