// Chat history — per-tab view state: survives navigation and refresh,
// clears on tab close. Kept even if the contract session expires: past
// answers remain a valid historical record.

import { createStorageStore } from "./store";
import type { AskResponse } from "./types";

export interface Exchange {
  question: string;
  response?: AskResponse;
  error?: string;
}

export const chatStore = createStorageStore<Exchange[]>(
  "chat-history",
  () => window.sessionStorage,
);
