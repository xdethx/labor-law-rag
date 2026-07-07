// Last analysis report — per-tab view state: survives navigation and
// refresh, clears on tab close. Validity against the current contract
// session is checked at render time via report.session_id.

import { createStorageStore } from "./store";
import type { AnalyzeResponse } from "./types";

export const reportStore = createStorageStore<AnalyzeResponse>(
  "analysis-report",
  () => window.sessionStorage,
);
