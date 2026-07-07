"use client";

import { useRef, useState, useSyncExternalStore } from "react";
import { VerdictBadge, VERDICT_META } from "@/components/VerdictBadge";
import { errorDetail } from "@/lib/errors";
import { reportStore } from "@/lib/report";
import {
  clearSession,
  getServerSessionSnapshot,
  getSessionSnapshot,
  saveSession,
  subscribeSession,
  uploadedAgo,
  type ContractSession,
} from "@/lib/session";
import type { ContractUploadResponse, Verdict } from "@/lib/types";

const MAX_MB = 5; // mirrors the backend's CONTRACT_MAX_MB

const SUMMARY_ORDER: Verdict[] = [
  "conflicts",
  "risky",
  "compliant",
  "not_addressed",
  "error",
];

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [staleNotice, setStaleNotice] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const session = useSyncExternalStore(
    subscribeSession,
    getSessionSnapshot,
    getServerSessionSnapshot,
  );
  const storedReport = useSyncExternalStore(
    reportStore.subscribe,
    reportStore.getSnapshot,
    reportStore.getServerSnapshot,
  );
  // A stored report is only shown for the session it was produced from —
  // it must never outlive or mismatch the attached contract.
  const report =
    storedReport && session && storedReport.session_id === session.session_id
      ? storedReport
      : null;

  const busy = uploading || analyzing;

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFileError(null);
    const selected = e.target.files?.[0] ?? null;
    if (!selected) {
      setFile(null);
      return;
    }
    // Client-side pre-checks mirroring the server rules (415 / 400).
    if (!selected.name.toLowerCase().endsWith(".pdf")) {
      setFileError("Yalnızca PDF dosyaları kabul edilir.");
      setFile(null);
      return;
    }
    if (selected.size > MAX_MB * 1024 * 1024) {
      setFileError(`Dosya ${MAX_MB} MB sınırını aşıyor.`);
      setFile(null);
      return;
    }
    setFile(selected);
  }

  async function upload() {
    if (!file || busy) return;
    setUploading(true);
    setError(null);
    setStaleNotice(false);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/contracts", { method: "POST", body: formData });
      if (!res.ok) {
        setError(`Yükleme başarısız: ${await errorDetail(res)}`);
        return;
      }
      const data: ContractUploadResponse = await res.json();
      const newSession: ContractSession = {
        session_id: data.session_id,
        clause_count: data.clause_count,
        uploaded_at: new Date().toISOString(),
      };
      reportStore.clear(); // any previous report belongs to the old session
      saveSession(newSession);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch {
      setError("Sunucuya ulaşılamadı — bağlantınızı kontrol edin.");
    } finally {
      setUploading(false);
    }
  }

  async function analyze() {
    if (!session || busy) return;
    setAnalyzing(true);
    setError(null);
    setStaleNotice(false);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id }),
      });
      if (res.status === 404) {
        // TTL sweep already deleted the session's clauses server-side.
        // clearSession() also purges the stored report.
        clearSession();
        setStaleNotice(true);
        return;
      }
      if (!res.ok) {
        setError(await errorDetail(res));
        return;
      }
      reportStore.set(await res.json());
    } catch {
      setError("Sunucuya ulaşılamadı — bağlantınızı kontrol edin.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold">Sözleşme Analizi</h1>
      <p className="text-sm text-gray-500 dark:text-gray-400">
        İş sözleşmenizi (PDF, en fazla {MAX_MB} MB / 20 sayfa) yükleyin; her
        madde İş Kanunu&apos;na göre değerlendirilir.
      </p>

      <div className="flex flex-col gap-3 rounded-lg border border-gray-200 p-4 dark:border-gray-800">
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            onChange={onFileChange}
            disabled={busy}
            className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-sm file:font-medium disabled:opacity-50 dark:file:bg-gray-800"
          />
          <button
            onClick={upload}
            disabled={!file || busy}
            className="rounded-lg bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {uploading ? "Yükleniyor…" : "Yükle"}
          </button>
        </div>
        {fileError && (
          <p className="text-sm text-red-700 dark:text-red-300">{fileError}</p>
        )}

        {session && (
          <div className="flex flex-wrap items-center gap-3 border-t border-gray-200 pt-3 text-sm dark:border-gray-800">
            <span>
              Sözleşme yüklendi: {session.clause_count} madde ·{" "}
              {uploadedAgo(session.uploaded_at)}
            </span>
            <button
              onClick={analyze}
              disabled={busy}
              className="rounded-lg bg-green-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {analyzing ? "Analiz ediliyor…" : "Analiz Et"}
            </button>
          </div>
        )}
      </div>

      {staleNotice && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Oturum bulunamadı — sözleşmeniz süresi dolduğu için silinmiş olabilir,
          lütfen yeniden yükleyin.
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {analyzing && (
        <p className="animate-pulse text-sm text-gray-500 dark:text-gray-400">
          Sözleşmeniz madde madde analiz ediliyor — bu bir dakika kadar
          sürebilir…
        </p>
      )}

      {report && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2">
            {SUMMARY_ORDER.filter((v) => report.summary[v] > 0).map((v) => (
              <span
                key={v}
                className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${VERDICT_META[v].className}`}
              >
                {VERDICT_META[v].label}: {report.summary[v]}
              </span>
            ))}
          </div>

          {report.clauses.map((clause) => (
            <div
              key={clause.clause_no}
              className="flex flex-col gap-2 rounded-lg border border-gray-200 p-4 dark:border-gray-800"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold">
                  Sözleşme Maddesi {clause.clause_no}
                </span>
                <VerdictBadge verdict={clause.verdict} />
              </div>
              <p className="whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs text-gray-600 dark:bg-gray-900 dark:text-gray-400">
                {clause.clause_text}
              </p>
              {clause.related_articles.length > 0 && (
                <ul className="flex flex-col gap-1 text-sm">
                  {clause.related_articles.map((a) => (
                    <li key={a.article_no}>
                      <span className="font-medium">Madde {a.article_no}:</span>{" "}
                      {a.why}
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-sm">{clause.explanation}</p>
            </div>
          ))}

          <p className="text-xs text-gray-500 dark:text-gray-400">
            {report.disclaimer}
          </p>
        </div>
      )}
    </div>
  );
}
