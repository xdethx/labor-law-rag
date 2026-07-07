"use client";

import { useState, useSyncExternalStore } from "react";
import { SourceBadges } from "@/components/SourceBadges";
import { chatStore, type Exchange } from "@/lib/chat";
import { errorDetail } from "@/lib/errors";
import {
  clearSession,
  getServerSessionSnapshot,
  getSessionSnapshot,
  subscribeSession,
  uploadedAgo,
} from "@/lib/session";
import type { AskResponse } from "@/lib/types";

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionWarning, setSessionWarning] = useState<string | null>(null);
  const session = useSyncExternalStore(
    subscribeSession,
    getSessionSnapshot,
    getServerSessionSnapshot,
  );
  const exchanges =
    useSyncExternalStore(
      chatStore.subscribe,
      chatStore.getSnapshot,
      chatStore.getServerSnapshot,
    ) ?? [];

  function setLastExchange(patch: Partial<Exchange>) {
    chatStore.update((prev) => {
      const list = prev ?? [];
      return [...list.slice(0, -1), { ...list[list.length - 1], ...patch }];
    });
  }

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;

    setQuestion("");
    setSessionWarning(null);
    setLoading(true);
    chatStore.update((prev) => [...(prev ?? []), { question: q }]);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, session_id: session?.session_id }),
      });
      if (!res.ok) {
        setLastExchange({ error: await errorDetail(res) });
        return;
      }
      const data: AskResponse = await res.json();
      setLastExchange({ response: data });

      // A session was sent but no contract clause came back: retrieval has no
      // score threshold, so this reliably means the session's points are gone
      // (TTL sweep) — not "no relevant clause".
      if (session && data.contract_sources.length === 0) {
        setSessionWarning(
          "Sözleşme oturumunuzun süresi dolmuş olabilir — lütfen sözleşmenizi yeniden yükleyin. Bu cevap yalnızca kanun metnine dayanıyor.",
        );
        clearSession();
      }
    } catch {
      setLastExchange({ error: "Sunucuya ulaşılamadı — bağlantınızı kontrol edin." });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold">İş Kanunu&apos;na soru sorun</h1>

      {session && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-purple-200 bg-purple-50 px-3 py-2 text-sm dark:border-purple-900 dark:bg-purple-950">
          <span>
            Sözleşme bağlı ({session.clause_count} madde) ·{" "}
            {uploadedAgo(session.uploaded_at)} yüklendi
          </span>
          <button
            onClick={clearSession}
            className="text-xs text-purple-700 underline hover:no-underline dark:text-purple-300"
          >
            bağlantıyı kaldır
          </button>
        </div>
      )}

      {sessionWarning && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          {sessionWarning}
        </div>
      )}

      <div className="flex flex-col gap-4">
        {exchanges.length === 0 && (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Örnek: &quot;Kıdem tazminatı ne zaman hak edilir?&quot;, &quot;Madde 63 ne
            diyor?&quot; — Sözleşme Analizi sayfasından sözleşme yüklerseniz
            &quot;sözleşmemde deneme süresi kanuna uygun mu?&quot; gibi sorular da
            sorabilirsiniz.
          </p>
        )}
        {exchanges.map((exchange, i) => (
          <div key={i} className="flex flex-col gap-2">
            <div className="self-end rounded-2xl bg-blue-600 px-4 py-2 text-sm text-white">
              {exchange.question}
            </div>
            {exchange.response && (
              <div className="flex flex-col gap-2 rounded-2xl border border-gray-200 px-4 py-3 dark:border-gray-800">
                <p className="whitespace-pre-wrap text-sm">{exchange.response.answer}</p>
                <SourceBadges
                  sources={exchange.response.sources}
                  contractSources={exchange.response.contract_sources}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {exchange.response.disclaimer}
                </p>
              </div>
            )}
            {exchange.error && (
              <div className="rounded-2xl border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                {exchange.error}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <p className="animate-pulse text-sm text-gray-500 dark:text-gray-400">
            Yanıt hazırlanıyor…
          </p>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="flex items-end gap-2"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          placeholder="Sorunuzu yazın…"
          className="flex-1 resize-none rounded-lg border border-gray-300 bg-transparent px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-700"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Sor
        </button>
      </form>
    </div>
  );
}
