import type { Verdict } from "@/lib/types";

export const VERDICT_META: Record<Verdict, { label: string; className: string }> = {
  compliant: {
    label: "Uyumlu",
    className: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-200",
  },
  risky: {
    label: "Riskli",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200",
  },
  conflicts: {
    label: "Kanuna aykırı",
    className: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
  },
  not_addressed: {
    label: "Düzenleme bulunamadı",
    className: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  },
  error: {
    label: "Analiz edilemedi",
    className: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  },
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const meta = VERDICT_META[verdict];
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${meta.className}`}
    >
      {meta.label}
    </span>
  );
}
