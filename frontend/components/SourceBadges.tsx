import type { ContractSource, LawSource } from "@/lib/types";

const TYPE_LABELS: Record<string, string> = {
  madde: "Madde",
  gecici: "Geçici Madde",
  ek: "Ek Madde",
};

export function SourceBadges({
  sources,
  contractSources = [],
}: {
  sources: LawSource[];
  contractSources?: ContractSource[];
}) {
  if (sources.length === 0 && contractSources.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {sources.map((s) => (
        <span
          key={`${s.article_type}-${s.article_no}`}
          title={s.article_title ?? undefined}
          className="inline-flex items-center rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-950 dark:text-blue-200"
        >
          {TYPE_LABELS[s.article_type] ?? "Madde"} {s.article_no}
          {s.repealed && <span className="ml-1 opacity-70">(mülga)</span>}
        </span>
      ))}
      {contractSources.map((c) => (
        <span
          key={`clause-${c.clause_no}`}
          title={c.text}
          className="inline-flex items-center rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-medium text-purple-800 dark:bg-purple-950 dark:text-purple-200"
        >
          Sözleşme {c.clause_no}
        </span>
      ))}
    </div>
  );
}
