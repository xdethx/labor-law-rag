// TypeScript mirrors of the backend pydantic response models (src/main.py).

export interface LawSource {
  article_no: number;
  article_type: string; // madde | gecici | ek
  article_title: string | null;
  repealed: boolean;
  score: number;
}

export interface ContractSource {
  clause_no: number;
  text: string;
  score: number;
}

export interface AskResponse {
  answer: string;
  sources: LawSource[];
  contract_sources: ContractSource[];
  disclaimer: string;
}

export interface ContractUploadResponse {
  session_id: string;
  clause_count: number;
}

export type Verdict = "compliant" | "risky" | "conflicts" | "not_addressed" | "error";

export type ErrorReason = "rate_limited" | "invalid_model_output" | "provider_error";

export interface RelatedArticle {
  article_no: number;
  why: string;
}

export interface ClauseAnalysis {
  clause_no: number;
  clause_text: string;
  verdict: Verdict;
  related_articles: RelatedArticle[];
  explanation: string;
  error_reason: ErrorReason | null;
}

export interface AnalyzeSummary {
  compliant: number;
  risky: number;
  conflicts: number;
  not_addressed: number;
  error: number;
}

export interface AnalyzeResponse {
  session_id: string;
  clause_count: number;
  clauses: ClauseAnalysis[];
  summary: AnalyzeSummary;
  disclaimer: string;
}
