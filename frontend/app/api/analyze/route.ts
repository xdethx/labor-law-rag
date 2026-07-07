import { proxyToRag } from "@/lib/rag";

// Long-running upstream call: one LLM call per contract clause, ~20-90s for
// a typical contract. Vercel Hobby's fluid-compute ceiling is 300s (verified
// against the installed Next.js docs, see maxDuration.md) — comfortable
// margin over the worst case.
export const maxDuration = 300;

export async function POST(request: Request) {
  return proxyToRag("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });
}
