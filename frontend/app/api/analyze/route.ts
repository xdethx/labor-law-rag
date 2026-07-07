import { proxyToRag } from "@/lib/rag";

// Long-running upstream call: one LLM call per contract clause.
export async function POST(request: Request) {
  return proxyToRag("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });
}
