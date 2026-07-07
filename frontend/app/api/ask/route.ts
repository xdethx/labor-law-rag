import { proxyToRag } from "@/lib/rag";

export async function POST(request: Request) {
  return proxyToRag("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });
}
