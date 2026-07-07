import { proxyToRag } from "@/lib/rag";

// Upstream parses the PDF and embeds every clause in one batch (CPU bge-m3).
// Comfortably under 60s even for a 20-page contract, but set explicitly
// rather than relying on the platform default (see analyze/route.ts).
export const maxDuration = 60;

export async function POST(request: Request) {
  // Re-sending the parsed FormData lets fetch set the multipart boundary.
  return proxyToRag("/contracts", {
    method: "POST",
    body: await request.formData(),
  });
}
