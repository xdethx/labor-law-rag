import { proxyToRag } from "@/lib/rag";

export async function POST(request: Request) {
  // Re-sending the parsed FormData lets fetch set the multipart boundary.
  return proxyToRag("/contracts", {
    method: "POST",
    body: await request.formData(),
  });
}
