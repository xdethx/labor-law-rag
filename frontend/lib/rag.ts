// Server-side only: imported exclusively by app/api/* route handlers.
// The RAG key stays in process.env — it must never reach a client bundle,
// so nothing here is NEXT_PUBLIC_.

export async function proxyToRag(path: string, init: RequestInit): Promise<Response> {
  const baseUrl = process.env.RAG_API_URL;
  const apiKey = process.env.RAG_API_KEY;
  if (!baseUrl || !apiKey) {
    return Response.json(
      { detail: "Sunucu yapılandırması eksik: RAG_API_URL / RAG_API_KEY (frontend/.env.local)" },
      { status: 500 },
    );
  }

  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${apiKey}`);

  let upstream: Response;
  try {
    upstream = await fetch(new URL(path, baseUrl), { ...init, headers });
  } catch {
    return Response.json({ detail: "RAG servisine ulaşılamıyor" }, { status: 502 });
  }

  // Forward status + body verbatim (FastAPI puts error messages in `detail`,
  // slowapi's 429 handler in `error` — the client reads both).
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
