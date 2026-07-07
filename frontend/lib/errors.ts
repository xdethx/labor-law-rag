// Turns a non-OK /api response into a user-facing message.

export async function errorDetail(res: Response): Promise<string> {
  if (res.status === 429) {
    return "Hız sınırına takıldınız — lütfen bir dakika bekleyip yeniden deneyin.";
  }

  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    // non-JSON body; fall through to the generic message
  }
  const record = (data ?? {}) as Record<string, unknown>;
  // FastAPI errors carry `detail`; slowapi's handler uses `error`.
  const detail =
    typeof record.detail === "string"
      ? record.detail
      : typeof record.error === "string"
        ? record.error
        : null;
  return detail ?? `Beklenmeyen bir hata oluştu (HTTP ${res.status}).`;
}
