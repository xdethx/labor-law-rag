"""Provider-agnostic LLM client (thin httpx, no vendor SDKs).

`LLM_PROVIDER` selects the branch at request time — ported from project 1's
pattern (Groq model deprecations were fixed with one env var, zero code).
Never hardcode a model ID without an env override; verify current IDs at
console.groq.com/docs/models before deploying.
"""

import httpx

from src.config import settings

_TIMEOUT = 30.0


def _call_openai_compatible(system: str, user: str) -> str:
    resp = httpx.post(
        f"{settings.openai_compatible_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_compatible_api_key}"},
        json={
            "model": settings.openai_compatible_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama(system: str, user: str) -> str:
    resp = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_anthropic(system: str, user: str) -> str:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": settings.anthropic_model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_gemini(system: str, user: str) -> str:
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}",
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


_PROVIDERS = {
    "openai_compatible": _call_openai_compatible,
    "ollama": _call_ollama,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


def generate(system: str, user: str) -> str:
    """Call the configured LLM provider; return the plain-text completion.

    Raises httpx.HTTPStatusError on a non-2xx response, KeyError if the
    provider's response shape doesn't match what we expect, or ValueError if
    LLM_PROVIDER is not one of the four supported values.
    """
    try:
        call = _PROVIDERS[settings.llm_provider]
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        ) from None
    return call(system, user)
