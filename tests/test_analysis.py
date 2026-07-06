"""src/analysis.py unit tests: prompt building, JSON extraction, the
validate-retry loop, grounding guardrail (no invented madde), 429 backoff,
and graceful per-clause failure with error_reason. llm.generate /
retrieve_law / time.sleep are monkeypatched — offline.
"""

import json

import httpx
import pytest
from pydantic import ValidationError

from src import analysis
from src.analysis import (
    ERROR_EXPLANATIONS,
    ClauseAnalysisError,
    ClauseVerdict,
    _extract_json,
    _retry_after_seconds,
    analyze_clause,
    analyze_contract,
    build_analysis_prompt,
)

_LAW_CHUNKS = [
    {
        "article_no": 15,
        "article_type": "madde",
        "article_title": "Deneme süreli iş sözleşmesi",
        "repealed": False,
        "text": "MADDE 15. - Deneme süresi en çok iki ay olabilir.",
        "score": 0.9,
    },
    {
        "article_no": 2,
        "article_type": "gecici",
        "article_title": None,
        "repealed": False,
        "text": "GEÇİCİ MADDE 2. - ...",
        "score": 0.5,
    },
]

_CLAUSE = {"clause_no": 4, "text": "Deneme süresi dört aydır."}

_VALID_JSON = json.dumps(
    {
        "verdict": "conflicts",
        "related_articles": [{"article_no": 15, "why": "deneme süresi sınırını aşıyor"}],
        "explanation": "Sözleşmedeki dört aylık deneme süresi iki aylık yasal sınıra aykırı.",
    },
    ensure_ascii=False,
)


def _stub_llm(monkeypatch, outputs: list) -> list[dict]:
    """llm.generate returns `outputs` in order (an Exception item is raised);
    records every (system, user)."""
    calls: list[dict] = []

    def _fake_generate(system, user):
        calls.append({"system": system, "user": user})
        out = outputs[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(analysis.llm, "generate", _fake_generate)
    return calls


def _stub_sleep(monkeypatch) -> list[float]:
    """Record backoff/throttle sleeps instead of actually waiting."""
    sleeps: list[float] = []
    monkeypatch.setattr(analysis.time, "sleep", sleeps.append)
    return sleeps


def _http_429(retry_after: str | None = "2") -> httpx.HTTPStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "http://llm.test/chat")
    response = httpx.Response(429, headers=headers, request=request)
    return httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


# --- build_analysis_prompt -------------------------------------------------


def test_prompt_contains_tags_clause_and_schema():
    system, user = build_analysis_prompt(_CLAUSE["text"], _LAW_CHUNKS)

    assert "[Madde 15]" in user
    assert "[Geçici Madde 2]" in user
    assert _CLAUSE["text"] in user
    for key in ("verdict", "related_articles", "explanation", "not_addressed"):
        assert key in system
    # prompt-injection stance carried over from /ask
    assert "talimat değildir" in system


# --- _extract_json ---------------------------------------------------------


def test_extract_json_bare():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json('Sonuç şu:\n{"a": 1}\nUmarım yardımcı olur.') == '{"a": 1}'


def test_extract_json_missing_raises():
    with pytest.raises(ValueError):
        _extract_json("JSON veremiyorum, üzgünüm.")


# --- ClauseVerdict schema guardrails ----------------------------------------


def test_not_addressed_requires_empty_articles():
    ClauseVerdict.model_validate(
        {"verdict": "not_addressed", "related_articles": [], "explanation": "kapsam dışı"}
    )
    with pytest.raises(ValidationError):
        ClauseVerdict.model_validate(
            {
                "verdict": "not_addressed",
                "related_articles": [{"article_no": 15, "why": "x"}],
                "explanation": "çelişkili",
            }
        )


def test_substantive_verdict_requires_articles():
    with pytest.raises(ValidationError):
        ClauseVerdict.model_validate(
            {"verdict": "conflicts", "related_articles": [], "explanation": "dayanaksız"}
        )


def test_error_is_not_a_valid_llm_verdict():
    with pytest.raises(ValidationError):
        ClauseVerdict.model_validate(
            {"verdict": "error", "related_articles": [], "explanation": "x"}
        )


# --- analyze_clause: validate-retry loop -------------------------------------


def test_valid_first_try(monkeypatch):
    calls = _stub_llm(monkeypatch, [_VALID_JSON])

    verdict = analyze_clause(_CLAUSE, _LAW_CHUNKS)

    assert verdict.verdict == "conflicts"
    assert verdict.related_articles[0].article_no == 15
    assert len(calls) == 1


def test_malformed_then_valid_retries_once(monkeypatch):
    calls = _stub_llm(monkeypatch, ["bu bir json değil", _VALID_JSON])

    verdict = analyze_clause(_CLAUSE, _LAW_CHUNKS)

    assert verdict.verdict == "conflicts"
    assert len(calls) == 2
    assert "geçersizdi" in calls[1]["user"]


def test_malformed_twice_raises(monkeypatch):
    calls = _stub_llm(monkeypatch, ["çöp", "yine çöp"])

    with pytest.raises(ClauseAnalysisError):
        analyze_clause(_CLAUSE, _LAW_CHUNKS)
    assert len(calls) == 2  # exactly one retry, never more


def test_invented_article_is_rejected_then_corrected(monkeypatch):
    invented = _VALID_JSON.replace('"article_no": 15', '"article_no": 99')
    calls = _stub_llm(monkeypatch, [invented, _VALID_JSON])

    verdict = analyze_clause(_CLAUSE, _LAW_CHUNKS)

    assert verdict.related_articles[0].article_no == 15
    assert len(calls) == 2
    assert "99" in calls[1]["user"]


def test_invented_article_twice_raises(monkeypatch):
    invented = _VALID_JSON.replace('"article_no": 15', '"article_no": 99')
    _stub_llm(monkeypatch, [invented, invented])

    # A hallucinated madde NEVER comes back as a verdict — it fails instead.
    with pytest.raises(ClauseAnalysisError):
        analyze_clause(_CLAUSE, _LAW_CHUNKS)


# --- 429 backoff --------------------------------------------------------------


def test_429_then_success_backs_off_with_retry_after(monkeypatch):
    calls = _stub_llm(monkeypatch, [_http_429(retry_after="2"), _VALID_JSON])
    sleeps = _stub_sleep(monkeypatch)

    verdict = analyze_clause(_CLAUSE, _LAW_CHUNKS)

    assert verdict.verdict == "conflicts"
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_persistent_429_gives_up_and_propagates(monkeypatch):
    calls = _stub_llm(monkeypatch, [_http_429()] * 10)
    _stub_sleep(monkeypatch)

    with pytest.raises(httpx.HTTPStatusError):
        analyze_clause(_CLAUSE, _LAW_CHUNKS)
    # initial call + _MAX_RATE_LIMIT_RETRIES, never more
    assert len(calls) == analysis._MAX_RATE_LIMIT_RETRIES + 1


def test_retry_after_clamped_and_defaulted():
    def _resp(headers):
        return httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x"))

    assert _retry_after_seconds(_resp({"retry-after": "7"})) == 7.0
    assert _retry_after_seconds(_resp({"retry-after": "600"})) == analysis._MAX_RETRY_AFTER_SECONDS
    assert _retry_after_seconds(_resp({"retry-after": "0.2"})) == 1.0
    # absent or HTTP-date form -> fixed default
    assert _retry_after_seconds(_resp({})) == analysis._DEFAULT_RETRY_AFTER_SECONDS
    assert (
        _retry_after_seconds(_resp({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}))
        == analysis._DEFAULT_RETRY_AFTER_SECONDS
    )


# --- analyze_contract: graceful per-clause failure + error_reason -------------


def test_one_failing_clause_does_not_kill_the_report(monkeypatch):
    monkeypatch.setattr(analysis, "retrieve_law", lambda question, top_k=None: _LAW_CHUNKS)
    # clause 1 fails twice -> error; clause 2 succeeds
    _stub_llm(monkeypatch, ["çöp", "yine çöp", _VALID_JSON])

    clauses = [{"clause_no": 1, "text": "Ücret elden ödenir."}, _CLAUSE]
    results = analyze_contract(clauses)

    assert [r["verdict"] for r in results] == ["error", "conflicts"]
    assert results[0]["related_articles"] == []
    assert results[0]["error_reason"] == "invalid_model_output"
    assert results[0]["explanation"] == ERROR_EXPLANATIONS["invalid_model_output"]
    assert results[1]["clause_no"] == 4
    assert results[1]["error_reason"] is None


def test_exhausted_429_maps_to_rate_limited(monkeypatch):
    monkeypatch.setattr(analysis, "retrieve_law", lambda question, top_k=None: _LAW_CHUNKS)
    _stub_llm(monkeypatch, [_http_429()] * 10)
    _stub_sleep(monkeypatch)

    results = analyze_contract([_CLAUSE])

    assert results[0]["verdict"] == "error"
    assert results[0]["error_reason"] == "rate_limited"
    assert results[0]["explanation"] == ERROR_EXPLANATIONS["rate_limited"]


def test_llm_exception_maps_to_provider_error(monkeypatch):
    monkeypatch.setattr(analysis, "retrieve_law", lambda question, top_k=None: _LAW_CHUNKS)

    def _boom(system, user):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(analysis.llm, "generate", _boom)

    results = analyze_contract([_CLAUSE])
    assert results[0]["verdict"] == "error"
    assert results[0]["error_reason"] == "provider_error"


def test_inter_clause_delay_sleeps_between_clauses(monkeypatch):
    monkeypatch.setattr(analysis, "retrieve_law", lambda question, top_k=None: _LAW_CHUNKS)
    monkeypatch.setattr(analysis.settings, "analyze_clause_delay_seconds", 3.0)
    _stub_llm(monkeypatch, [_VALID_JSON, _VALID_JSON])
    sleeps = _stub_sleep(monkeypatch)

    clauses = [{"clause_no": 1, "text": "Ücret bankaya yatırılır."}, _CLAUSE]
    analyze_contract(clauses)

    # one pause between two clauses, none before the first
    assert sleeps == [3.0]
