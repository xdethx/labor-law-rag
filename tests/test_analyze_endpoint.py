"""POST /analyze endpoint tests: auth, 404 on unknown session, and the
aggregated report shape (summary counts + disclaimer). get_session_clauses
and analyze_contract are monkeypatched — offline, no Qdrant, no LLM.
"""

import pytest
from fastapi.testclient import TestClient

from src import main
from src.config import DISCLAIMER, settings

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setattr(settings, "rag_api_key", "test-key")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    # This module posts to /analyze (3/minute) more than 3 times total;
    # reset slowapi's in-memory counter so tests don't 429 each other.
    main.limiter.reset()


AUTH = {"Authorization": "Bearer test-key"}

_CLAUSES = [
    {"session_id": "abc", "clause_no": 1, "text": "Deneme süresi dört aydır."},
    {"session_id": "abc", "clause_no": 2, "text": "Taraflar iyi niyetle davranır."},
]

_RESULTS = [
    {
        "clause_no": 1,
        "clause_text": "Deneme süresi dört aydır.",
        "verdict": "conflicts",
        "related_articles": [{"article_no": 15, "why": "deneme süresi en çok iki ay"}],
        "explanation": "Dört aylık deneme süresi Madde 15'teki sınıra aykırı.",
    },
    {
        "clause_no": 2,
        "clause_text": "Taraflar iyi niyetle davranır.",
        "verdict": "not_addressed",
        "related_articles": [],
        "explanation": "Bağlamdaki maddeler bu hükmü düzenlemiyor.",
    },
]


def test_analyze_requires_api_key():
    resp = client.post("/analyze", json={"session_id": "abc"})
    assert resp.status_code == 401


def test_analyze_unknown_session_is_404(monkeypatch):
    monkeypatch.setattr(main, "get_session_clauses", lambda session_id: [])

    resp = client.post("/analyze", json={"session_id": "does-not-exist"}, headers=AUTH)

    assert resp.status_code == 404
    assert "no contract" in resp.json()["detail"]


def test_analyze_returns_report_with_summary_and_disclaimer(monkeypatch):
    monkeypatch.setattr(main, "get_session_clauses", lambda session_id: _CLAUSES)
    monkeypatch.setattr(main, "analyze_contract", lambda clauses: _RESULTS)

    resp = client.post("/analyze", json={"session_id": "abc"}, headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "abc"
    assert body["clause_count"] == 2
    assert [c["verdict"] for c in body["clauses"]] == ["conflicts", "not_addressed"]
    assert body["clauses"][0]["related_articles"] == [
        {"article_no": 15, "why": "deneme süresi en çok iki ay"}
    ]
    assert body["clauses"][0]["error_reason"] is None
    assert body["summary"] == {
        "compliant": 0,
        "risky": 0,
        "conflicts": 1,
        "not_addressed": 1,
        "error": 0,
    }
    # summary counts add up to clause_count
    assert sum(body["summary"].values()) == body["clause_count"]
    assert body["disclaimer"] == DISCLAIMER


def test_analyze_error_verdict_carries_error_reason(monkeypatch):
    failed = [
        {
            "clause_no": 1,
            "clause_text": "x",
            "verdict": "error",
            "related_articles": [],
            "explanation": "Bu madde analiz edilemedi.",
            "error_reason": "rate_limited",
        }
    ]
    monkeypatch.setattr(main, "get_session_clauses", lambda session_id: _CLAUSES[:1])
    monkeypatch.setattr(main, "analyze_contract", lambda clauses: failed)

    resp = client.post("/analyze", json={"session_id": "abc"}, headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["error"] == 1
    assert body["clauses"][0]["error_reason"] == "rate_limited"
