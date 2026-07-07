"""Global daily request cap (M8 public-demo backstop) on POST /ask.

retrieve_law and llm.generate are monkeypatched so no model/network is
touched. The counter is an in-memory module-level dict, reset per test.
"""

import pytest
from fastapi.testclient import TestClient

from src import main
from src.config import settings

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_law",
        lambda question, top_k=None: [
            {
                "article_no": 1,
                "article_type": "madde",
                "article_title": "Amaç",
                "repealed": False,
                "text": "MADDE 1. - ...",
                "score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(main.llm, "generate", lambda system, user: "cevap [Madde 1]")
    monkeypatch.setattr(settings, "rag_api_key", "test-key")
    main._daily_usage["date"] = ""
    main._daily_usage["count"] = 0


def _ask():
    return client.post("/ask", json={"question": "q"}, headers={"Authorization": "Bearer test-key"})


def test_cap_disabled_by_default():
    monkeypatch_cap = settings.daily_request_cap
    assert monkeypatch_cap == 0
    for _ in range(5):
        assert _ask().status_code == 200


def test_cap_blocks_once_budget_spent(monkeypatch):
    monkeypatch.setattr(settings, "daily_request_cap", 2)
    assert _ask().status_code == 200
    assert _ask().status_code == 200
    resp = _ask()
    assert resp.status_code == 429


def test_cap_resets_on_date_rollover(monkeypatch):
    monkeypatch.setattr(settings, "daily_request_cap", 1)
    assert _ask().status_code == 200
    assert _ask().status_code == 429

    main._daily_usage["date"] = "2000-01-01"
    assert _ask().status_code == 200
