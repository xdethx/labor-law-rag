"""Grounding contract on POST /ask: sources + disclaimer always returned,
even when the (mocked) LLM refuses to answer from a weak/off-corpus match.
"""

import pytest
from fastapi.testclient import TestClient

from src import main
from src.config import DISCLAIMER, settings

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setattr(settings, "rag_api_key", "test-key")


def test_refusal_path_still_returns_sources_and_disclaimer(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_law",
        lambda question, top_k=None: [
            {
                "article_no": 41,
                "article_type": "madde",
                "article_title": "Fazla çalışma",
                "repealed": False,
                "text": "MADDE 41. - ...",
                "score": 0.42,
            }
        ],
    )
    monkeypatch.setattr(
        main.llm, "generate", lambda system, user: "Bu konu mevcut kapsamda yer almamaktadır."
    )

    resp = client.post(
        "/ask",
        json={"question": "boşanma davası nasıl açılır"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == DISCLAIMER
    assert body["sources"][0]["article_no"] == 41
    assert "mevcut kapsamda" in body["answer"]


def test_grounded_path_cites_retrieved_article(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_law",
        lambda question, top_k=None: [
            {
                "article_no": 32,
                "article_type": "madde",
                "article_title": "Ücretin tanımı",
                "repealed": False,
                "text": "MADDE 32. - ...",
                "score": 0.95,
            }
        ],
    )
    monkeypatch.setattr(main.llm, "generate", lambda system, user: "Cevap [Madde 32] uyarınca ...")

    resp = client.post(
        "/ask",
        json={"question": "ücret nedir"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["article_no"] == 32
    assert "[Madde 32]" in body["answer"]
