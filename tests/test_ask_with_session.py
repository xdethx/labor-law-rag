"""POST /ask with session_id: retrieves law + that session's contract
clauses, returns contract_sources, and cites [Sözleşme n] in the prompt sent
to the LLM. retrieve_law/retrieve_contract/llm.generate are monkeypatched.
"""

import pytest
from fastapi.testclient import TestClient

from src import main
from src.config import settings

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setattr(settings, "rag_api_key", "test-key")


AUTH = {"Authorization": "Bearer test-key"}

_LAW_CHUNK = {
    "article_no": 15,
    "article_type": "madde",
    "article_title": "Deneme süreli iş sözleşmesi",
    "repealed": False,
    "text": "MADDE 15. - Deneme süresi en çok iki ay olabilir.",
    "score": 0.9,
}

_CONTRACT_CHUNK = {
    "session_id": "abc",
    "clause_no": 4,
    "text": "Deneme süresi dört aydır.",
    "score": 0.8,
    "source": "contract",
}


def _stub_pipeline(monkeypatch, captured_prompt):
    monkeypatch.setattr(main, "retrieve_law", lambda question, top_k=None: [_LAW_CHUNK])
    monkeypatch.setattr(
        main, "retrieve_contract", lambda question, session_id, top_k=None: [_CONTRACT_CHUNK]
    )

    def _fake_generate(system, user):
        captured_prompt["system"] = system
        captured_prompt["user"] = user
        return "Sözleşmenizdeki [Sözleşme 4] deneme süresi, [Madde 15] uyarınca kanuna aykırıdır."

    monkeypatch.setattr(main.llm, "generate", _fake_generate)


def test_ask_with_session_id_returns_contract_sources(monkeypatch):
    captured: dict = {}
    _stub_pipeline(monkeypatch, captured)

    resp = client.post(
        "/ask",
        json={"question": "deneme süresi kanuna uygun mu?", "session_id": "abc"},
        headers=AUTH,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["contract_sources"] == [{"clause_no": 4, "text": "Deneme süresi dört aydır.", "score": 0.8}]
    assert body["sources"][0]["article_no"] == 15
    assert "[Sözleşme 4]" in body["answer"]
    assert "[Sözleşme 4]" in captured["user"]
    assert "[Madde 15]" in captured["user"]


def test_ask_without_session_id_has_empty_contract_sources(monkeypatch):
    captured: dict = {}
    _stub_pipeline(monkeypatch, captured)

    resp = client.post(
        "/ask", json={"question": "deneme süresi nedir?"}, headers=AUTH
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["contract_sources"] == []
    assert "[Sözleşme" not in captured["user"]
