"""Bearer auth on POST /ask: fail-closed, constant-time compare.

retrieve_law and llm.generate are monkeypatched so no model/network is touched.
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


def test_missing_auth_header_rejected():
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 401


def test_wrong_key_rejected():
    resp = client.post("/ask", json={"question": "q"}, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_correct_key_accepted():
    resp = client.post("/ask", json={"question": "q"}, headers={"Authorization": "Bearer test-key"})
    assert resp.status_code == 200


def test_fail_closed_when_server_key_unset(monkeypatch):
    monkeypatch.setattr(settings, "rag_api_key", "")
    resp = client.post("/ask", json={"question": "q"}, headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 500
