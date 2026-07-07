"""POST /contracts and DELETE /contracts/{session_id}: upload hardening,
auth, and the happy path. src.contracts + src.retrieval calls are
monkeypatched so no model/network/Qdrant is touched.
"""

import pytest
from fastapi.testclient import TestClient

from src import main
from src.config import settings
from src.contracts import EncryptedPDFError

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setattr(settings, "rag_api_key", "test-key")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    # This module posts to /contracts (5/minute) more than 5 times total;
    # reset slowapi's in-memory counter so tests don't 429 each other.
    main.limiter.reset()


AUTH = {"Authorization": "Bearer test-key"}


def _pdf_file(name="contract.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake"):
    return {"file": (name, data, content_type)}


def test_upload_rejects_non_pdf_extension():
    resp = client.post("/contracts", headers=AUTH, files=_pdf_file(name="contract.docx"))
    assert resp.status_code == 415


def test_upload_rejects_wrong_content_type():
    resp = client.post(
        "/contracts", headers=AUTH, files=_pdf_file(content_type="application/octet-stream")
    )
    assert resp.status_code == 415


def test_upload_rejects_oversize_file(monkeypatch):
    monkeypatch.setattr(settings, "contract_max_mb", 1)
    oversized = b"0" * (2 * 1024 * 1024)
    resp = client.post("/contracts", headers=AUTH, files=_pdf_file(data=oversized))
    assert resp.status_code == 400


def test_upload_rejects_non_pdf_content_disguised_as_pdf():
    # A .txt renamed to .pdf with a spoofed content-type: extension/MIME check
    # alone can't catch this — the real extract_contract_text (no monkeypatch)
    # must reject the garbage bytes as an invalid PDF, not 500 or accept it.
    resp = client.post(
        "/contracts", headers=AUTH, files=_pdf_file(data=b"just plain text, not a pdf at all")
    )
    assert resp.status_code == 400


def test_upload_rejects_encrypted_pdf(monkeypatch):
    def _raise_encrypted(pdf_bytes):
        raise EncryptedPDFError("encrypted")

    monkeypatch.setattr(main, "extract_contract_text", _raise_encrypted)
    resp = client.post("/contracts", headers=AUTH, files=_pdf_file())
    assert resp.status_code == 400


def test_upload_rejects_too_many_pages(monkeypatch):
    monkeypatch.setattr(main, "extract_contract_text", lambda pdf_bytes: ("some text", 999))
    monkeypatch.setattr(settings, "contract_max_pages", 20)
    resp = client.post("/contracts", headers=AUTH, files=_pdf_file())
    assert resp.status_code == 400


def test_upload_rejects_when_no_clauses_found(monkeypatch):
    monkeypatch.setattr(main, "extract_contract_text", lambda pdf_bytes: ("", 1))
    monkeypatch.setattr(main, "split_clauses", lambda text: [])
    resp = client.post("/contracts", headers=AUTH, files=_pdf_file())
    assert resp.status_code == 422


def test_upload_happy_path_returns_session_and_clause_count(monkeypatch):
    monkeypatch.setattr(main, "extract_contract_text", lambda pdf_bytes: ("clause text", 1))
    monkeypatch.setattr(
        main, "split_clauses", lambda text: [{"clause_no": 1, "text": "clause text"}]
    )
    monkeypatch.setattr(main, "upsert_contract_clauses", lambda session_id, clauses: len(clauses))

    resp = client.post("/contracts", headers=AUTH, files=_pdf_file())

    assert resp.status_code == 200
    body = resp.json()
    assert body["clause_count"] == 1
    assert len(body["session_id"]) > 0


def test_upload_requires_auth():
    resp = client.post("/contracts", files=_pdf_file())
    assert resp.status_code == 401


def test_delete_contract_requires_auth():
    resp = client.delete("/contracts/some-session-id")
    assert resp.status_code == 401


def test_delete_contract_is_idempotent(monkeypatch):
    monkeypatch.setattr(main, "delete_contract", lambda session_id: None)
    resp = client.delete("/contracts/unknown-session-id", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
