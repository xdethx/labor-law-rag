"""Offline tests for the flag-gated rerank step (M4).

No model load, no qdrant, no network: the cross-encoder is monkeypatched and
the qdrant client is faked, so the tests exercise the gating and ordering
logic only.
"""

from types import SimpleNamespace

import pytest

from src import retrieval
from src.config import settings
from src.retrieval import _rerank, retrieve_law


def _chunk(article_id: str, text: str) -> dict:
    return {"article_id": article_id, "text": text, "score": 0.5}


# --- _rerank ordering --------------------------------------------------------

def test_rerank_reorders_by_score_and_truncates(monkeypatch):
    chunks = [_chunk("madde-1", "a"), _chunk("madde-2", "b"), _chunk("madde-3", "c")]
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts: [0.1, 9.0, 5.0])

    result = _rerank("soru", chunks, top_k=2)

    assert [c["article_id"] for c in result] == ["madde-2", "madde-3"]
    assert [c["score"] for c in result] == [9.0, 5.0]


def test_rerank_empty_chunks_no_scorer_call(monkeypatch):
    def boom(q, texts):
        raise AssertionError("reranker must not be called for empty input")

    monkeypatch.setattr(retrieval, "rerank_scores", boom)
    assert _rerank("soru", [], top_k=5) == []


# --- flag gating in retrieve_law ---------------------------------------------

class FakeClient:
    def __init__(self):
        self.last_limit = None

    def query_points(self, *, limit, **kwargs):
        self.last_limit = limit
        points = [
            SimpleNamespace(payload={"article_id": f"madde-{i}", "text": f"text {i}"}, score=1.0 / i)
            for i in range(1, min(limit, 10) + 1)
        ]
        return SimpleNamespace(points=points)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(retrieval, "get_client", lambda: client)
    monkeypatch.setattr(retrieval, "embed_query_hybrid", lambda q: ([0.0], {"1": 0.5}))
    return client


def test_flag_off_no_rerank_and_narrow_fetch(monkeypatch, fake_client):
    monkeypatch.setattr(settings, "rerank_enabled", False)

    def boom(q, texts):
        raise AssertionError("reranker must not be called when RERANK_ENABLED=false")

    monkeypatch.setattr(retrieval, "rerank_scores", boom)
    chunks = retrieve_law("soru", top_k=5)

    assert fake_client.last_limit == 5
    assert len(chunks) == 5
    # RRF order untouched.
    assert [c["article_id"] for c in chunks] == [f"madde-{i}" for i in range(1, 6)]


def test_flag_on_wide_fetch_then_rerank(monkeypatch, fake_client):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    # Score inversely: the LAST fused candidate is the most relevant.
    monkeypatch.setattr(
        retrieval, "rerank_scores", lambda q, texts: list(range(len(texts)))
    )
    chunks = retrieve_law("soru", top_k=3)

    assert fake_client.last_limit == settings.top_k_rerank_candidates
    assert len(chunks) == 3
    assert chunks[0]["article_id"] == "madde-10"  # deepest candidate won
    assert chunks[0]["score"] == 9.0  # reranker score replaced the RRF score
