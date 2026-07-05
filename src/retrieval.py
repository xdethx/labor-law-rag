"""Dense-only retrieval over the `law` collection (M1 baseline).

Hybrid (dense + sparse, RRF-fused) arrives at M3. This module intentionally
does one thing: embed the query, run a single dense kNN search, return
payloads + scores.
"""

from qdrant_client import QdrantClient

from src.config import DENSE_VECTOR_NAME, LAW_COLLECTION, settings
from src.embeddings import embed_query

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Lazy singleton Qdrant client. Docker URL by default; falls back to
    embedded local mode if QDRANT_URL is empty and QDRANT_PATH is set."""
    global _client
    if _client is None:
        if settings.qdrant_url:
            _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        elif settings.qdrant_path:
            _client = QdrantClient(path=settings.qdrant_path)
        else:
            raise RuntimeError("Set QDRANT_URL (docker) or QDRANT_PATH (embedded) in .env")
    return _client


def retrieve_law(question: str, top_k: int | None = None) -> list[dict]:
    """Embed `question` and return the top-k law article payloads, each with
    a `score` key added, highest similarity first."""
    top_k = top_k or settings.top_k_final
    vector = embed_query(question)
    results = get_client().query_points(
        collection_name=LAW_COLLECTION,
        query=vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in results.points]
