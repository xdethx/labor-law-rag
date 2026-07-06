"""Hybrid retrieval over the `law` collection (M3) + optional reranking (M4).

One bge-m3 forward pass embeds the query as dense + sparse; the Qdrant Query
API runs both prefetch branches (dense top-k, sparse top-k) and fuses them
server-side with Reciprocal Rank Fusion. RRF over score mixing because dense
cosine scores (bounded) and sparse scores (unbounded) live on different
scales (ARCHITECTURE §2.3).

With RERANK_ENABLED, the fused candidate pool (top-50) is re-scored by the
bge-reranker-v2-m3 cross-encoder and cut to the final top-k (ARCHITECTURE §2.4).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from src.config import DENSE_VECTOR_NAME, LAW_COLLECTION, SPARSE_VECTOR_NAME, settings
from src.embeddings import embed_query_hybrid, rerank_scores, to_indices_values

# Stamped into eval results so runs are comparable across milestones.
RETRIEVAL_MODE = "hybrid-rrf-rerank" if settings.rerank_enabled else "hybrid-rrf"

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
    """Embed `question` (dense + sparse) and return the top-k law article
    payloads, each with a `score` key added. `score` is the RRF fusion
    score (rank-based) — or, with RERANK_ENABLED, the cross-encoder
    relevance score (unbounded logit)."""
    top_k = top_k or settings.top_k_final
    # Reranking needs a wider candidate pool: fetch top-50 fused (and widen the
    # prefetch branches to match — 20+20 could never fill a 50-deep pool).
    fetch_k = max(settings.top_k_rerank_candidates, top_k) if settings.rerank_enabled else top_k
    dense_k = max(settings.top_k_dense, fetch_k) if settings.rerank_enabled else settings.top_k_dense
    sparse_k = max(settings.top_k_sparse, fetch_k) if settings.rerank_enabled else settings.top_k_sparse
    dense, weights = embed_query_hybrid(question)
    indices, values = to_indices_values(weights)
    results = get_client().query_points(
        collection_name=LAW_COLLECTION,
        prefetch=[
            Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=dense_k),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR_NAME,
                limit=sparse_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_k,
        with_payload=True,
    )
    chunks = [{**point.payload, "score": point.score} for point in results.points]
    if settings.rerank_enabled:
        chunks = _rerank(question, chunks, top_k)
    return chunks


def _rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score `chunks` with the cross-encoder and return the top-k,
    with `score` replaced by the reranker relevance score."""
    if not chunks:
        return chunks
    scores = rerank_scores(question, [chunk["text"] for chunk in chunks])
    rescored = [{**chunk, "score": score} for chunk, score in zip(chunks, scores)]
    rescored.sort(key=lambda chunk: chunk["score"], reverse=True)
    return rescored[:top_k]
