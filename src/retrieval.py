"""Hybrid retrieval over the `law` collection (M3).

One bge-m3 forward pass embeds the query as dense + sparse; the Qdrant Query
API runs both prefetch branches (dense top-k, sparse top-k) and fuses them
server-side with Reciprocal Rank Fusion. RRF over score mixing because dense
cosine scores (bounded) and sparse scores (unbounded) live on different
scales (ARCHITECTURE §2.3).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from src.config import DENSE_VECTOR_NAME, LAW_COLLECTION, SPARSE_VECTOR_NAME, settings
from src.embeddings import embed_query_hybrid, to_indices_values

# Stamped into eval results so runs are comparable across milestones.
RETRIEVAL_MODE = "hybrid-rrf"

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
    score (rank-based), not a cosine similarity."""
    top_k = top_k or settings.top_k_final
    dense, weights = embed_query_hybrid(question)
    indices, values = to_indices_values(weights)
    results = get_client().query_points(
        collection_name=LAW_COLLECTION,
        prefetch=[
            Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=settings.top_k_dense),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR_NAME,
                limit=settings.top_k_sparse,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in results.points]
