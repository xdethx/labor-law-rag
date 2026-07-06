"""Hybrid retrieval over the `law` collection (M3) + optional reranking (M4),
and the per-session `contracts` collection (M5).

One bge-m3 forward pass embeds the query as dense + sparse; the Qdrant Query
API runs both prefetch branches (dense top-k, sparse top-k) and fuses them
server-side with Reciprocal Rank Fusion. RRF over score mixing because dense
cosine scores (bounded) and sparse scores (unbounded) live on different
scales (ARCHITECTURE §2.3). `_hybrid_search` is the shared plumbing for both
collections; `retrieve_law` adds reranking, `retrieve_contract` adds the
session_id filter.

With RERANK_ENABLED, the fused candidate pool (top-50) is re-scored by the
bge-reranker-v2-m3 cross-encoder and cut to the final top-k (ARCHITECTURE §2.4).

Contract corpus lifecycle (ARCHITECTURE §3.2): one `contracts` collection,
multitenant via a `session_id` payload filter (never per-user collections).
Points are ephemeral — deleted on explicit DELETE or swept past
CONTRACT_TTL_HOURS at startup.
"""

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import Filter as QdrantFilter

from src.config import (
    CONTRACTS_COLLECTION,
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    LAW_COLLECTION,
    SPARSE_VECTOR_NAME,
    settings,
)
from src.embeddings import embed_hybrid, embed_query_hybrid, rerank_scores, to_indices_values

# Constant namespace for deriving contract point ids from (session_id, clause_no)
# — same pattern as src/ingest.py's article point ids.
_CONTRACT_POINT_NAMESPACE = uuid.NAMESPACE_DNS

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


def _hybrid_search(
    collection: str,
    question: str,
    limit: int,
    dense_k: int,
    sparse_k: int,
    query_filter: QdrantFilter | None = None,
) -> list[dict]:
    """Shared dense+sparse RRF query, used by both `law` and `contracts`.

    `query_filter` (e.g. a session_id match) is applied to BOTH prefetch
    branches so session isolation holds regardless of which branch a point
    is fused from.
    """
    dense, weights = embed_query_hybrid(question)
    indices, values = to_indices_values(weights)
    results = get_client().query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=dense_k, filter=query_filter),
            Prefetch(
                query=SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR_NAME,
                limit=sparse_k,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in results.points]


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
    chunks = _hybrid_search(LAW_COLLECTION, question, fetch_k, dense_k, sparse_k)
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


def ensure_contracts_collection(client: QdrantClient) -> None:
    """Create the `contracts` collection if absent (idempotent). Same named
    vectors as `law`, plus payload indexes for session filtering and the
    TTL sweep's range query."""
    if client.collection_exists(CONTRACTS_COLLECTION):
        return
    client.create_collection(
        collection_name=CONTRACTS_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )
    client.create_payload_index(
        CONTRACTS_COLLECTION, field_name="session_id", field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        CONTRACTS_COLLECTION, field_name="uploaded_at", field_schema=PayloadSchemaType.FLOAT
    )


def upsert_contract_clauses(session_id: str, clauses: list[dict]) -> int:
    """Embed clause texts (dense + sparse) and upsert into `contracts`.

    Point ids are uuid5(namespace, f"{session_id}-{clause_no}") so a
    same-session re-upload overwrites cleanly. Returns the clause count.
    """
    if not clauses:
        return 0
    client = get_client()
    ensure_contracts_collection(client)

    dense_vectors, sparse_weights = embed_hybrid([c["text"] for c in clauses])
    uploaded_at = time.time()

    points = []
    for clause, dense, weights in zip(clauses, dense_vectors, sparse_weights):
        indices, values = to_indices_values(weights)
        point_id = str(
            uuid.uuid5(_CONTRACT_POINT_NAMESPACE, f"{session_id}-{clause['clause_no']}")
        )
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values),
                },
                payload={
                    "session_id": session_id,
                    "clause_no": clause["clause_no"],
                    "text": clause["text"],
                    "uploaded_at": uploaded_at,
                    "source": "contract",
                },
            )
        )
    client.upsert(collection_name=CONTRACTS_COLLECTION, points=points)
    return len(points)


def retrieve_contract(question: str, session_id: str, top_k: int | None = None) -> list[dict]:
    """Hybrid-retrieve this session's contract clauses only. No reranking —
    the candidate pool per session is already small (tens of clauses)."""
    top_k = top_k or settings.top_k_contract
    query_filter = QdrantFilter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )
    return _hybrid_search(
        CONTRACTS_COLLECTION,
        question,
        top_k,
        settings.top_k_dense,
        settings.top_k_sparse,
        query_filter=query_filter,
    )


def get_session_clauses(session_id: str) -> list[dict]:
    """Return ALL clause payloads of a session, sorted by clause_no (M6:
    /analyze iterates over the full contract, not a retrieved subset).
    Empty list if the collection or the session doesn't exist."""
    client = get_client()
    if not client.collection_exists(CONTRACTS_COLLECTION):
        return []

    session_filter = QdrantFilter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )
    clauses: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=CONTRACTS_COLLECTION,
            scroll_filter=session_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        clauses.extend(point.payload for point in points)
        if offset is None:
            break
    clauses.sort(key=lambda clause: clause["clause_no"])
    return clauses


def delete_contract(session_id: str) -> None:
    """Delete all points for a session. Idempotent: deleting an unknown or
    already-deleted session is a no-op."""
    client = get_client()
    if not client.collection_exists(CONTRACTS_COLLECTION):
        return
    client.delete(
        collection_name=CONTRACTS_COLLECTION,
        points_selector=FilterSelector(
            filter=QdrantFilter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )
        ),
    )


def sweep_expired_contracts(client: QdrantClient) -> int:
    """Delete contract points older than CONTRACT_TTL_HOURS. Runs once at
    startup; a scheduled recurring job is deferred (see ROADMAP)."""
    if not client.collection_exists(CONTRACTS_COLLECTION):
        return 0
    cutoff = time.time() - settings.contract_ttl_hours * 3600
    before = client.get_collection(CONTRACTS_COLLECTION).points_count
    client.delete(
        collection_name=CONTRACTS_COLLECTION,
        points_selector=FilterSelector(
            filter=QdrantFilter(must=[FieldCondition(key="uploaded_at", range=Range(lt=cutoff))])
        ),
    )
    after = client.get_collection(CONTRACTS_COLLECTION).points_count
    return before - after
