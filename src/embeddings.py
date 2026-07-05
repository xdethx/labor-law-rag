"""bge-m3 embedding wrapper.

Dense output only for M1 (the measured baseline). Sparse lexical weights are
part of the same forward pass and arrive at M3 for hybrid retrieval — the
model is loaded once either way, so adding sparse later is cheap.

The SAME model embeds documents (ingest) and queries (retrieval) — this
module is the only place bge-m3 is instantiated, so that invariant holds.
"""

from FlagEmbedding import BGEM3FlagModel

from src.config import EMBEDDING_MODEL

_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        # CPU target (HF Spaces free tier has no GPU) -> fp16 buys nothing.
        _model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)
    return _model


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings for a batch of texts (ingest path)."""
    output = _get_model().encode(
        texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    return [vec.tolist() for vec in output["dense_vecs"]]


def embed_query(text: str) -> list[float]:
    """Dense embedding for a single query string (retrieval path)."""
    return embed_dense([text])[0]


def embed_sparse(texts: list[str]):
    """Sparse lexical-weight embeddings — deferred to M3 (hybrid retrieval)."""
    raise NotImplementedError("sparse embeddings arrive at M3")
