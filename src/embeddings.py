"""bge-m3 embedding wrapper — dense + sparse in one forward pass (M3).

bge-m3 emits dense (1024-d) and sparse lexical-weight vectors from the same
encode call, which is what makes hybrid retrieval possible with one model.

The SAME model embeds documents (ingest) and queries (retrieval) — this
module is the only place bge-m3 is instantiated, so that invariant holds.
This module stays qdrant-free: it returns plain Python types and the callers
wrap them in Qdrant models.
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


def embed_hybrid(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    """Dense vectors + sparse lexical weights for a batch, one forward pass.

    Sparse weights come back as FlagEmbedding emits them: one dict per text
    mapping token id (string) -> weight. Use `to_indices_values` to convert.
    """
    output = _get_model().encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    dense = [vec.tolist() for vec in output["dense_vecs"]]
    return dense, output["lexical_weights"]


def embed_query_hybrid(text: str) -> tuple[list[float], dict[str, float]]:
    """Dense vector + sparse lexical weights for a single query string."""
    dense, sparse = embed_hybrid([text])
    return dense[0], sparse[0]


def to_indices_values(weights: dict[str, float]) -> tuple[list[int], list[float]]:
    """FlagEmbedding lexical weights -> Qdrant sparse format.

    Token ids arrive as strings and weights as numpy scalars; Qdrant wants
    parallel lists of int indices and plain float values.
    """
    indices = [int(token_id) for token_id in weights]
    values = [float(weight) for weight in weights.values()]
    return indices, values
