"""bge-m3 embedding wrapper — dense + sparse in one forward pass (M3),
plus the bge-reranker-v2-m3 cross-encoder (M4).

bge-m3 emits dense (1024-d) and sparse lexical-weight vectors from the same
encode call, which is what makes hybrid retrieval possible with one model.

The SAME model embeds documents (ingest) and queries (retrieval) — this
module is the only place any model is instantiated, so that invariant holds.
This module stays qdrant-free: it returns plain Python types and the callers
wrap them in Qdrant models.
"""

import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import EMBEDDING_MODEL, RERANKER_MODEL

_model: BGEM3FlagModel | None = None
_reranker: tuple[AutoTokenizer, AutoModelForSequenceClassification] | None = None

# Pairs per forward pass — bounds CPU memory when scoring a 50-deep pool of
# full maddeler padded to 512 tokens.
_RERANK_BATCH_SIZE = 8
_RERANK_MAX_LENGTH = 512


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


def _get_reranker() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    # transformers directly, NOT FlagEmbedding's FlagReranker: transformers v5
    # removed tokenizer.prepare_for_model, which FlagEmbedding 1.4.0's reranker
    # still calls. bge-reranker-v2-m3 is a plain sequence-classification head
    # (one logit per pair), so the direct call is equivalent.
    global _reranker
    if _reranker is None:
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        model.eval()
        _reranker = (tokenizer, model)
    return _reranker


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score for each (query, text) pair.

    Scores are raw logits (unbounded, higher = more relevant) — only their
    ordering matters.
    """
    tokenizer, model = _get_reranker()
    scores: list[float] = []
    for start in range(0, len(texts), _RERANK_BATCH_SIZE):
        batch = texts[start:start + _RERANK_BATCH_SIZE]
        inputs = tokenizer(
            [query] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=_RERANK_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1)
        scores.extend(float(score) for score in logits)
    return scores


def to_indices_values(weights: dict[str, float]) -> tuple[list[int], list[float]]:
    """FlagEmbedding lexical weights -> Qdrant sparse format.

    Token ids arrive as strings and weights as numpy scalars; Qdrant wants
    parallel lists of int indices and plain float values.
    """
    indices = [int(token_id) for token_id in weights]
    values = [float(weight) for weight in weights.values()]
    return indices, values
