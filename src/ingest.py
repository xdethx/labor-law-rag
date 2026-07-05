"""articles.json -> embed -> (re)create the `law` Qdrant collection.

Offline step, run manually:  python -m src.ingest
Never runs on the request path (CLAUDE.md). Safe to re-run: the collection is
dropped and recreated each time, and point ids are deterministic (uuid5 of
article_id), so a re-run is idempotent.
"""

import json
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import DENSE_VECTOR_NAME, EMBEDDING_DIM, LAW_COLLECTION
from src.embeddings import embed_dense
from src.retrieval import get_client

ARTICLES_PATH = Path("data/processed/articles.json")

# Stable namespace for deriving Qdrant point ids from article_id.
_POINT_NAMESPACE = uuid.NAMESPACE_DNS


def load_articles() -> list[dict]:
    return json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))


def content_for(article: dict) -> str:
    """Text actually embedded. Title is separate metadata but folding it into
    the embedded content is a cheap recall win (title often names the topic
    the article body doesn't restate, e.g. 'Kıdem tazminatı')."""
    if article["article_title"]:
        return f"{article['article_title']}\n{article['text']}"
    return article["text"]


def build_collection(client) -> None:
    if client.collection_exists(LAW_COLLECTION):
        client.delete_collection(LAW_COLLECTION)
    client.create_collection(
        collection_name=LAW_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
    )


def main() -> None:
    articles = load_articles()
    client = get_client()
    build_collection(client)

    vectors = embed_dense([content_for(a) for a in articles])

    points = [
        PointStruct(
            id=str(uuid.uuid5(_POINT_NAMESPACE, article["article_id"])),
            vector={DENSE_VECTOR_NAME: vector},
            payload={**article, "source": "law"},
        )
        for article, vector in zip(articles, vectors)
    ]
    client.upsert(collection_name=LAW_COLLECTION, points=points)

    print(f"Ingested {len(points)} points -> collection {LAW_COLLECTION!r}")
    print(f"  first: {articles[0]['article_id']}, last: {articles[-1]['article_id']}")


if __name__ == "__main__":
    main()
