"""Retrieval evaluation harness (M2).

Runs the labeled dataset through the PRODUCTION retrieval path
(src.retrieval.retrieve_law) and reports recall@5, recall@10, MRR@10 per
question type and overall. No LLM involved — free, fast, deterministic.

Usage:  python -m eval.run          (qdrant must be up, `law` ingested)

`off-corpus` items (empty expected_articles) test LLM refusal, not retrieval;
they are counted but excluded from the metrics.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DATASET_PATH = Path("eval/dataset.jsonl")
RESULTS_DIR = Path("eval/results")

VALID_TYPES = {"direct", "paraphrase", "exact-term", "article-number", "off-corpus"}

K_VALUES = (5, 10)
RETRIEVE_K = max(K_VALUES)


# --- metrics (pure functions, unit-tested offline) ---------------------------

def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of expected ids present in the top-k retrieved ids."""
    if not expected:
        raise ValueError("recall undefined for empty expected set")
    hits = sum(1 for e in expected if e in retrieved[:k])
    return hits / len(expected)


def mrr_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Reciprocal rank of the first expected id in the top-k, else 0.0."""
    if not expected:
        raise ValueError("MRR undefined for empty expected set")
    for rank, article_id in enumerate(retrieved[:k], start=1):
        if article_id in expected:
            return 1.0 / rank
    return 0.0


# --- dataset ------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question", "").strip():
            raise ValueError(f"{path}:{line_no}: empty question")
        if item.get("type") not in VALID_TYPES:
            raise ValueError(f"{path}:{line_no}: unknown type {item.get('type')!r}")
        if item["type"] == "off-corpus":
            if item.get("expected_articles"):
                raise ValueError(f"{path}:{line_no}: off-corpus must have no expected articles")
        elif not item.get("expected_articles"):
            raise ValueError(f"{path}:{line_no}: expected_articles required for type {item['type']}")
        items.append(item)
    return items


# --- runner -------------------------------------------------------------------

def evaluate(items: list[dict]) -> list[dict]:
    """Retrieve for every item; return per-item results (scored where labeled)."""
    from src.retrieval import retrieve_law  # deferred: loads bge-m3 on first call

    results = []
    for item in items:
        chunks = retrieve_law(item["question"], top_k=RETRIEVE_K)
        retrieved = [c["article_id"] for c in chunks]
        result = {**item, "retrieved": retrieved}
        if item["expected_articles"]:
            result["metrics"] = {
                **{f"recall@{k}": recall_at_k(item["expected_articles"], retrieved, k) for k in K_VALUES},
                "mrr@10": mrr_at_k(item["expected_articles"], retrieved, 10),
            }
        results.append(result)
    return results


def aggregate(results: list[dict]) -> dict:
    """Mean metrics per type and overall. off-corpus contributes counts only."""
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    def mean_metrics(group: list[dict]) -> dict:
        scored = [r["metrics"] for r in group if "metrics" in r]
        if not scored:
            return {"n": len(group)}
        return {
            "n": len(group),
            **{key: sum(m[key] for m in scored) / len(scored) for key in scored[0]},
        }

    aggregates = {t: mean_metrics(group) for t, group in sorted(by_type.items())}
    aggregates["OVERALL"] = mean_metrics([r for r in results if "metrics" in r])
    return aggregates


def print_report(results: list[dict], aggregates: dict) -> None:
    metric_cols = ["recall@5", "recall@10", "mrr@10"]
    print(f"\n{'type':<16} {'n':>3}  " + "  ".join(f"{c:>9}" for c in metric_cols))
    print("-" * 52)
    for type_name, agg in aggregates.items():
        cells = "  ".join(
            f"{agg[c]:>9.3f}" if c in agg else f"{'n/a':>9}" for c in metric_cols
        )
        print(f"{type_name:<16} {agg['n']:>3}  {cells}")

    misses = [r for r in results if "metrics" in r and r["metrics"]["recall@5"] < 1.0]
    if misses:
        print(f"\nMisses (recall@5 < 1.0): {len(misses)}")
        for r in misses:
            print(f"  [{r['type']}] {r['question']}")
            print(f"    expected: {r['expected_articles']}  got top-5: {r['retrieved'][:5]}")


def write_results(results: list[dict], aggregates: dict) -> Path:
    from src.config import settings
    from src.retrieval import get_client
    from src.config import LAW_COLLECTION

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "timestamp_utc": timestamp,
        "config": {
            "retrieval": "dense-only",  # update at M3/M4
            "retrieve_k": RETRIEVE_K,
            "top_k_final": settings.top_k_final,
            "collection_points": get_client().count(LAW_COLLECTION).count,
        },
        "aggregates": aggregates,
        "items": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    items = load_dataset()
    print(f"Loaded {len(items)} items from {DATASET_PATH}")
    results = evaluate(items)
    aggregates = aggregate(results)
    print_report(results, aggregates)
    path = write_results(results, aggregates)
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
