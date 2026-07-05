"""Unit tests for eval.run metric functions — pure math, no Qdrant/network."""

import pytest

from eval.run import mrr_at_k, recall_at_k


def test_recall_hit_at_rank_1():
    assert recall_at_k(["madde-63"], ["madde-63", "madde-41"], 5) == 1.0


def test_recall_miss():
    assert recall_at_k(["madde-63"], ["madde-1", "madde-2"], 5) == 0.0


def test_recall_partial_multi_expected():
    retrieved = ["madde-17", "madde-2", "madde-3"]
    assert recall_at_k(["madde-17", "gecici-6"], retrieved, 5) == 0.5


def test_recall_respects_k_cutoff():
    retrieved = ["madde-1", "madde-2", "madde-3", "madde-4", "madde-5", "madde-63"]
    assert recall_at_k(["madde-63"], retrieved, 5) == 0.0
    assert recall_at_k(["madde-63"], retrieved, 10) == 1.0


def test_mrr_rank_1():
    assert mrr_at_k(["madde-63"], ["madde-63"], 10) == 1.0


def test_mrr_rank_3():
    assert mrr_at_k(["madde-63"], ["madde-1", "madde-2", "madde-63"], 10) == pytest.approx(1 / 3)


def test_mrr_first_expected_counts():
    # Two expected ids at ranks 2 and 4 -> reciprocal rank of the FIRST hit.
    retrieved = ["madde-1", "gecici-6", "madde-2", "madde-17"]
    assert mrr_at_k(["madde-17", "gecici-6"], retrieved, 10) == pytest.approx(1 / 2)


def test_mrr_miss_is_zero():
    assert mrr_at_k(["madde-63"], ["madde-1", "madde-2"], 10) == 0.0


def test_empty_expected_rejected():
    with pytest.raises(ValueError):
        recall_at_k([], ["madde-1"], 5)
    with pytest.raises(ValueError):
        mrr_at_k([], ["madde-1"], 10)
