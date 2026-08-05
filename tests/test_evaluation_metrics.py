import pytest
from src.evaluation.retrieval_metrics import evaluate_metrics


def test_evaluate_metrics_computes_expected_scores():
    relevance_matrix = [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ]

    metrics = evaluate_metrics(relevance_matrix, k_values=[1, 3])

    assert metrics["hit_rate@1"] == 0.25
    assert metrics["hit_rate@3"] == 0.75
    assert metrics["mrr@1"] == 0.25
    assert metrics["mrr@3"] == pytest.approx((1.0 + 0.5 + 0.3333333333333333) / 4)
    assert metrics["precision@1"] == 0.25
    assert metrics["precision@3"] == pytest.approx(0.25)


def test_evaluate_metrics_handles_empty_matrix():
    metrics = evaluate_metrics([], k_values=[1, 3])

    assert metrics["hit_rate@1"] == 0.0
    assert metrics["mrr@3"] == 0.0
    assert metrics["precision@1"] == 0.0
