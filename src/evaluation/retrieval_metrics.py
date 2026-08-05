"""
Evaluation metrics for retrieval performance.
"""

from __future__ import annotations


def evaluate_metrics(
    relevance_matrix: list[list[bool]],
    k_values: list[int],
) -> dict[str, float]:
    """
    Compute retrieval metrics from a relevance judgments matrix.

    Args:
        relevance_matrix: Each row is a list of booleans indicating whether the
                          retrieved document at that rank is relevant.
        k_values: Thresholds for metrics (e.g. [1, 5]).

    Returns:
        A dictionary containing hit rate, MRR, and precision for each k.
    """

    if not relevance_matrix:
        return {
            **{f"hit_rate@{k}": 0.0 for k in k_values},
            **{f"mrr@{k}": 0.0 for k in k_values},
            **{f"precision@{k}": 0.0 for k in k_values},
        }

    def hit_rate_at_k(row: list[bool], k: int) -> float:
        return 1.0 if any(row[:k]) else 0.0

    def reciprocal_rank_at_k(row: list[bool], k: int) -> float:
        for rank, relevant in enumerate(row[:k], start=1):
            if relevant:
                return 1.0 / rank
        return 0.0

    def precision_at_k(row: list[bool], k: int) -> float:
        if k == 0:
            return 0.0
        return sum(row[:k]) / k

    metrics: dict[str, float] = {}
    num_queries = len(relevance_matrix)

    for k in k_values:
        hit_rates = [hit_rate_at_k(row, k) for row in relevance_matrix]
        rr_scores = [reciprocal_rank_at_k(row, k) for row in relevance_matrix]
        precisions = [precision_at_k(row, k) for row in relevance_matrix]

        metrics[f"hit_rate@{k}"] = sum(hit_rates) / num_queries
        metrics[f"mrr@{k}"] = sum(rr_scores) / num_queries
        metrics[f"precision@{k}"] = sum(precisions) / num_queries

    return metrics
