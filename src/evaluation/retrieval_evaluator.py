"""
Unified retrieval evaluator.

Supports evaluating multiple retrieval methods against
the same ground truth dataset with consistent metrics.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.evaluation.retrieval_metrics import evaluate_metrics


class RetrievalEvaluator:
    """
    Evaluate one or more retrieval methods using ground truth data.

    Ground truth format (CSV):
        question,chunk_id,source_file,page_number,chunk_index

    Each row represents: "For this question, the correct answer is chunk_id."
    """

    def __init__(self, ground_truth_path: str | Path):
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> list[dict]:
        """Load ground truth from CSV."""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {self.ground_truth_path}"
            )

        df = pd.read_csv(self.ground_truth_path)
        required = {"question", "chunk_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Ground truth CSV missing columns: {missing}")

        return df.to_dict(orient="records")

    def evaluate(
        self,
        search_function: Callable[[str], list[dict]],
        method_name: str = "unknown",
        top_k: int = 5,
    ) -> dict[str, float]:
        """
        Evaluate a single retrieval method.

        Args:
            search_function: Callable that takes a query string and returns
                           a list of result dicts, each with an "id" key.
            method_name: Name of the method for reporting.
            top_k: Number of top results to evaluate.

        Returns:
            Dictionary of metrics.
        """
        relevance_total = []

        for item in tqdm(self.ground_truth, desc=f"Evaluating {method_name}"):
            query = item["question"]
            expected_id = str(item["chunk_id"])

            try:
                results = search_function(query)
            except Exception as e:
                print(f"Search failed for query '{query[:50]}...': {e}")
                relevance_total.append([False] * top_k)
                continue

            # Truncate to top_k and check relevance
            top_results = results[:top_k]
            relevance = [
                str(doc.get("id", "")) == expected_id
                for doc in top_results
            ]

            # Pad if fewer than top_k results returned
            while len(relevance) < top_k:
                relevance.append(False)

            relevance_total.append(relevance)

        metrics = evaluate_metrics(relevance_total, k_values=[top_k])
        metrics["method"] = method_name
        metrics["total_queries"] = len(self.ground_truth)
        metrics["top_k"] = top_k

        return metrics

    def evaluate_all(
        self,
        methods: dict[str, Callable[[str], list[dict]]],
        top_k: int = 5,
    ) -> list[dict[str, float]]:
        """
        Evaluate multiple retrieval methods and return comparison.

        Args:
            methods: Dict of {method_name: search_function}.
            top_k: K value for evaluation.

        Returns:
            List of metric dicts, one per method.
        """
        results = []
        for name, fn in methods.items():
            result = self.evaluate(fn, method_name=name, top_k=top_k)
            results.append(result)
        return results

    def print_report(
        self,
        results: list[dict[str, float]],
        top_k: int = 5,
    ) -> None:
        """
        Pretty-print evaluation comparison table.
        """
        if not results:
            print("No results to report.")
            return

        # Determine metric columns
        metric_keys = [
            k for k in results[0].keys()
            if k not in ("method", "total_queries", "top_k")
        ]

        # Header
        header = f"{'Method':<25}" + "".join(f"{k:<18}" for k in metric_keys) + "Queries"
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Rows
        for r in results:
            row = f"{r['method']:<25}"
            for k in metric_keys:
                val = r.get(k, 0.0)
                if isinstance(val, float):
                    row += f"{val:<18.4f}"
                else:
                    row += f"{str(val):<18}"
            row += str(r.get("total_queries", 0))
            print(row)

        print("=" * len(header))

        # Highlight best by hit_rate@top_k
        hit_key = f"hit_rate@{top_k}"
        best = max(results, key=lambda x: x.get(hit_key, 0.0))
        print(f"\n🏆 Best Method: {best['method']}")
        print(f"   Hit Rate@{top_k}: {best.get(hit_key, 0):.4f}")
        print(f"   MRR@{top_k}: {best.get(f'mrr@{top_k}', 0):.4f}")