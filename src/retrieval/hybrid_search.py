"""
Hybrid search using Reciprocal Rank Fusion (RRF).

Combines:
- Vector semantic search
- Keyword BM25 search

Reference:
Cormack et al. (2009) - Reciprocal Rank Fusion
"""

from collections import defaultdict


class HybridSearch:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_search,
        keyword_search,
        rrf_k: int = 60,
    ):
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Perform hybrid retrieval using RRF.
        """
        candidate_k = top_k * 10

        vector_results = self.vector_search.search(
            query=query,
            top_k=candidate_k,
            source_filter=source_filter,
        )

        keyword_results = self.keyword_search.search(
            query=query,
            top_k=candidate_k,
            source_filter=source_filter,
        )

        return self._rrf(
            vector_results,
            keyword_results,
            top_k,
        )

    def _rrf(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        score = Σ 1 / (k + rank)
        """
        scores = defaultdict(float)
        documents = {}

        # Process vector rankings
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1 / (self.rrf_k + rank)

            if doc_id not in documents:
                documents[doc_id] = doc.copy()
                documents[doc_id]["vector_rank"] = rank
                documents[doc_id]["keyword_rank"] = None

        # Process keyword rankings
        for rank, doc in enumerate(keyword_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1 / (self.rrf_k + rank)

            if doc_id not in documents:
                documents[doc_id] = doc.copy()
                documents[doc_id]["vector_rank"] = None
                documents[doc_id]["keyword_rank"] = rank
            else:
                documents[doc_id]["keyword_rank"] = rank

        # Build final results
        results = []
        for doc_id, score in scores.items():
            doc = documents[doc_id]
            doc["rrf_score"] = score
            results.append(doc)

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]