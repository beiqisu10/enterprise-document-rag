"""
Run retrieval evaluation across all methods.
"""

from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.retrieval.vector_search import VectorSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.keyword_store import KeywordStore
from src.retrieval.vector_store import VectorStore


def main():
    # Initialize stores
    keyword_store = KeywordStore()
    keyword_store.load()

    vector_store = VectorStore()

    # Initialize search methods
    vector_search = VectorSearch(vector_store=vector_store)
    keyword_search = KeywordSearch(keyword_store=keyword_store)
    hybrid_search = HybridSearch(vector_search, keyword_search)
    reranker = Reranker()

    # Wrap methods for consistent interface
    def vector_only(query: str) -> list[dict]:
        return vector_search.search(query, top_k=10)

    def keyword_only(query: str) -> list[dict]:
        return keyword_search.search(query, top_k=10)

    def hybrid_rrf(query: str) -> list[dict]:
        return hybrid_search.search(query, top_k=10)

    def hybrid_rerank(query: str) -> list[dict]:
        candidates = hybrid_search.search(query, top_k=20)
        return reranker.rerank(query, candidates, top_k=10)

    methods = {
        "Vector Only": vector_only,
        "Keyword Only": keyword_only,
        "Hybrid (RRF)": hybrid_rrf,
        "Hybrid + Reranker": hybrid_rerank,
    }

    # Run evaluation
    evaluator = RetrievalEvaluator("data/ground-truth-retrieval.csv")
    results = evaluator.evaluate_all(methods, top_k=5)
    evaluator.print_report(results)


if __name__ == "__main__":
    main()