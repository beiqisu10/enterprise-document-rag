"""
Run simplified LLM evaluation.

Compares:
- basic vs strict prompt
- top_k = 3 vs 5 vs 8
"""

from src.evaluation.llm_evaluator import LLMEvaluator, LLMConfig

from src.retrieval.keyword_store import KeywordStore
from src.retrieval.vector_store import VectorStore

from src.retrieval.vector_search import VectorSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_rerank_retriever import HybridRerankRetriever


def main():
    # ------------------------
    # Build Retriever
    # ------------------------

    keyword_store = KeywordStore()
    keyword_store.load()

    vector_store = VectorStore()

    vector_search = VectorSearch(vector_store)
    keyword_search = KeywordSearch(keyword_store)

    hybrid_search = HybridSearch(vector_search, keyword_search)
    reranker = Reranker()

    retriever = HybridRerankRetriever(hybrid_search, reranker)

    configs = [
        # Baseline: basic prompt with 5 chunks
        LLMConfig(
            name="basic_k5",
            prompt_version="basic",
            top_k=5,
        ),
        # Comparison 1: strict prompt (forced citations + refusal)
        LLMConfig(
            name="strict_k5",
            prompt_version="strict",
            top_k=5,
        ),
        # # Comparison 2: less context
        # LLMConfig(
        #     name="basic_k3",
        #     prompt_version="basic",
        #     top_k=3,
        # ),
        # # Comparison 3: more context
        # LLMConfig(
        #     name="basic_k8",
        #     prompt_version="basic",
        #     top_k=8,
        # ),
    ]

    evaluator = LLMEvaluator(
        ground_truth_path="data/ground-truth-retrieval.csv",
        retriever=retriever,
    )

    # Quick validation with 30 samples, run full set after confirming it works
    results = evaluator.evaluate_all(configs, sample_size=30)
    evaluator.print_report(results)

    # Save results
    import json
    with open("data/llm_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()