"""
RAG pipeline orchestration.

Flow: Query -> Retrieval -> Reranking -> Prompt construction -> LLM generation
"""

import time

from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.llm.prompt import build_prompt
from src.llm.client import LLMClient
from src.monitoring.logger import log_query


class RAGPipeline:
    def __init__(self, retriever: HybridSearch, reranker: Reranker, llm_client: LLMClient):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm_client

    def answer(
        self,
        query: str,
        retrieval_k: int = 20,
        rerank_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> dict:
        """
        Run full RAG pipeline.

        Flow: Retrieval -> Reranking -> Prompt -> LLM

        Returns:
            {
                "query": str,
                "answer": str,
                "documents": list[dict],
                "retrieved_count": int,
                "usage": dict | None,
            }
        """
        start_time = time.time()

        # 1. Retrieve candidates
        try:
            candidates = self.retriever.search(query, top_k=retrieval_k, source_filter=source_filter)
        except Exception as e:
            return {
                "query": query,
                "answer": "Search system temporarily unavailable. Please try again.",
                "documents": [],
                "retrieved_count": 0,
                "usage": None,
                "error": f"retrieval_failed: {e}",
            }

        # 2. Handle empty results
        if not candidates:
            return {
                "query": query,
                "answer": "I could not find relevant information in the documents.",
                "documents": [],
                "retrieved_count": 0,
                "usage": None,
            }

        # 3. Rerank
        try:
            top_documents = self.reranker.rerank(query=query, documents=candidates, top_k=rerank_k)
        except Exception:
            top_documents = candidates[:rerank_k]  # fallback

        # 4. Build prompt
        prompt = build_prompt(query=query, documents=top_documents)

        # 5. Generate answer
        try:
            response = self.llm.generate(prompt)
        except Exception as e:
            return {
                "query": query,
                "answer": "LLM generation failed. Please try again later.",
                "documents": top_documents,
                "retrieved_count": len(candidates),
                "usage": None,
                "error": f"generation_failed: {e}",
            }

        # 6. Monitoring logging
        log_query(
            question=query,
            latency=time.time() - start_time,
            retrieved_count=len(candidates),
            reranked_count=len(top_documents),
            usage=response.get("usage"),
        )

        # 7. Return response
        return {
            "query": query,
            "answer": response["answer"],
            "documents": top_documents,
            "retrieved_count": len(candidates),
            "usage": response.get("usage"),
        }