class HybridRerankRetriever:
    """Hybrid search + Cross-Encoder reranking."""
    
    def __init__(self, hybrid_search, reranker):
        self.hybrid_search = hybrid_search
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid_search.search(query, top_k=20)
        return self.reranker.rerank(query, candidates, top_k)