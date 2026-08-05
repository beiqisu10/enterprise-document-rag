"""
Cross-Encoder Reranker module.

Responsible for:
- Re-ranking retrieved documents based on query-document relevance
- Improving precision after hybrid retrieval (Vector + BM25 + RRF)

Pipeline:
    Query
      |
      v
 Vector Search + Keyword Search
      |
      v
     RRF
      |
      v
 Cross Encoder Reranker
      |
      v
 Top K Documents

Usage:
    from src.retrieval.reranker import Reranker
    reranker = Reranker()
    results = reranker.rerank(
        query="How does AWS protect data at rest?",
        documents=candidates,
        top_k=5,
    )
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross Encoder based document reranker.
    Unlike embedding models (bi-encoders), cross encoders evaluate query and document together,
    producing a more accurate relevance score.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        """
        Initialize reranker model.

        Args:
            model_name: HuggingFace cross encoder model.
                       Recommended: BAAI/bge-reranker-base, BAAI/bge-reranker-large
            device: cpu / cuda / mps. None means auto detection.
        """
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-rank retrieved documents.

        Args:
            query: User question.
            documents: Candidate documents from hybrid search.
                      Example: {"id": "...", "content": "...", "metadata": {}}
            top_k: Number of final documents returned.

        Returns:
            Documents sorted by rerank score.
        """
        if not documents:
            return []

        # Build query-document pairs
        pairs = []
        valid_documents = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            pairs.append((query, content))
            valid_documents.append(doc)

        if not pairs:
            return []

        # Cross encoder inference
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        # Attach rerank scores
        results = []
        for doc, score in zip(valid_documents, scores):
            ranked_doc = doc.copy()
            ranked_doc["rerank_score"] = float(score)
            results.append(ranked_doc)

        # Highest relevance first
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]