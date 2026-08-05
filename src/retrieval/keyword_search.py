"""
Keyword search.

Responsible for:
- Querying BM25 index
- Formatting search results
"""

from src.retrieval.keyword_store import KeywordStore


class KeywordSearch:
    """
    Keyword retrieval using BM25.
    """
    def __init__(self, keyword_store: KeywordStore):
        self.keyword_store = keyword_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Search documents using keywords.

        Args:
            query:
                User question.

            top_k:
                Number of results.

            source_filter:
                Optional PDF file filter.

        Returns:
            List of matched chunks.
        """
        results = self.keyword_store.search(query=query, top_k=top_k)

        if source_filter:
            results = [
                doc
                for doc in results
                if doc["metadata"]["source_file"]
                in source_filter
            ]

        return results