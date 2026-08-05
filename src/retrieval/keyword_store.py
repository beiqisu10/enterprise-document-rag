"""
BM25 keyword store.

Responsible for:
- Creating BM25 index
- Storing document chunks for keyword retrieval
- Querying keyword matches
- Persisting index to disk
"""

from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import DocumentChunk


class KeywordStore:
    """
    BM25 wrapper.
    """

    def __init__(
        self,
        persist_directory: str = "data/bm25_store",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.index_file = self.persist_directory / "bm25_index.pkl"
        self.documents_file = self.persist_directory / "documents.pkl"

        self.bm25 = None
        self.documents = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Convert text into tokens.

        Keep:
        - AWS
        - SSE-KMS
        - CVE-2025-xxxx
        """
        return re.findall(r"[a-z0-9][a-z0-9\-_.]+", text.lower())

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        """
        Build BM25 index from document chunks.

        Args:
            chunks:
                Chunked documents from Chunker.
        """
        self.documents = []

        for chunk in chunks:
            self.documents.append(
                {
                    "id": f"{chunk.source_file}_{chunk.page_number}_{chunk.chunk_index}",
                    "content": chunk.content,
                    "metadata": {
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                }
            )

        tokenized_documents = [self._tokenize(doc["content"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def save(self) -> None:
        """Persist BM25 index and documents."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(self.documents_file, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> None:
        """Load BM25 index and documents."""
        if not self.index_file.exists():
            raise FileNotFoundError("BM25 index does not exist")

        with open(self.index_file, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(self.documents_file, "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Keyword search.

        Args:
            query:
                User question.
            top_k:
                Number of results.

        Returns:
            Matched document chunks.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for index in ranked_indexes[:top_k]:
            doc = self.documents[index].copy()
            doc["keyword_score"] = float(scores[index])
            results.append(doc)

        return results

    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self.documents)

    def reset(self) -> None:
        """Remove stored BM25 files."""
        if self.index_file.exists():
            self.index_file.unlink()

        if self.documents_file.exists():
            self.documents_file.unlink()

        self.bm25 = None
        self.documents = []