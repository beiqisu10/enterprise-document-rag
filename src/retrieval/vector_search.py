from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore


class VectorSearch:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:

        # 1. Convert question to embedding
        query_embedding = self.embedder.embed_query(query)

        # 2. Search vector database
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_filter=source_filter,
        )

        return results