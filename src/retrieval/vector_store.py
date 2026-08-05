"""
ChromaDB vector store.

Responsible for:
- Creating collections
- Storing document embeddings
- Querying similar documents
"""

from chromadb import PersistentClient

from src.ingestion.chunker import DocumentChunk


class VectorStore:
    """
    ChromaDB wrapper.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "enterprise_documents",
    ):
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store document chunks with embeddings.
        """
        ids = [
            f"{chunk.source_file}_{chunk.page_number}_{chunk.chunk_index}"
            for chunk in chunks
        ]

        documents = [
            chunk.content
            for chunk in chunks
        ]

        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Search similar document chunks.

        Args:
            query_embedding:
                Embedding vector of user query.
            top_k:
                Number of similar chunks to return.
            source_filter:
                Optional list of document names to search from.

                Example:
                [
                    "wellarchitected-framework.pdf",
                    "spark.pdf"
                ]

        Returns:
            List of matched chunks with metadata.
        """
        # Build Chroma metadata filter
        where = None
        if source_filter:
            where = {"source_file": {"$in": source_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        formatted_results = []
        for doc, metadata, distance, doc_id in zip(
            documents,
            metadatas,
            distances,
            ids,
        ):
            formatted_results.append(
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return formatted_results

    def count(self) -> int:
        """Number of indexed chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Remove all indexed documents."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )