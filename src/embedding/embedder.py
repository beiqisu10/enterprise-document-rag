"""
Embedding module.

Responsible for:
- Generating embeddings from document chunks
- Generating embeddings for user queries
"""

from openai import OpenAI
from src.config import settings
from src.ingestion.chunker import DocumentChunk


class Embedder:
    """
    OpenAI embedding client.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def embed_documents(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings in batches.

        Args:
            chunks:
                Document chunks.

            batch_size:
                Number of chunks per API request.

        Returns:
            List of embedding vectors.
        """

        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [
                chunk.content
                for chunk in batch
            ]

            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

            batch_embeddings = [
                item.embedding
                for item in response.data
            ]

            all_embeddings.extend(batch_embeddings)

            print(
                f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks"
            )

        return all_embeddings

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate embedding for a user query.
        """

        response = self.client.embeddings.create(
            model=self.model,
            input=query,
        )

        return response.data[0].embedding