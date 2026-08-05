"""
Document chunking for RAG.

Responsible for:
- Splitting document pages into smaller chunks
- Preserving metadata for retrieval and citation
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_loader import DocumentPage


@dataclass
class DocumentChunk:
    """
    Represents one chunk ready for embedding.
    """
    content: str
    source_file: str
    page_number: int
    chunk_index: int


class DocumentChunker:
    """
    Split document pages into smaller chunks.

    Default settings:
        chunk_size = 1000 characters
        chunk_overlap = 200 characters
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """
        Split pages into chunks.

        Args:
            pages:
                List of DocumentPage objects.

        Returns:
            List of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self.splitter.split_text(page.content)

            for text in page_chunks:
                cleaned = text.strip()

                if not cleaned:
                    continue

                chunks.append(
                    DocumentChunk(
                        content=cleaned,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

        return chunks