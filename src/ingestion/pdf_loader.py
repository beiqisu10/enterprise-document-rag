"""
PDF document loader.

Responsible for:
- Loading PDF files
- Extracting text page by page
- Preserving document metadata for RAG citation
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class DocumentPage:
    """
    Represents one page extracted from a document.
    """
    content: str
    page_number: int
    source_file: str


class PDFLoader:
    """
    Load PDF files and extract page-level text.

    Output:
        List[DocumentPage]

    Example:
        [
            DocumentPage(
                content="Airflow scheduler...",
                page_number=10,
                source_file="airflow.pdf"
            )
        ]
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> list[DocumentPage]:
        """Extract text from PDF pages."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")

        pages = []

        try:
            document = fitz.open(self.file_path)

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                # Skip empty pages
                if not text:
                    continue

                pages.append(
                    DocumentPage(
                        content=text,
                        page_number=page_number,
                        source_file=self.file_path.name
                    )
                )

            document.close()

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {self.file_path}: {e}") from e

        return pages