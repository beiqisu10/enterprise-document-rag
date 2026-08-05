from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_store import KeywordStore

PDF_PATH = "data/raw/wellarchitected-framework.pdf"

# 1. Load PDF
loader = PDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded pages: {len(pages)}")

# 2. Chunk documents
chunker = DocumentChunker()
chunks = chunker.chunk(pages)
print(f"Created chunks: {len(chunks)}")

# 3. Generate embeddings
embedder = Embedder()
embeddings = embedder.embed_documents(chunks)
print(f"Generated embeddings: {len(embeddings)}")

# 4. Store in ChromaDB
store = VectorStore()
store.reset()
store.add_documents(chunks, embeddings)
print(f"ChromaDB documents: {store.count()}")

keyword_store = KeywordStore()
keyword_store.reset()
keyword_store.add_documents(chunks)
keyword_store.save()