"""
FastAPI application for Enterprise Document RAG.

Flow:

Client
  |
  v
FastAPI
  |
  v
RAGPipeline
  |
  +-- Hybrid Retrieval
  |
  +-- Reranking
  |
  +-- Prompt Construction
  |
  +-- LLM Generation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.vector_search import VectorSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_store import KeywordStore
from src.llm.client import LLMClient


# ============================================================
# Dependency initialization
# ============================================================
rag_pipeline: RAGPipeline | None = None


def create_rag_pipeline() -> RAGPipeline:
    """
    Build RAG dependencies.

    Components:
    - Vector search
    - Keyword search
    - Hybrid retrieval
    - Reranker
    - LLM client
    """
    keyword_store = KeywordStore()
    keyword_store.load()

    vector_store = VectorStore()

    vector_search = VectorSearch(vector_store)
    keyword_search = KeywordSearch(keyword_store)

    hybrid_search = HybridSearch(vector_search, keyword_search)
    reranker = Reranker()
    llm_client = LLMClient()

    return RAGPipeline(
        retriever=hybrid_search,
        reranker=reranker,
        llm_client=llm_client,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline

    print("Initializing RAG pipeline...")
    rag_pipeline = create_rag_pipeline()
    print("RAG pipeline ready.")

    yield

    print("Shutting down RAG pipeline.")


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="Enterprise Document RAG API",
    description=(
        "Retrieval Augmented Generation system "
        "for enterprise document question answering."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Request / Response models
# ============================================================
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    retrieval_k: int = Field(default=20, description="Number of retrieval candidates")
    rerank_k: int = Field(default=5, description="Number of documents passed to LLM")
    source_filter: list[str] | None = Field(default=None, description="Optional document filter")


class Source(BaseModel):
    source_file: str | None
    page: int | None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    usage: dict | None = None


# ============================================================
# Routes
# ============================================================
@app.get("/")
def health_check():
    return {"status": "ok", "service": "enterprise-document-rag"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready")

    try:
        result = rag_pipeline.answer(
            query=request.question,
            retrieval_k=request.retrieval_k,
            rerank_k=request.rerank_k,
            source_filter=request.source_filter,
        )

        documents = result.get("documents", [])
        sources = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            sources.append(
                Source(
                    source_file=metadata.get("source_file"),
                    page=metadata.get("page_number"),
                )
            )

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            usage=result.get("usage"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal RAG processing error") from e