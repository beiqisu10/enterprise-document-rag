import pytest

from src.rag.rag_pipeline import RAGPipeline


class DummyRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5, source_filter=None):
        return self.results


class DummyReranker:
    def rerank(self, query, documents, top_k=5):
        return documents[:top_k]


class DummyLLMClient:
    def generate(self, prompt):
        return {"answer": "dummy answer", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def test_rag_pipeline_returns_answer_and_documents():
    retriever = DummyRetriever([
        {"id": "doc1", "content": "text1", "metadata": {"source_file": "file1.pdf", "page_number": 1}},
        {"id": "doc2", "content": "text2", "metadata": {"source_file": "file2.pdf", "page_number": 2}},
    ])
    reranker = DummyReranker()
    llm = DummyLLMClient()

    pipeline = RAGPipeline(retriever=retriever, reranker=reranker, llm_client=llm)

    result = pipeline.answer("What is this?", retrieval_k=5, rerank_k=2)

    assert result["query"] == "What is this?"
    assert result["answer"] == "dummy answer"
    assert len(result["documents"]) == 2
    assert result["retrieved_count"] == 2
    assert result["usage"]["total_tokens"] == 2


def test_rag_pipeline_handles_no_documents():
    retriever = DummyRetriever([])
    reranker = DummyReranker()
    llm = DummyLLMClient()

    pipeline = RAGPipeline(retriever=retriever, reranker=reranker, llm_client=llm)

    result = pipeline.answer("What is this?", retrieval_k=5, rerank_k=2)

    assert result["answer"] == "I could not find relevant information in the documents."
    assert result["documents"] == []
    assert result["retrieved_count"] == 0
