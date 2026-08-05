# Enterprise Document RAG

# Problem Statement

Enterprise documents often contain large amounts of unstructured information, making it difficult for users to quickly find accurate answers.

Traditional keyword-based search can miss the semantic meaning behind user questions, while manually reviewing long documents is time-consuming and inefficient.

This project solves this problem by building an AI-powered document question answering system that combines semantic search, keyword retrieval, and reranking to provide accurate, grounded answers with relevant document citations.

# Overview

An enterprise-oriented Retrieval-Augmented Generation (RAG) system for question answering over large document collections.

The system automatically ingests PDF documents, indexes them using both dense vector embeddings and BM25 keyword search, reranks retrieved passages with a Cross-Encoder model, and generates grounded answers using OpenAI GPT models.

It also includes retrieval evaluation, LLM-as-a-Judge evaluation, monitoring dashboards, and user feedback collection.

---

# Architecture

```
                    PDF Documents
                          │
                          ▼
                 Document Ingestion
             (Load → Chunk → Embedding)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
   ChromaDB Vector Store               BM25 Store
        │                                   │
        └──────────── Hybrid Search ────────┘
                      (RRF Fusion)
                          │
                          ▼
                  Cross Encoder Reranker
                          │
                          ▼
                     Prompt Builder
                          │
                          ▼
                   OpenAI Chat Model
                          │
                          ▼
                FastAPI + Streamlit UI
                          │
                          ▼
          Monitoring Dashboard & Feedback
```

---

# Features

- Automated PDF ingestion pipeline
- Recursive document chunking
- Dense vector retrieval using ChromaDB
- BM25 keyword retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- OpenAI-powered answer generation
- FastAPI REST API
- Streamlit web application
- Retrieval evaluation (Hit Rate, MRR, NDCG)
- LLM-as-a-Judge evaluation
- Query monitoring dashboard
- User feedback collection

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.13 |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| Keyword Search | BM25 |
| Reranker | BAAI/bge-reranker-base |
| Backend | FastAPI |
| Frontend | Streamlit |
| Monitoring | Streamlit Dashboard |
| Evaluation | Retrieval Metrics + LLM Judge |

---

# Requirements

- Python 3.13+
- OpenAI API Key

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd enterprise-document-rag
```

Install all dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Ingest Documents

Build the vector index and BM25 index.

```bash
uv run python scripts/ingest_documents.py
```

---

## Run with Docker Compose

After creating the `.env` file, build and start the application:

```bash
docker compose up --build
```

This launches:

- FastAPI API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

The Streamlit application communicates with the FastAPI service through the internal Docker network.

---

## Local Development

Run the services locally without Docker Compose.

### Start the FastAPI Backend

```bash
uv run uvicorn src.api.main:app --reload
```

API Endpoint:

```
POST /query
```

Example request:

```json
{
  "question": "How does AWS protect data at rest?",
  "retrieval_k": 20,
  "rerank_k": 5,
  "source_filter": [
    "wellarchitected-framework.pdf"
  ]
}
```

Example response:

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {
      "source_file": "wellarchitected-framework.pdf",
      "page": 399
    }
  ]
}
```

### Start the Streamlit Web UI

```bash
uv run streamlit run src/frontend/app.py
```

The Streamlit interface provides:

- Natural language chat
- Source citations
- Retrieval configuration
- User feedback
- Monitoring dashboard
![chat](./assets/screenshots/chat.png)
---

# Monitoring

Every query is automatically logged.

Logged metrics include:

- Query timestamp
- Latency
- Retrieval count
- Reranked document count
- Prompt tokens
- Completion tokens
- Total tokens

The dashboard visualizes:

- Total queries
- Average latency
- Retrieved documents
- Token usage
- User feedback
- Most frequently asked questions
![dashboard](./assets/screenshots/dashboard.png)
![query_logs](./assets/screenshots/query_logs.png)
![feedback_logs](./assets/screenshots/feedback_logs.png)
---

# Evaluation

## Retrieval Evaluation

Metrics:

- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- NDCG@K

Results on 300 ground-truth questions:
| Method                | Hit Rate\@5 | MRR\@5     | NDCG\@5    |
| --------------------- | ----------- | ---------- | ---------- |
| Vector Only           | 0.8833      | 0.7489     | 0.7849     |
| Keyword Only          | 0.8433      | 0.7458     | 0.7728     |
| Hybrid (RRF)          | 0.9167      | 0.8106     | 0.8382     |
| **Hybrid + Reranker** | **0.9367**  | **0.8570** | **0.8792** |

Run:

```bash
python scripts/test_retrieval.py
```

---

## LLM Evaluation

Answers are evaluated using GPT-4o-mini as an LLM Judge.

Evaluation dimensions:

- Correctness
- Faithfulness
- Citation Quality
- Helpfulness
- Hallucination

Results on 30 ground-truth questions:
| Config         | Correctness | Faithfulness | Hallucination | Citation | Overall  |
| -------------- | ----------- | ------------ | ------------- | -------- | -------- |
| basic\_k5      | 4.70        | 4.70         | 6.67%         | 4.77     | 4.73     |
| **strict\_k5** | **4.97**    | **4.97**     | **0.00%**     | **4.97** | **4.97** |

Run:

```bash
python scripts/test_llm_evaluator.py
```

---

# Design Decisions

## Hybrid Retrieval
Dense retrieval provides semantic understanding while BM25 improves exact keyword matching.
RRF combines both approaches without requiring score normalization.

## Reranking
A Cross Encoder reranker is applied after retrieval because it provides higher accuracy while avoiding the cost of scoring the entire corpus.

## Vector Database
ChromaDB was selected for local development due to simplicity and support for persistent vector storage.

---

# Project Structure

```
src/
├── api/
├── embedding/
├── evaluation/
├── frontend/
├── ingestion/
├── llm/
├── monitoring/
├── rag/
└── retrieval/

scripts/
    generate_ground_truth
    ingest_documents.py
    run_llm_evaluation.py
    run_retrieval_evaluation.py

src/frontend/
    app.py
    dashboard.py
    feedback.py

data/
    raw/
    chroma_db/
    bm25_store/
    logs/
```

---

# Testing

```bash
pytest
```

---

# Future Improvements

- Multi-document upload
- Authentication
- Persistent feedback database
- Grafana monitoring
- LangSmith tracing
- Streaming responses
- Async retrieval pipeline

---

# Productionisation

The current ingestion workflow is implemented as a Python pipeline.
In production, this can be orchestrated using Airflow or Prefect with:

- Scheduled ingestion jobs
- Incremental document processing
- Failure retry
- Data quality checks
- Index refresh management

---

# License

MIT