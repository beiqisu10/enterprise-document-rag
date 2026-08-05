"""
Application logging utilities for RAG monitoring.

Stores:
- Query logs
- User feedback
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("data/logs")
QUERY_LOG_FILE = LOG_DIR / "query_logs.jsonl"
FEEDBACK_LOG_FILE = LOG_DIR / "feedback.jsonl"


def _ensure_log_dir():
    """Create log directory if it does not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(file_path: Path, data: dict):
    """Append one JSON record per line."""
    _ensure_log_dir()

    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_query(
    question: str,
    latency: float,
    retrieved_count: int,
    reranked_count: int,
    usage: dict | None = None,
):
    """
    Log one RAG query.

    Args:
        question:
            User query.
        latency:
            Total RAG response time in seconds.
        retrieved_count:
            Number of documents returned by retrieval.
        reranked_count:
            Number of documents passed to LLM.
        usage:
            LLM token usage.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "latency_seconds": round(latency, 3),
        "retrieved_count": retrieved_count,
        "reranked_count": reranked_count,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
    }

    _append_jsonl(QUERY_LOG_FILE, record)


def log_feedback(
    question: str,
    answer: str,
    feedback: str,
):
    """
    Log user feedback.

    Example:
        feedback="positive"
        feedback="negative"
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "feedback": feedback,
    }

    _append_jsonl(FEEDBACK_LOG_FILE, record)