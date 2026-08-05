"""
User feedback collection for Enterprise Document RAG.

Stores user ratings for generated answers and updates a single record
for the same question/answer pair instead of appending duplicates.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_FILE = Path("data/logs/feedback.jsonl")


def _normalize_feedback_label(rating: int) -> str:
    if rating == 1:
        return "positive"
    if rating == 0:
        return "negative"
    return "neutral"


def _generate_feedback_id(question: str, answer: str) -> str:
    payload = f"{question}\n{answer}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_feedback_records() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []

    records = []
    with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_feedback_records(records: list[dict]) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_feedback(
    question: str,
    answer: str,
    rating: int,
    comment: str | None = None,
) -> None:
    """
    Save or update user feedback.

    Args:
        question: User question.
        answer: Generated answer.
        rating:
            1 = helpful
            0 = not helpful
            -1 = comment only
        comment: Optional user comment.
    """
    feedback_id = _generate_feedback_id(question, answer)
    records = _load_feedback_records()

    existing = None
    for record in records:
        if record.get("feedback_id") == feedback_id:
            existing = record
            break

    if existing is None:
        existing = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,
            "feedback": _normalize_feedback_label(rating),
            "comment": comment,
        }
        records.append(existing)
    else:
        if rating != -1:
            existing["rating"] = rating
            existing["feedback"] = _normalize_feedback_label(rating)
        if comment is not None and comment != "":
            existing["comment"] = comment
        existing["timestamp"] = datetime.now(timezone.utc).isoformat()

    _write_feedback_records(records)


def load_feedback() -> list[dict]:
    """Load all feedback records."""
    return _load_feedback_records()


def get_feedback(question: str, answer: str) -> dict | None:
    """Return the saved feedback record for a question and answer pair."""
    feedback_id = _generate_feedback_id(question, answer)
    for record in _load_feedback_records():
        if record.get("feedback_id") == feedback_id:
            return record
    return None
