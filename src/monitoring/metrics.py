"""
Metrics loader and aggregation functions
for RAG monitoring dashboard.
"""

from pathlib import Path

import pandas as pd

LOG_DIR = Path("data/logs")
QUERY_LOG_FILE = LOG_DIR / "query_logs.jsonl"
FEEDBACK_LOG_FILE = LOG_DIR / "feedback.jsonl"


def load_query_logs() -> pd.DataFrame:
    """Load RAG query logs."""
    if not QUERY_LOG_FILE.exists():
        return pd.DataFrame()

    df = pd.read_json(QUERY_LOG_FILE, lines=True)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_feedback() -> pd.DataFrame:
    """Load user feedback logs."""
    if not FEEDBACK_LOG_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_json(FEEDBACK_LOG_FILE, lines=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_summary_metrics() -> dict:
    """Calculate dashboard KPI metrics."""
    query_df = load_query_logs()
    feedback_df = load_feedback()

    total_queries = len(query_df)

    avg_latency = (
        query_df["latency_seconds"].mean()
        if not query_df.empty
        else 0
    )

    avg_retrieved = (
        query_df["retrieved_count"].mean()
        if not query_df.empty
        else 0
    )

    positive_feedback = 0
    if not feedback_df.empty:
        if "feedback" in feedback_df.columns:
            positive_feedback = (feedback_df["feedback"] == "positive").mean()
        elif "rating" in feedback_df.columns:
            positive_feedback = (feedback_df["rating"] == 1).mean()

    return {
        "total_queries": total_queries,
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_retrieved_documents": round(avg_retrieved, 2),
        "positive_feedback_rate": round(positive_feedback * 100, 2),
    }


def get_daily_query_volume() -> pd.DataFrame:
    """Query count by day."""
    df = load_query_logs()

    if df.empty:
        return pd.DataFrame()

    result = (
        df
        .set_index("timestamp")
        .resample("D")
        .size()
        .reset_index(name="queries")
    )

    return result


def get_latency_distribution() -> pd.Series:
    """Latency values for chart."""
    df = load_query_logs()

    if df.empty:
        return pd.Series(dtype=float)

    return df["latency_seconds"]


def get_feedback_distribution() -> pd.Series:
    """
    Positive vs negative feedback distribution.
    Supports both:
    - feedback: positive/negative
    - rating: 1/0
    """
    df = load_feedback()
    if df.empty:
        return pd.Series(dtype=int)

    # New format
    if "rating" in df.columns:
        return (
            df["rating"]
            .map(
                {
                    1: "Positive",
                    0: "Negative",
                    -1: "Comment",
                }
            )
            .value_counts()
        )

    # Old format
    if "feedback" in df.columns:
        return df["feedback"].value_counts()

    return pd.Series(dtype=int)


def get_top_questions(limit: int = 10) -> pd.Series:
    """Most frequent questions."""
    df = load_query_logs()

    if df.empty:
        return pd.Series(dtype=int)

    return df["question"].value_counts().head(limit)