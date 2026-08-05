"""
RAG Monitoring Dashboard.

Displays:
- Query volume
- Latency
- Token usage
- User feedback
- Popular questions
"""

import streamlit as st

from src.monitoring.metrics import (
    load_query_logs,
    load_feedback,
    get_summary_metrics,
    get_daily_query_volume,
    get_latency_distribution,
    get_feedback_distribution,
    get_top_questions,
)


def render_dashboard():
    st.title("📊 Enterprise Document RAG Monitoring")

    # =====================================================
    # Summary KPIs
    # =====================================================
    try:
        metrics = get_summary_metrics()
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", metrics.get("total_queries", 0))

    with col2:
        st.metric("Avg Latency (s)", metrics.get("avg_latency_seconds", 0))

    with col3:
        st.metric("Avg Retrieved Docs", metrics.get("avg_retrieved_documents", 0))

    with col4:
        st.metric("Positive Feedback %", f"{metrics.get('positive_feedback_rate', 0)}%")

    st.divider()

    # =====================================================
    # Load Logs
    # =====================================================
    query_logs = load_query_logs()

    # =====================================================
    # Chart 1 - Query Volume
    # =====================================================
    st.subheader("📈 Query Volume Over Time")

    daily_queries = get_daily_query_volume()

    if not daily_queries.empty:
        daily_queries = daily_queries.set_index("timestamp")
        st.line_chart(daily_queries["queries"])
    else:
        st.info("No query data available.")

    # =====================================================
    # Chart 2 - Latency
    # =====================================================
    st.subheader("⏱ Response Latency Distribution")

    latency = get_latency_distribution()

    if not latency.empty:
        st.bar_chart(latency)
    else:
        st.info("No latency data available.")

    # =====================================================
    # Chart 3 - Token Usage
    # =====================================================
    st.subheader("🔢 Token Usage Over Time")

    if not query_logs.empty and "total_tokens" in query_logs.columns:
        token_df = query_logs[["timestamp", "total_tokens"]].copy()
        token_df = token_df.set_index("timestamp")
        st.line_chart(token_df)
    else:
        st.info("No token usage data available.")

    # =====================================================
    # Chart 4 - Feedback
    # =====================================================
    st.subheader("👍 User Feedback")

    feedback_distribution = get_feedback_distribution()

    if not feedback_distribution.empty:
        st.bar_chart(feedback_distribution)
    else:
        st.info("No feedback collected yet.")

    # =====================================================
    # Chart 5 - Top Questions
    # =====================================================
    st.subheader("🔥 Most Asked Questions")

    top_questions = get_top_questions(limit=10)

    if not top_questions.empty:
        st.bar_chart(top_questions)
    else:
        st.info("No question history available.")

    st.divider()

    # =====================================================
    # Raw Logs
    # =====================================================
    with st.expander("📄 View Raw Query Logs"):
        if not query_logs.empty:
            st.dataframe(query_logs, width="stretch")
        else:
            st.info("No query logs available.")

    # =====================================================
    # Feedback Logs
    # =====================================================
    with st.expander("💬 View Feedback Logs"):
        feedback_logs = load_feedback()

        if not feedback_logs.empty:
            st.dataframe(feedback_logs, width="stretch")
        else:
            st.info("No feedback logs available.")