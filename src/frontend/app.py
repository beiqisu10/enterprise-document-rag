"""
Streamlit UI for Enterprise Document RAG.
"""

import hashlib
import requests
import streamlit as st

from src.frontend.dashboard import render_dashboard
from src.frontend.feedback import get_feedback, save_feedback

API_URL = st.secrets.get("API_URL") or "http://localhost:8000/query"


def _feedback_widget_key(prefix: str, question: str, answer: str) -> str:
    payload = f"{question}\n{answer}".encode("utf-8")
    fingerprint = hashlib.sha256(payload).hexdigest()
    return f"{prefix}_{fingerprint}"

st.set_page_config(
    page_title="Enterprise Document RAG",
    page_icon="📚",
    layout="wide",
)

# =========================================================
# Sidebar
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📊 Monitoring"],
)

if page == "📊 Monitoring":
    render_dashboard()
    st.stop()

# =========================================================
# Chat Page
# =========================================================
st.title("📚 Enterprise Document RAG")
st.caption("Enterprise knowledge assistant powered by Hybrid Search + Reranker + LLM")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input(
    "Ask a question",
    placeholder="How does AWS protect data at rest?",
)

with st.expander("⚙ Retrieval Settings"):
    col1, col2 = st.columns(2)

    with col1:
        retrieval_k = st.slider("Retrieval Candidates", 5, 50, 20)

    with col2:
        rerank_k = st.slider("Documents to LLM", 1, 10, 5)

if st.button("Ask", type="primary", disabled=not question):
    payload = {
        "question": question,
        "retrieval_k": retrieval_k,
        "rerank_k": rerank_k,
    }

    with st.spinner("Searching documents..."):
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

    st.session_state.history.append(result)

# =========================================================
# Conversation History
# =========================================================
for idx, item in enumerate(reversed(st.session_state.history[-1:])):
    st.divider()

    st.subheader(f"Question")
    st.write(item["question"])

    st.subheader("Answer")
    st.write(item["answer"])

    st.subheader("Sources")
    if item.get("sources"):
        for s in item["sources"]:
            st.markdown(f"- **{s['source_file']}** (Page {s['page']})")
    else:
        st.info("No sources.")

    usage = item.get("usage")
    if usage:
        st.caption(
            f"Tokens: {usage['total_tokens']} "
            f"(Prompt {usage['prompt_tokens']} / Completion {usage['completion_tokens']})"
        )

    # =====================================================
    # Feedback
    # =====================================================
    st.subheader("Feedback")

    feedback_record = get_feedback(
        question=item["question"],
        answer=item["answer"],
    )

    if feedback_record:
        st.info(
            f"Saved feedback: {feedback_record['feedback']}"
            + (
                f", comment: {feedback_record['comment']}"
                if feedback_record.get("comment")
                else ""
            )
        )

    existing_comment = feedback_record.get("comment") if feedback_record else ""
    existing_rating = feedback_record.get("rating") if feedback_record else -1

    col1, col2 = st.columns(2)

    helpful_key = _feedback_widget_key("helpful", item["question"], item["answer"])
    not_helpful_key = _feedback_widget_key("not_helpful", item["question"], item["answer"])
    comment_key = _feedback_widget_key("comment", item["question"], item["answer"])
    comment_submit_key = _feedback_widget_key("comment_submit", item["question"], item["answer"])

    with col1:
        if st.button("👍 Helpful", key=helpful_key):
            save_feedback(
                question=item["question"],
                answer=item["answer"],
                rating=1,
                comment=existing_comment,
            )
            st.rerun()

    with col2:
        if st.button("👎 Not Helpful", key=not_helpful_key):
            save_feedback(
                question=item["question"],
                answer=item["answer"],
                rating=0,
                comment=existing_comment,
            )
            st.rerun()

    comment = st.text_input(
        "Optional comment",
        value=existing_comment,
        key=comment_key,
    )

    if st.button("Submit Comment", key=comment_submit_key):
        save_feedback(
            question=item["question"],
            answer=item["answer"],
            rating=existing_rating,
            comment=comment,
        )
        st.rerun()
