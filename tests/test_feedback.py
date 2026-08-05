from pathlib import Path

from src.frontend.feedback import load_feedback, save_feedback


def test_save_feedback_updates_existing_record(tmp_path):
    feedback_file = tmp_path / "feedback.jsonl"
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Monkeypatch the FEEDBACK_FILE path in the module
    import src.frontend.feedback as feedback_module
    feedback_module.FEEDBACK_FILE = feedback_file

    question = "What is RAG?"
    answer = "RAG means retrieval augmented generation."

    save_feedback(question=question, answer=answer, rating=1)
    save_feedback(question=question, answer=answer, rating=0)
    save_feedback(question=question, answer=answer, rating=-1, comment="Needs more detail")

    records = load_feedback()
    assert len(records) == 1
    record = records[0]
    assert record["question"] == question
    assert record["answer"] == answer
    assert record["rating"] == 0
    assert record["feedback"] == "negative"
    assert record["comment"] == "Needs more detail"
