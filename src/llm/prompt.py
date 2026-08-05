"""
Prompt builder for Retrieval-Augmented Generation (RAG).

Responsible for:
- Formatting retrieved documents into context
- Building system and user prompts
- Limiting context length
- Enforcing grounded answers with citations
"""

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do NOT use outside knowledge.
- If the answer cannot be found in the context, reply exactly:
  "I don't know based on the provided documents."
- If the context is incomplete, say so.
- Do not make up facts or assumptions.
- Cite every factual statement using references such as [1], [2].
- If multiple documents support the same statement, cite all relevant references.
- Keep answers concise, accurate, and professional.
""".strip()


def build_prompt(query: str, documents: list[dict], max_chars: int = 12000) -> dict[str, str]:
    """
    Build prompts for the LLM.

    Args:
        query: User question.
        documents: Retrieved documents with content, metadata.source_file, metadata.page_number.
        max_chars: Maximum context length (approximate).

    Returns:
        {"system": "...", "user": "..."}
    """
    if not documents:
        return {
            "system": SYSTEM_PROMPT,
            "user": f"Question:\n{query}\n\nNo relevant context was retrieved.\n\nAnswer:",
        }

    context_sections = []
    total_chars = 0

    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {})
        section = (
            f"[{idx}]\n"
            f"Source: {metadata.get('source_file', 'Unknown')}\n"
            f"Page: {metadata.get('page_number', 'Unknown')}\n"
            f"Content:\n{doc.get('content', '').strip()}\n"
        )

        if total_chars + len(section) > max_chars:
            break

        context_sections.append(section)
        total_chars += len(section)

    context = "\n".join(context_sections)

    user_prompt = f"""
Context
=======
{context}

Question
========
{query}

Instructions
============
Answer ONLY using the context above.

Requirements:
1. Cite supporting sources using [1], [2], etc.
2. If the answer cannot be found, say:
   "I don't know based on the provided documents."
3. Do not invent information.
4. Keep the answer clear and concise.

Answer:
""".strip()

    return {"system": SYSTEM_PROMPT, "user": user_prompt}