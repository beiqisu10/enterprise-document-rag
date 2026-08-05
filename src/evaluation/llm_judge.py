"""
LLM-as-a-Judge for evaluating RAG output quality.

Evaluates:
- Correctness: Is the answer factually correct based on the context?
- Faithfulness: Does the answer contain hallucinations?
- Citation Quality: Are sources cited correctly?
- Helpfulness: Does it fully answer the user's question?
"""

import json
from collections.abc import Sequence

from openai import OpenAI

from src.config import settings


JUDGE_PROMPT = """
You are a STRICT evaluator. Be critical and picky.

Score 1-5 for each criterion. A score of 5 means PERFECT, almost never given.
Score 3 means "acceptable but flawed". Score 1 means "completely wrong".

Criteria:
1. correctness: Does the answer match the context EXACTLY? Any minor deviation = -1 point.
2. faithfulness: Any information not explicitly in the context = hallucination. Score 1 if hallucinated.
3. citation_quality: Every factual claim must have a citation like [1], [2]. Missing citation = -2 points. Wrong citation = score 1.
4. helpfulness: Does it directly answer the question? Vague or overly cautious = lower score.

Also output:
- hallucinated: true if ANY info is not in the context
- refused_correctly: true only if the context truly lacks the answer AND the assistant said "I don't know"

JSON format ONLY:
{{
  "correctness": 4,
  "faithfulness": 3,
  "citation_quality": 5,
  "helpfulness": 4,
  "hallucinated": false,
  "refused_correctly": true,
  "reasoning": "specific critique here"
}}

Context:
{context}

Question:
{question}

Answer:
{answer}
""".strip()


class LLMJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict:
        prompt = JUDGE_PROMPT.format(
            context=context[:8000],  # judge prompt 不要太长
            question=question,
            answer=answer,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict but fair evaluator. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result["judge_model"] = self.model
        return result

    def evaluate_batch(
        self,
        items: Sequence[dict],
    ) -> list[dict]:
        """
        Args:
            items: List of {"question": ..., "answer": ..., "context": ...}
        """
        results = []
        for item in items:
            score = self.evaluate(
                question=item["question"],
                answer=item["answer"],
                context=item["context"],
            )
            results.append({**item, **score})
        return results