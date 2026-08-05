"""
End-to-end LLM evaluation for RAG.

Compares different prompt strategies using:
- Same retrieval pipeline
- Same LLM model
- LLM-as-a-Judge evaluation

The goal is to select the best prompt configuration.
"""

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd
from tqdm.auto import tqdm

from src.evaluation.llm_judge import LLMJudge
from src.llm.client import LLMClient
from src.llm.prompt import build_prompt


@dataclass
class LLMConfig:
    """Configuration for one LLM evaluation experiment. Different configs represent different prompt strategies."""
    name: str
    prompt_version: str
    top_k: int = 5


class LLMEvaluator:
    def __init__(
          self,
          ground_truth_path: str | Path,
          retriever,
      ):
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.retriever =retriever
        self.judge = LLMJudge()  # Judge model
        self.llm_client = LLMClient(model="gpt-4.1-mini")  # Production LLM

    def _load_ground_truth(self, path: str | Path) -> list[dict]:
        df = pd.read_csv(path)
        required = {"question", "chunk_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Ground truth missing columns: {missing}")
        return df.to_dict(orient="records")

    def _build_prompt_by_version(self, query: str, docs: list[dict], version: str) -> dict:
        if version == "basic":
            return build_prompt(query=query, documents=docs)

        elif version == "strict":
            prompt = build_prompt(query=query, documents=docs)
            prompt["system"] = """
You are an enterprise knowledge assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not found in the context, say: "I don't know."
3. Do not use external knowledge.
4. Cite every factual statement using references: [1], [2], etc.
""".strip()
            return prompt

        else:
            raise ValueError(f"Unknown prompt version: {version}")

    def evaluate_config(self, config: LLMConfig, sample_size: int | None = None) -> dict:
        items = self.ground_truth.copy()
        if sample_size and len(items) > sample_size:
            items = random.sample(items, sample_size)

        judge_inputs = []
        for item in tqdm(items, desc=f"Evaluating {config.name}"):
            query = item["question"]

            # 1. Retrieval
            docs = self.retriever.search(query, top_k=config.top_k)

            # 2. Prompt construction
            prompt = self._build_prompt_by_version(query, docs, config.prompt_version)

            # 3. LLM generation
            try:
                response = self.llm_client.generate(prompt)
                answer = response["answer"]
            except Exception as e:
                answer = f"[LLM ERROR: {e}]"

            # 4. Prepare judge context
            context = "\n\n".join([f"[{i+1}] {doc['content']}" for i, doc in enumerate(docs)])

            judge_inputs.append({
                "question": query,
                "answer": answer,
                "context": context,
                "config": config.name,
            })

        # 5. LLM Judge
        judged = self.judge.evaluate_batch(judge_inputs)
        return self._aggregate(judged, config)

    def _aggregate(self, judged: list[dict], config: LLMConfig) -> dict:
        n = len(judged)
        if n == 0:
            return {"config": config.name, "n": 0}

        correctness = sum(x["correctness"] for x in judged) / n
        faithfulness = sum(x["faithfulness"] for x in judged) / n
        citation = sum(x["citation_quality"] for x in judged) / n
        helpfulness = sum(x["helpfulness"] for x in judged) / n

        overall = correctness * 0.4 + faithfulness * 0.3 + citation * 0.2 + helpfulness * 0.1

        return {
            "config": config.name,
            "prompt": config.prompt_version,
            "top_k": config.top_k,
            "n": n,
            "avg_correctness": correctness,
            "avg_faithfulness": faithfulness,
            "avg_citation_quality": citation,
            "avg_helpfulness": helpfulness,
            "overall_score": overall,
            "hallucination_rate": sum(1 for x in judged if x.get("hallucinated")) / n,
        }

    def evaluate_all(self, configs: list[LLMConfig], sample_size: int | None = None) -> list[dict]:
        results = []
        for config in configs:
            results.append(self.evaluate_config(config, sample_size))
        return results

    def print_report(self, results: list[dict]) -> None:
        if not results:
            print("No results.")
            return

        print("=" * 100)
        print(f"{'Config':20}{'Correct':12}{'Faithful':12}{'Halluc.%':12}{'Citation':12}{'Helpful':12}{'Overall':12}")
        print("-" * 100)

        for r in results:
            print(f"{r['config']:20}{r['avg_correctness']:<12.2f}{r['avg_faithfulness']:<12.2f}{r['hallucination_rate']:<12.2%}{r['avg_citation_quality']:<12.2f}{r['avg_helpfulness']:<12.2f}{r['overall_score']:<12.2f}")

        print("=" * 100)
        best = max(results, key=lambda x: x["overall_score"])
        print(f"\n🏆 Best Prompt: {best['config']}")
        print(f"Overall Score: {best['overall_score']:.2f}/5")