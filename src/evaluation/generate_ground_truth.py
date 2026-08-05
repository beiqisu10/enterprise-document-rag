"""
Generate ground truth dataset for retrieval evaluation.

This script:
1. Loads indexed document chunks
2. Filters noisy chunks
3. Samples representative chunks
4. Uses OpenAI to generate evaluation questions
5. Saves ground truth CSV

Output: data/ground-truth-retrieval.csv
"""

import json
import random
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from src.config import settings
from src.ingestion.chunker import DocumentChunk
from src.retrieval.keyword_store import KeywordStore

# ==========================
# Configuration
# ==========================

MIN_CHUNK_LENGTH = 150
SAMPLE_SIZE = 300
QUESTIONS_PER_CHUNK = 1
OUTPUT_PATH = Path("data/ground-truth-retrieval.csv")
MODEL = "gpt-4o-mini"

# ==========================
# Prompt
# ==========================

PROMPT_TEMPLATE = """
You are generating evaluation questions for an enterprise RAG retrieval system.

Given the document chunk below, generate {n_questions} realistic user questions.

The questions must:
- Be answerable ONLY from this chunk
- Test whether a retrieval system can find this exact chunk
- Be specific and technically meaningful
- Avoid generic questions

Document chunk:
{content}

Return JSON in this format:
{{"questions": ["question 1", "question 2"]}}
""".strip()

# ==========================
# Load chunks
# ==========================

def load_chunks() -> list[DocumentChunk]:
    """Load chunks from KeywordStore which keeps original document chunks and metadata."""
    store = KeywordStore()
    store.load()

    chunks = []
    for doc in store.documents:
        chunks.append(DocumentChunk(
            content=doc["content"],
            source_file=doc["metadata"]["source_file"],
            page_number=doc["metadata"]["page_number"],
            chunk_index=doc["metadata"]["chunk_index"],
        ))

    print(f"Loaded {len(chunks)} chunks")
    return chunks

# ==========================
# Filter + Sampling
# ==========================

def filter_and_sample_chunks(chunks: list[DocumentChunk]):
    valid_chunks = []

    for chunk in chunks:
        content = chunk.content.strip()

        # Remove very short chunks
        if len(content) < MIN_CHUNK_LENGTH:
            continue

        # Remove obvious headers
        if content.startswith("AWS Well-Architected Framework") and len(content) < 200:
            continue

        valid_chunks.append(chunk)

    print(f"Valid chunks after filtering: {len(valid_chunks)}")

    if len(valid_chunks) <= SAMPLE_SIZE:
        return valid_chunks

    sampled = random.sample(valid_chunks, SAMPLE_SIZE)
    print(f"Sampled {len(sampled)} chunks")
    return sampled

# ==========================
# Generate Questions
# ==========================

def generate_questions(client: OpenAI, chunks: list[DocumentChunk]):
    results = []

    for chunk in tqdm(chunks, desc="Generating questions"):
        prompt = PROMPT_TEMPLATE.format(
            n_questions=QUESTIONS_PER_CHUNK,
            content=chunk.content[:2500],
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You generate JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            data = json.loads(response.choices[0].message.content)
            questions = data.get("questions", [])

            chunk_id = f"{chunk.source_file}_{chunk.page_number}_{chunk.chunk_index}"

            for question in questions[:QUESTIONS_PER_CHUNK]:
                results.append({
                    "question": question.strip(),
                    "chunk_id": chunk_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "chunk_content": chunk.content[:500],
                })

        except Exception as e:
            print(f"Failed chunk {chunk.chunk_index}: {e}")

    return results

# ==========================
# Main
# ==========================

def main():
    client = OpenAI(api_key=settings.openai_api_key)

    # 1. Load chunks
    chunks = load_chunks()

    # 2. Filter + sample
    chunks = filter_and_sample_chunks(chunks)

    # 3. Generate questions
    results = generate_questions(client, chunks)

    # 4. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved {len(df)} questions")
    print(OUTPUT_PATH)

if __name__ == "__main__":
    main()