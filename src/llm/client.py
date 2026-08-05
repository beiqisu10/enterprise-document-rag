from openai import OpenAI, APIError, RateLimitError
from src.config import settings
import time

from src.llm.prompt import SYSTEM_PROMPT


class LLMClient:
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def generate(self, prompt: dict | str, max_retries: int = 3) -> dict:
        """
        Generate answer from LLM.

        Args:
            prompt: Either a raw string (legacy) or {"system": ..., "user": ...}
            max_retries: Retry on rate limit / server error.

        Returns:
            {
                "answer": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "model": str
            }
        """
        if isinstance(prompt, str):
            system = SYSTEM_PROMPT
            user = prompt
        else:
            system = prompt.get("system", SYSTEM_PROMPT)
            user = prompt.get("user", "")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )

                return {
                    "answer": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": self.model,
                }

            except RateLimitError:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
                raise

            except APIError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

    def generate_stream(self, prompt: dict | str):
        """
        Stream tokens for real-time UI display.
        Yields text chunks.
        """
        if isinstance(prompt, str):
            system = SYSTEM_PROMPT
            user = prompt
        else:
            system = prompt.get("system", SYSTEM_PROMPT)
            user = prompt.get("user", "")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content