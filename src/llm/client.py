import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    """
    Reliable OpenAI client wrapper.

    Guarantees:
      - .chat(system_prompt, user_prompt) -> str
      - .run(prompt) or .run(system_prompt, user_prompt) -> str (backwards compatible)
    Adds:
      - retries with exponential backoff
      - request timeout
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it before running."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", str(temperature)))
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages: List[dict] = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt or ""},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout_s,
                )
                if not resp.choices:
                    return ""
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                last_err = e
                # backoff: 1s, 2s, 4s...
                sleep_s = 2**attempt
                time.sleep(sleep_s)

        # If we exhausted retries, raise a clean error for UI/eval.
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def run(self, *args: Any, **kwargs: Any) -> str:
        """
        Backwards-compat adapter for older code calling `llm.run(...)`.

        Supports:
            llm.run(prompt)
            llm.run(system_prompt, user_prompt)
            llm.run(prompt="...")
        """
        if "prompt" in kwargs and len(args) == 0:
            system_prompt = "You are a helpful assistant."
            user_prompt = str(kwargs["prompt"])
        elif len(args) == 1:
            system_prompt = "You are a helpful assistant."
            user_prompt = str(args[0])
        elif len(args) >= 2:
            system_prompt = str(args[0])
            user_prompt = str(args[1])
        else:
            raise ValueError("LLMClient.run expected a prompt or (system_prompt, user_prompt).")

        return self.chat(system_prompt, user_prompt)
