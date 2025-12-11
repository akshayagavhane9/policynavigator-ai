import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """
    Simple wrapper around OpenAI chat completions.

    Usage:
        llm = LLMClient()
        answer = llm.chat(system_prompt, user_prompt)
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export it before running the app."
            )

        self.client = OpenAI(api_key=api_key)
        # You can override via env var if you want
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the OpenAI chat completion API and return the assistant's reply text.
        """
        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        if not response.choices:
            return ""

        content = response.choices[0].message.content or ""
        return content.strip()
    
    def run(self, *args, **kwargs) -> str:
        """
        Backwards-compat adapter for older code that calls `llm.run(...)`.

        Supports:
            llm.run(prompt)
            llm.run(system_prompt, user_prompt)
        and internally delegates to `self.chat(...)`.
        """
        # Only keyword we care about is `prompt` if someone used it
        if "prompt" in kwargs and len(args) == 0:
            user_prompt = kwargs["prompt"]
            system_prompt = "You are a helpful assistant."
        elif len(args) == 1:
            # Single positional: treat as user prompt with default system
            user_prompt = args[0]
            system_prompt = "You are a helpful assistant."
        elif len(args) >= 2:
            # Two positionals: (system, user)
            system_prompt, user_prompt = args[0], args[1]
        else:
            raise ValueError("LLMClient.run expected 1 or 2 positional arguments.")

        return self.chat(system_prompt, user_prompt)
