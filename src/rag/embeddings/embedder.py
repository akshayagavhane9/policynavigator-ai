import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Embedder:
    """
    Stable embedding wrapper.

    Guarantees:
      - embed_documents(List[str]) -> List[List[float]]
      - embed_query(str) -> List[float]
    Backwards-compatible:
      - embed(List[str]) -> List[List[float]]
    """

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it before running."
            )

        self.client = OpenAI(api_key=api_key)
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Defensive: filter empty strings to avoid API complaining
        safe_texts = [t if (t and t.strip()) else " " for t in texts]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=safe_texts,
                    timeout=self.timeout_s,
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                time.sleep(2**attempt)

        raise RuntimeError(f"Embedding request failed after retries: {last_err}")

    def embed_query(self, text: str) -> List[float]:
        vecs = self.embed_documents([text])
        return vecs[0] if vecs else []

    # Backwards compatible alias (your VectorDB already uses embed()).
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
