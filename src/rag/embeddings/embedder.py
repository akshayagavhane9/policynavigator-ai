import os
from typing import List

from openai import OpenAI


class Embedder:
    """
    Thin wrapper around OpenAI embeddings.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["hello world", "another text"])
    """

    def __init__(self, model: str | None = None) -> None:
        # You can override via env var if you want
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export it before running the app."
            )

        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return a list of embedding vectors, one per input text.
        """
        if not texts:
            return []

        # OpenAI embeddings API
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # response.data is a list of objects with a `.embedding` attribute
        return [d.embedding for d in response.data]
