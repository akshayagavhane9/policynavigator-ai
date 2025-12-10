from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around a SentenceTransformer model for embedding docs & queries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents. Returns list of embedding vectors (lists of floats).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        """
        return self.embed_documents([text])[0]
