import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np


class VectorDB:
    """
    Very simple file-based vector store.

    - Stores texts, metadatas and embeddings in a pickle file.
    - Uses the provided Embedder to create embeddings.
    - Provides similarity_search(query, embedder, k) used by main.py.
    """

    def __init__(self, persist_path: str = "data/vectorstore") -> None:
        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, "index.pkl")

        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        self._loaded = False

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load existing index from disk (if it exists)."""
        if self._loaded:
            return

        if os.path.exists(self.index_file):
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)

            self.texts = data.get("texts", [])
            self.metadatas = data.get("metadatas", [])
            emb_list = data.get("embeddings", [])
            if emb_list:
                self.embeddings = np.array(emb_list, dtype="float32")
            else:
                self.embeddings = None

        self._loaded = True

    def persist(self) -> None:
        """Persist current index to disk."""
        os.makedirs(self.persist_path, exist_ok=True)

        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }

        with open(self.index_file, "wb") as f:
            pickle.dump(data, f)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embedder: Any,
    ) -> None:
        """
        Add texts + metadatas to the index and compute embeddings.

        The embedder is expected to have a method:
            embed(texts: List[str]) -> List[List[float]]
        """
        self.load()

        if not texts:
            return

        new_embs = np.array(embedder.embed(texts), dtype="float32")

        if self.embeddings is None:
            self.embeddings = new_embs
            self.texts = list(texts)
            self.metadatas = list(metadatas)
        else:
            self.embeddings = np.concatenate([self.embeddings, new_embs], axis=0)
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def similarity_search(
        self,
        query: str,
        embedder: Any,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k similar chunks for the query.

        We return a list of dicts with 'text' and 'metadata' keys, so that
        main._get_doc_text/_get_doc_metadata work correctly.
        """
        self.load()

        if self.embeddings is None or len(self.texts) == 0:
            return []

        # Embed query (we expect embedder.embed to return List[List[float]])
        q_emb = np.array(embedder.embed([query])[0], dtype="float32")

        # Cosine similarity between query and all docs
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        denom = doc_norms * q_norm
        denom[denom == 0] = 1e-10

        sims = (self.embeddings @ q_emb) / denom

        # Get top-k indices (highest similarity)
        k = min(k, len(self.texts))
        top_indices = sims.argsort()[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.texts[int(idx)],
                    "metadata": self.metadatas[int(idx)],
                    "score": float(sims[int(idx)]),
                }
            )

        return results
