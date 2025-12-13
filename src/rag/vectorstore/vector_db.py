import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cosine_sim_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    mat: (N, D), vec: (D,)
    returns sims: (N,)
    """
    doc_norms = np.linalg.norm(mat, axis=1)
    q_norm = np.linalg.norm(vec)
    denom = doc_norms * q_norm
    denom[denom == 0] = 1e-10
    return (mat @ vec) / denom


def mmr_select(
    doc_embs: np.ndarray,
    query_emb: np.ndarray,
    doc_sims_to_query: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Maximum Marginal Relevance selection on a candidate set.

    doc_embs: (M, D)
    query_emb: (D,)
    doc_sims_to_query: (M,) precomputed cosine similarities to query
    """
    if k <= 0 or len(doc_embs) == 0:
        return []

    selected: List[int] = []
    candidate_indices = list(range(len(doc_embs)))

    # Precompute doc-doc cosine similarity matrix for diversity penalty
    # (M, M)
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    denom = norms @ norms.T
    denom[denom == 0] = 1e-10
    doc_doc_sims = (doc_embs @ doc_embs.T) / denom

    while len(selected) < min(k, len(candidate_indices)):
        if not selected:
            # pick the highest similarity to query first
            best = int(np.argmax(doc_sims_to_query))
            selected.append(best)
            candidate_indices.remove(best)
            continue

        best_score = -1e9
        best_idx = None

        for idx in candidate_indices:
            sim_to_query = float(doc_sims_to_query[idx])
            max_sim_to_selected = max(float(doc_doc_sims[idx, s]) for s in selected)

            score = lambda_mult * sim_to_query - (1.0 - lambda_mult) * max_sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected


class VectorDB:
    """
    Simple file-based vector store.

    Upgrades (top-25% wow factor):
      - retrieve top_k_raw candidates
      - optional MMR diversification re-ranking
      - metadata-based dedupe (source + chunk_id if present)
    """

    def __init__(self, persist_path: str = "data/vectorstore") -> None:
        self.persist_path = persist_path
        self.index_file = os.path.join(persist_path, "index.pkl")

        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if os.path.exists(self.index_file):
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)

            self.texts = data.get("texts", [])
            self.metadatas = data.get("metadatas", [])
            emb_list = data.get("embeddings", [])
            self.embeddings = (
                np.array(emb_list, dtype="float32") if emb_list else None
            )
        self._loaded = True

    def persist(self) -> None:
        os.makedirs(self.persist_path, exist_ok=True)
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(data, f)

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embedder: Any,
    ) -> None:
        self.load()
        if not texts:
            return

        # Support both embed() and embed_documents()
        if hasattr(embedder, "embed_documents"):
            new_vecs = embedder.embed_documents(texts)
        else:
            new_vecs = embedder.embed(texts)

        new_embs = np.array(new_vecs, dtype="float32")

        if self.embeddings is None:
            self.embeddings = new_embs
            self.texts = list(texts)
            self.metadatas = list(metadatas)
        else:
            self.embeddings = np.concatenate([self.embeddings, new_embs], axis=0)
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)

    def similarity_search(
        self,
        query: str,
        embedder: Any,
        k: int = 5,
        top_k_raw: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        dedupe: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k chunks for query.

        Args:
          top_k_raw: initial candidate pool size
          use_mmr: apply MMR diversification
          dedupe: remove duplicates by (source, chunk_id) if present
        """
        self.load()
        if self.embeddings is None or len(self.texts) == 0:
            return []

        # Embed query (support embed_query / embed)
        if hasattr(embedder, "embed_query"):
            q_vec = embedder.embed_query(query)
        else:
            q_vec = embedder.embed([query])[0]
        q_emb = np.array(q_vec, dtype="float32")

        sims_all = _cosine_sim_matrix(self.embeddings, q_emb)

        # Candidate pool
        raw_n = min(max(top_k_raw, k), len(self.texts))
        raw_indices = sims_all.argsort()[::-1][:raw_n]
        raw_indices = [int(i) for i in raw_indices]

        # Dedupe early (helps MMR)
        if dedupe:
            seen = set()
            filtered = []
            for idx in raw_indices:
                meta = self.metadatas[idx] if idx < len(self.metadatas) else {}
                key = (
                    str(meta.get("source", "unknown")),
                    str(meta.get("chunk_id", idx)),
                )
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(idx)
            raw_indices = filtered

        if not raw_indices:
            return []

        # If no MMR, take top-k as-is
        if not use_mmr:
            final_indices = raw_indices[: min(k, len(raw_indices))]
        else:
            # Build candidate embedding matrix for MMR
            cand_embs = self.embeddings[np.array(raw_indices, dtype=int)]
            cand_sims = np.array([float(sims_all[i]) for i in raw_indices], dtype="float32")

            selected_local = mmr_select(
                doc_embs=cand_embs,
                query_emb=q_emb,
                doc_sims_to_query=cand_sims,
                k=min(k, len(raw_indices)),
                lambda_mult=mmr_lambda,
            )
            final_indices = [raw_indices[i] for i in selected_local]

        results: List[Dict[str, Any]] = []
        for idx in final_indices:
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(sims_all[idx]),
                }
            )
        return results
