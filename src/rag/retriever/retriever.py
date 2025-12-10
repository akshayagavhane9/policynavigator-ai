from typing import List, Dict, Any

from src.rag.embeddings.embedder import Embedder
from src.rag.vectorstore.vector_db import query_collection


_embedder = Embedder()


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 5,
    collection_name: str = "policies",
) -> List[Dict[str, Any]]:
    """
    Given a natural language question, embed it and query the vector store.
    Returns a list of chunks with text, metadata, and score.
    """
    query_embedding = _embedder.embed_query(question)
    results = query_collection(query_embedding, top_k=top_k, collection_name=collection_name)

    # Optional: sort by score (distance) if needed
    results = sorted(results, key=lambda x: x["score"])
    return results
