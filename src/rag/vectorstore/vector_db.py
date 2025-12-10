import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings


# Ensure the directory exists
VECTOR_DB_DIR = os.path.join("data", "kb_processed", "chroma_db")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

_client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR,
    settings=Settings(allow_reset=True),
)


def get_collection(name: str = "policies"):
    """
    Get or create a Chroma collection.
    """
    return _client.get_or_create_collection(name=name)


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = "policies",
) -> None:
    """
    Upsert chunk docs + embeddings into the vector store.
    """
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings lengths do not match")

    collection = get_collection(collection_name)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def query_collection(
    query_embedding: List[float],
    top_k: int = 5,
    collection_name: str = "policies",
) -> List[Dict[str, Any]]:
    """
    Query the vector store with a single embedding.
    Returns list of dicts: {'text', 'metadata', 'score'}
    """
    collection = get_collection(collection_name)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # result fields: ids, documents, metadatas, distances
    docs = []
    for idx in range(len(result["documents"][0])):
        docs.append({
            "id": result["ids"][0][idx],
            "text": result["documents"][0][idx],
            "metadata": result["metadatas"][0][idx],
            "score": result["distances"][0][idx],
        })

    return docs
