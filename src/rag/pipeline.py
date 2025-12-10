import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

try:
    from .loaders.pdf_loader import load_pdf
    from .loaders.docx_loader import load_docx
    from .loaders.text_loader import load_text
    from .preprocessors.cleaner import clean_text
    from .preprocessors.chunker import chunk_text
    from .embeddings.embedder import Embedder
    from .vectorstore.vector_db import upsert_chunks
    from .retriever.retriever import retrieve_relevant_chunks
except ImportError:
    # Fallback for when running directly
    from src.rag.loaders.pdf_loader import load_pdf
    from src.rag.loaders.docx_loader import load_docx
    from src.rag.loaders.text_loader import load_text
    from src.rag.preprocessors.cleaner import clean_text
    from src.rag.preprocessors.chunker import chunk_text
    from src.rag.embeddings.embedder import Embedder
    from src.rag.vectorstore.vector_db import upsert_chunks
    from src.rag.retriever.retriever import retrieve_relevant_chunks


_embedder = Embedder()


def _load_any(path: str) -> List[Dict[str, Any]]:
    """
    Dispatch loader based on file extension.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".pdf"]:
        return load_pdf(path)
    elif ext in [".docx"]:
        return load_docx(path)
    elif ext in [".txt", ".md"]:
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} for path {path}")


def index_documents(
    file_paths: List[str],
    collection_name: str = "policies",
    max_chars: int = 800,
    overlap: int = 200,
) -> None:
    """
    End-to-end indexing:
    - load docs
    - clean text
    - chunk
    - embed
    - store in vector DB
    """
    all_chunks: List[Dict[str, Any]] = []
    print(f"[RAG] Indexing documents: {file_paths}")

    for path in file_paths:
        print(f"[RAG] Loading: {path}")
        docs = _load_any(path)
        for doc in docs:
            print(f"[RAG] Cleaning text for doc: {doc['id']}")
            cleaned = clean_text(doc["text"])

            print(f"[RAG] Chunking text for doc: {doc['id']}")
            chunks = chunk_text(
                cleaned,
                max_chars=max_chars,
                overlap=overlap,
                doc_id=doc["id"],
                base_metadata=doc["metadata"],
            )
            all_chunks.extend(chunks)

    print(f"[RAG] Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("[RAG] No chunks to index.")
        return

    texts = [c["text"] for c in all_chunks]

    print("[RAG] Creating embeddings... (first time can be slow)")
    embeddings = _embedder.embed_documents(texts)
    print("[RAG] Embeddings created, upserting to vector store...")

    upsert_chunks(all_chunks, embeddings, collection_name=collection_name)
    print(f"[RAG] Indexed {len(all_chunks)} chunks into collection '{collection_name}'.")

def build_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a single context string from retrieved chunks,
    including light metadata for traceability.
    """
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        src = meta.get("source", "unknown_source")
        idx = meta.get("chunk_index", "?")
        header = f"[Source: {os.path.basename(src)} | Chunk: {idx}]"
        parts.append(header + "\n" + chunk["text"])
    return "\n\n---\n\n".join(parts)


def retrieve_context(
    question: str,
    top_k: int = 5,
    collection_name: str = "policies",
) -> str:
    """
    High-level helper: retrieve relevant chunks and return formatted context string.
    """
    chunks = retrieve_relevant_chunks(
        question=question,
        top_k=top_k,
        collection_name=collection_name,
    )
    return build_context_from_chunks(chunks)


if __name__ == "__main__":
    # Small manual test:
    # 1. Put a sample PDF/TXT in data/kb_raw/
    # 2. Run: python -m src.rag.pipeline
    SAMPLE_DOC = os.path.join("data", "kb_raw", "Academic_integrity_policyNEU.pdf")

    if os.path.exists(SAMPLE_DOC):
        index_documents([SAMPLE_DOC])
        ctx = retrieve_context("What is the late submission policy?")
        print("=== Retrieved Context ===")
        print(ctx)
    else:
        print("Put a sample_policy.pdf in data/kb_raw/ to test indexing.")
