import os
import time
from typing import Any, Dict, List

from src.rag.loaders.pdf_loader import load_pdf
from src.rag.loaders.docx_loader import load_docx
from src.rag.loaders.text_loader import load_text
from src.rag.preprocessors.cleaner import clean_text
from src.rag.preprocessors.chunker import chunk_text
from src.rag.embeddings.embedder import Embedder
from src.rag.vectorstore.vector_db import VectorDB
from src.llm.client import LLMClient

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

# We keep paths relative to the project root (where you run streamlit).
KB_RAW_DIR = "data/kb_raw"
KB_PROCESSED_DIR = "data/kb_processed"
VECTOR_DB_PATH = "data/vectorstore"


def _ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    os.makedirs(KB_RAW_DIR, exist_ok=True)
    os.makedirs(KB_PROCESSED_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)


# ---------------------------------------------------------------------
# Ingestion & indexing
# ---------------------------------------------------------------------


def ingest_and_index_documents(file_paths: List[str]) -> int:
    """
    Ingest the given raw files, chunk them, embed them, and update the vector store.

    Called from the Streamlit sidebar when you click "Index Documents".

    Args:
        file_paths: list of paths inside data/kb_raw/

    Returns:
        int: number of chunks indexed.
    """
    if not file_paths:
        print("[INGEST] No files passed to ingest_and_index_documents.")
        return 0

    _ensure_dirs()

    embedder = Embedder()
    vectordb = VectorDB(persist_path=VECTOR_DB_PATH)

    all_chunks: List[Dict[str, Any]] = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        print(f"[INGEST] Processing {path} (ext={ext})")

        # ----------------- loading -----------------
        try:
            if ext == ".pdf":
                raw_text = load_pdf(path)
            elif ext == ".docx":
                raw_text = load_docx(path)
            else:
                raw_text = load_text(path)
        except Exception as e:
            print(f"[INGEST] Failed to load {path}: {e}")
            continue

        if not raw_text or not str(raw_text).strip():
            print(f"[INGEST] No text extracted from {path}, skipping.")
            continue

        print(f"[INGEST] Extracted {len(str(raw_text))} characters from {path}")

        # ----------------- cleaning & chunking -----------------
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)
        print(f"[INGEST] Created {len(chunks)} chunks from {path}")

        base = os.path.splitext(os.path.basename(path))[0]
        cleaned_out = os.path.join(KB_PROCESSED_DIR, f"{base}.txt")
        with open(cleaned_out, "w", encoding="utf-8") as f:
            f.write(cleaned)

        for i, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{base}_{i}",
                    "text": ch,
                    "source": os.path.basename(path),
                }
            )

    if not all_chunks:
        print("[INGEST] No chunks generated from any file.")
        return 0

    # ----------------- embeddings & vector store -----------------
    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {"source": c["source"], "chunk_id": c["id"]}
        for c in all_chunks
    ]

    vectordb.add_texts(texts=texts, metadatas=metadatas, embedder=embedder)
    vectordb.persist()

    print(f"[INGEST] Indexed total {len(all_chunks)} chunks.")
    return len(all_chunks)


# ---------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------


def answer_question(
    question: str,
    answer_style: str = "Strict policy quote",
    rewrite_query: bool = True,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Answer a policy question using the indexed vector store.

    This is what the Streamlit UI calls when you click "Generate Answer".

    Returns a dict that ALWAYS has at least:
        - "answer": str
        - "citations": List[Dict[str, Any]]
        - "confidence_label": "high" | "medium" | "low"
        - "confidence_score": float in [0,1]
        - "used_query": str
        - "latency_ms": int
    """
    start_time = time.time()
    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "citations": [],
            "confidence_label": "low",
            "confidence_score": 0.0,
            "used_query": "",
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    _ensure_dirs()

    embedder = Embedder()
    vectordb = VectorDB(persist_path=VECTOR_DB_PATH)
    vectordb.load()

    llm = LLMClient()

    # ------------------------------------------------------------------
    # Optional: rewrite query for retrieval
    # ------------------------------------------------------------------
    used_query = question
    if rewrite_query:
        try:
            rewrite_system = (
                "You are a query rewriter for a university policy RAG system. "
                "Rewrite the user's question into a concise search query that will "
                "match the relevant policy clauses. Do NOT answer the question."
            )
            rewrite_user = f"Original question:\n{question}"
            rewritten = llm.chat(rewrite_system, rewrite_user)
            if isinstance(rewritten, str) and rewritten.strip():
                used_query = rewritten.strip()
        except Exception as e:
            print(f"[REWRITE] Failed to rewrite query: {e}")
            used_query = question

    # ------------------------------------------------------------------
    # Retrieve top-k chunks
    # ------------------------------------------------------------------
    try:
        docs = vectordb.similarity_search(used_query, embedder=embedder, k=k)
    except Exception as e:
        print(f"[RETRIEVAL] similarity_search failed: {e}")
        docs = []

    latency_ms = int((time.time() - start_time) * 1000)

    if not docs:
        return {
            "answer": "I couldn't find relevant information for this question in the indexed documents.",
            "citations": [],
            "confidence_label": "low",
            "confidence_score": 0.1,
            "used_query": used_query,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Build context & citations
    # ------------------------------------------------------------------
    context_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        meta = doc.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk_id", f"chunk_{i}")

        context_blocks.append(
            f"[{i + 1}] Source: {source} (chunk {chunk_id})\n{text}"
        )
        citations.append(
            {"source": source, "chunk_id": chunk_id, "rank": i + 1}
        )

    context_str = "\n\n".join(context_blocks)

    # ------------------------------------------------------------------
    # Style instructions
    # ------------------------------------------------------------------
    if answer_style.lower().startswith("strict"):
        style_instruction = (
            "Answer ONLY using direct information from the context. "
            "Quote or closely paraphrase relevant sentences. "
            "If the answer is not present, say you cannot answer based on the provided policy text."
        )
    else:
        style_instruction = (
            "Explain the answer in clear, student-friendly language, "
            "but base everything strictly on the context. "
            "Do not invent policy rules that are not in the text."
        )

    system_prompt = (
        "You are PolicyNavigator AI, an assistant that answers questions about university policies. "
        "You must stay faithful to the provided context and avoid hallucinations."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"{style_instruction}\n\n"
        f"Context:\n{context_str}"
    )

    # ------------------------------------------------------------------
    # Call LLM for final answer
    # ------------------------------------------------------------------
    try:
        answer_text = llm.chat(system_prompt, user_prompt)
        if not isinstance(answer_text, str) or not answer_text.strip():
            answer_text = "I’m sorry, I couldn’t generate an answer."
    except Exception as e:
        print(f"[LLM] Failed to generate answer: {e}")
        answer_text = "Something went wrong while consulting the policy documents."

    # Simple confidence heuristic: more retrieved chunks ⇒ higher confidence
    if len(docs) >= 6:
        conf_label, conf_score = "high", 0.9
    elif len(docs) >= 3:
        conf_label, conf_score = "medium", 0.6
    else:
        conf_label, conf_score = "low", 0.3

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence_label": conf_label,
        "confidence_score": conf_score,
        "used_query": used_query,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------
# Optional: allow running some quick manual tests from CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Small manual smoke test:
      python -m src.main
    (Assumes you already have files in data/kb_raw and OPENAI_API_KEY set.)
    """
    _ensure_dirs()
    kb_files = [
        os.path.join(KB_RAW_DIR, f)
        for f in os.listdir(KB_RAW_DIR)
        if os.path.isfile(os.path.join(KB_RAW_DIR, f))
    ]
    print(f"[MAIN] Found {len(kb_files)} files in {KB_RAW_DIR}: {kb_files}")

    if kb_files:
        ingest_and_index_documents(kb_files)
        res = answer_question(
            "How does Northeastern define cheating in the academic integrity policy?",
            answer_style="Strict policy quote",
        )
        print("\n[ANSWER]")
        print(res.get("answer"))
        print("\n[CITATIONS]")
        print(res.get("citations"))
    else:
        print("[MAIN] No files in kb_raw; add a policy PDF/TXT first.")
