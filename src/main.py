import os
import time
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv

from src.rag.loaders.pdf_loader import load_pdf
from src.rag.loaders.docx_loader import load_docx
from src.rag.loaders.text_loader import load_text
from src.rag.preprocessors.cleaner import clean_text
from src.rag.preprocessors.chunker import chunk_text
from src.rag.embeddings.embedder import Embedder
from src.rag.vectorstore.vector_db import VectorDB
from src.llm.client import LLMClient

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

KB_RAW_DIR = "data/kb_raw"
KB_PROCESSED_DIR = "data/kb_processed"
VECTOR_DB_PATH = "data/vectorstore"

# Retrieval/guardrail defaults (grade-friendly)
DEFAULT_TOP_K = 5
DEFAULT_TOP_K_RAW = 20
DEFAULT_USE_MMR = True
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_DEDUPE = True

# If max similarity is below this, we should abstain instead of risking hallucination
DEFAULT_ABSTAIN_SIM_THRESHOLD = 0.35


def _ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    os.makedirs(KB_RAW_DIR, exist_ok=True)
    os.makedirs(KB_PROCESSED_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)


# ---------------------------------------------------------------------
# Simple helpers for confidence & hallucination
# ---------------------------------------------------------------------


def _confidence_label_from_score(score: float) -> str:
    """Map a similarity-based score into a label."""
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _detect_hallucination(
    similarities: List[float], threshold: float = 0.5
) -> Tuple[bool, str, float]:
    """
    Very simple hallucination detector:

    - Look at the max similarity among retrieved chunks.
    - If max_sim < threshold → high hallucination risk, flag=True.
    - If threshold <= max_sim < 0.7 → medium risk, no hard flag.
    - Otherwise → low risk.

    Returns:
        (hallucination_flag, risk_label, max_similarity)
    """
    if not similarities:
        return True, "high", 0.0

    max_sim = max(similarities)

    if max_sim < threshold:
        hallucinated = True
        risk = "high"
    elif max_sim < 0.7:
        hallucinated = False
        risk = "medium"
    else:
        hallucinated = False
        risk = "low"

    return hallucinated, risk, max_sim


# ---------------------------------------------------------------------
# Ingestion & indexing
# ---------------------------------------------------------------------


def ingest_and_index_documents(file_paths: List[str]) -> int:
    """
    Ingest raw files, chunk them, embed them, and update the vector store.

    Supports:
      - PDF (page-aware; preserves page metadata)
      - DOCX
      - TXT/MD

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

        # ----------------- PDF: page-aware ingestion -----------------
        if ext == ".pdf":
            try:
                page_docs = load_pdf(path)  # List[{"text":..., "metadata":{page,...}}]
            except Exception as e:
                print(f"[INGEST] Failed to load PDF {path}: {e}")
                continue

            if not page_docs:
                print(f"[INGEST] No pages extracted from {path}, skipping.")
                continue

            base = os.path.splitext(os.path.basename(path))[0]

            for page_doc in page_docs:
                raw_text = page_doc.get("text", "")
                meta = page_doc.get("metadata", {}) or {}
                page_num: Optional[int] = meta.get("page")

                if not raw_text or not str(raw_text).strip():
                    continue

                cleaned = clean_text(raw_text)
                chunks = chunk_text(cleaned)

                # Save per-page cleaned text (debug-friendly)
                page_suffix = f"_p{page_num}" if page_num is not None else ""
                cleaned_out = os.path.join(KB_PROCESSED_DIR, f"{base}{page_suffix}.txt")
                with open(cleaned_out, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                for i, ch in enumerate(chunks):
                    all_chunks.append(
                        {
                            "id": f"{base}{page_suffix}_{i}",
                            "text": ch,
                            "source": os.path.basename(path),
                            "page": page_num,
                        }
                    )

            continue  # move to next file

        # ----------------- DOCX / TXT / MD: single-doc ingestion -----------------
        try:
            if ext == ".docx":
                raw_text = load_docx(path)
            else:
                raw_text = load_text(path)
        except Exception as e:
            print(f"[INGEST] Failed to load {path}: {e}")
            continue

        if not raw_text or not str(raw_text).strip():
            print(f"[INGEST] No text extracted from {path}, skipping.")
            continue

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
                    "page": None,
                }
            )

    if not all_chunks:
        print("[INGEST] No chunks generated from any file.")
        return 0

    # ----------------- embeddings & vector store -----------------
    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {"source": c["source"], "chunk_id": c["id"], "page": c.get("page")}
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
    k: int = DEFAULT_TOP_K,
    eval_mode: bool = False,
    # ranking/ablation switches
    use_mmr: bool = DEFAULT_USE_MMR,
    top_k_raw: int = DEFAULT_TOP_K_RAW,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    dedupe: bool = DEFAULT_DEDUPE,
    # abstain guardrail (prevents “confident wrong”)
    abstain_sim_threshold: float = DEFAULT_ABSTAIN_SIM_THRESHOLD,
) -> Dict[str, Any]:
    """
    Answer a policy question using the indexed vector store.

    Returns dict with:
      - answer, citations, confidence_label, confidence_score,
        hallucination_flag, hallucination_risk, used_query, latency_ms
    """
    start_time = time.time()
    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "citations": [],
            "confidence_label": "low",
            "confidence_score": 0.0,
            "hallucination_flag": False,
            "hallucination_risk": "high",
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
    # Adaptive MMR: avoid hurting relevance on small/focused KBs
    # ------------------------------------------------------------------
    adaptive_use_mmr = bool(use_mmr)
    if adaptive_use_mmr:
        # If you don't have enough raw candidates, MMR can over-diversify and hurt relevance.
        if int(top_k_raw) < 15:
            adaptive_use_mmr = False

    # ------------------------------------------------------------------
    # Retrieve top-k chunks (top_k_raw + MMR + dedupe)
    # ------------------------------------------------------------------
    try:
        docs = vectordb.similarity_search(
            used_query,
            embedder=embedder,
            k=k,
            top_k_raw=top_k_raw,
            use_mmr=adaptive_use_mmr,   # ✅ IMPORTANT
            mmr_lambda=mmr_lambda,
            dedupe=dedupe,
        )
    except Exception as e:
        print(f"[RETRIEVAL] similarity_search failed: {e}")
        docs = []

    if not docs:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "answer": "I couldn't find relevant information for this question in the indexed documents.",
            "citations": [],
            "confidence_label": "low",
            "confidence_score": 0.1,
            "hallucination_flag": True,
            "hallucination_risk": "high",
            "used_query": used_query,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Build context, citations & similarity list
    # ------------------------------------------------------------------
    context_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []
    similarities: List[float] = []

    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        meta = doc.get("metadata", {}) or {}

        score = doc.get("score", meta.get("score", 0.0))
        try:
            sim = float(score)
        except Exception:
            sim = 0.0

        similarities.append(sim)

        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        page = meta.get("page")

        context_blocks.append(
            f"[{i + 1}] Source: {source} (chunk {chunk_id}, page {page}) | sim={sim:.2f}\n{text}"
        )
        citations.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "page": page,
                "rank": i + 1,
                "similarity": sim,
                "text": text,
            }
        )

    context_str = "\n\n".join(context_blocks)

    # ------------------------------------------------------------------
    # Hallucination detection & confidence
    # ------------------------------------------------------------------
    hallucination_flag, hallucination_risk, max_sim = _detect_hallucination(
        similarities, threshold=0.5
    )
    confidence_score = float(max_sim)
    confidence_label = _confidence_label_from_score(confidence_score)

    # ------------------------------------------------------------------
    # Abstain guardrail
    # ------------------------------------------------------------------
    if confidence_score < float(abstain_sim_threshold):
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "answer": "Not covered in the provided policy excerpts.",
            "citations": citations,
            "confidence_label": "low",
            "confidence_score": confidence_score,
            "hallucination_flag": True,
            "hallucination_risk": "high",
            "used_query": used_query,
            "latency_ms": latency_ms,
        }

    grounding_warning = ""
    if hallucination_flag:
        grounding_warning = (
            "Important: The retrieved policy passages do NOT strongly match the question. "
            "If the answer is not clearly supported by the excerpts, explicitly say that the "
            "policy is unclear or not covered, and encourage the student to check official NEU documents."
        )

    # ------------------------------------------------------------------
    # Style instructions
    # ------------------------------------------------------------------
    if eval_mode:
        style_instruction = (
            "EVALUATION MODE:\n"
            "- Answer in AT MOST 2 sentences.\n"
            "- Use only information explicitly present in the context.\n"
            "- Prefer quoting one short clause if possible.\n"
            "- Do not add extra explanation, examples, or advice.\n"
            "- If the context does not answer the question, reply exactly: "
            "\"Not covered in the provided policy excerpts.\""
        )
    else:
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
        "You must stay faithful to the provided context and avoid hallucinations. "
        "If the context does not clearly answer the question, say so and recommend checking the "
        "official policy documents.\n\n"
        f"{style_instruction}\n\n"
        f"{grounding_warning}"
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Below are policy excerpts retrieved as potentially relevant context:\n\n"
        f"{context_str}\n\n"
        "Using ONLY the information above, answer the question.\n"
        "If the text does not clearly answer it:\n"
        '- In evaluation mode: reply exactly "Not covered in the provided policy excerpts."\n'
        "- Otherwise: say the policy is unclear or not covered and encourage checking official policy."
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

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "hallucination_flag": hallucination_flag,
        "hallucination_risk": hallucination_risk,
        "used_query": used_query,
        "latency_ms": latency_ms,
        "ranking": {
            "use_mmr": adaptive_use_mmr,   # report actual
            "top_k_raw": top_k_raw,
            "mmr_lambda": mmr_lambda,
            "dedupe": dedupe,
        },
    }


if __name__ == "__main__":
    """
    Small manual smoke test:
      python -m src.main
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

        q = "How does Northeastern define cheating in the academic integrity policy?"

        print("\n--- BASELINE (no rewrite, no MMR) ---")
        r0 = answer_question(q, rewrite_query=False, use_mmr=False, k=5, top_k_raw=5)
        print(r0.get("answer"))
        print("Confidence:", r0.get("confidence_label"), r0.get("confidence_score"))

        print("\n--- IMPROVED (rewrite + MMR) ---")
        r1 = answer_question(q, rewrite_query=True, use_mmr=True, k=5, top_k_raw=20)
        print(r1.get("answer"))
        print("Confidence:", r1.get("confidence_label"), r1.get("confidence_score"))

    else:
        print("[MAIN] No files in kb_raw; add a policy PDF/TXT first.")
