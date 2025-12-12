# scripts/smoke_test.py

import os
import sys
from pathlib import Path
from pprint import pprint

# Ensure project root is on sys.path so `src` imports work
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

    
from src.main import (
    KB_RAW_DIR,
    ingest_and_index_documents,
    answer_question,
)
from src.rag.embeddings.embedder import Embedder
from src.rag.vectorstore.vector_db import VectorDB


def test_ingestion():
    print("\n=== 1) INGESTION & INDEXING TEST ===")
    if not os.path.isdir(KB_RAW_DIR):
        print(f"[ERROR] KB_RAW_DIR does not exist: {KB_RAW_DIR}")
        return

    kb_files = [
        os.path.join(KB_RAW_DIR, f)
        for f in os.listdir(KB_RAW_DIR)
        if os.path.isfile(os.path.join(KB_RAW_DIR, f))
    ]

    if not kb_files:
        print(f"[ERROR] No files found in {KB_RAW_DIR}. Add at least one PDF/DOCX/TXT.")
        return

    print(f"[INFO] Found {len(kb_files)} file(s):")
    for f in kb_files:
        print("  -", f)

    n_chunks = ingest_and_index_documents(kb_files)
    print(f"[RESULT] Ingested and indexed {n_chunks} chunks.")


def test_vector_search():
    print("\n=== 2) VECTOR DB SEARCH TEST ===")
    embedder = Embedder()
    vectordb = VectorDB(persist_path="data/vectorstore")
    vectordb.load()

    query = "late submission penalty"
    print(f"[INFO] Query: {query!r}")

    try:
        docs = vectordb.similarity_search(query, embedder=embedder, k=3)
    except Exception as e:
        print("[ERROR] similarity_search failed:", e)
        return

    if not docs:
        print("[WARN] No docs returned from similarity_search.")
        return

    print(f"[RESULT] Retrieved {len(docs)} docs. Top results:")
    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").replace("\n", " ")[:200]
        meta = doc.get("metadata", {}) or {}
        score = doc.get("score", meta.get("score", "N/A"))
        print(f"\n--- Doc #{i} ---")
        print("Score:", score)
        print("Meta:", meta)
        print("Text snippet:", text, "...")


def test_answer_question():
    print("\n=== 3) ANSWER_QUESTION END-TO-END TEST ===")
    query = "What is the late submission policy?"
    print(f"[INFO] Question: {query!r}")

    res = answer_question(
        query,
        answer_style="Strict policy quote",
        rewrite_query=True,
        k=5,
    )

    print("\n[ANSWER]")
    print(res.get("answer", ""))

    print("\n[METADATA]")
    meta = {
        "confidence_label": res.get("confidence_label"),
        "confidence_score": res.get("confidence_score"),
        "hallucination_flag": res.get("hallucination_flag"),
        "hallucination_risk": res.get("hallucination_risk"),
        "used_query": res.get("used_query"),
        "latency_ms": res.get("latency_ms"),
    }
    pprint(meta)

    print("\n[CITATIONS]")
    citations = res.get("citations", []) or []
    if not citations:
        print("No citations returned.")
    else:
        for c in citations:
            print(
                f"- source={c.get('source')} "
                f"chunk_id={c.get('chunk_id')} "
                f"rank={c.get('rank')} "
                f"sim={c.get('similarity')} "
            )


if __name__ == "__main__":
    print(">>> Running PolicyNavigator smoke tests...")
    test_ingestion()
    test_vector_search()
    test_answer_question()
    print("\n>>> Done.")
