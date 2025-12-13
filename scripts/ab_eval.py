import os
import sys
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# ✅ Add project root so `import src...` works
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from src.main import ingest_and_index_documents, answer_question, KB_RAW_DIR  # type: ignore

QUESTIONS: List[str] = [
    "How does Northeastern define cheating in the academic integrity policy?",
    "What happens after an academic integrity violation is reported?",
    "What are examples of plagiarism mentioned in the policy?",
    "What is the general appeal process mentioned in the policy?",
    "Are students allowed to collaborate on assignments, and under what conditions?",
]

OUT_CSV = "results/ab_eval_runs.csv"
OUT_JSON = "results/ab_eval_summary.json"


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def run_one(question: str, variant: str) -> Dict[str, Any]:
    if variant == "baseline":
        res = answer_question(
            question,
            answer_style="Strict policy quote",
            rewrite_query=False,
            use_mmr=False,
            k=5,
            top_k_raw=5,
        )
    else:
        res = answer_question(
            question,
            answer_style="Strict policy quote",
            rewrite_query=True,
            use_mmr=True,
            k=5,
            top_k_raw=20,
        )

    citations = res.get("citations") or []
    max_sim = safe_float(res.get("confidence_score", 0.0))
    hall_flag = bool(res.get("hallucination_flag", False))
    risk = res.get("hallucination_risk", "unknown")
    used_query = res.get("used_query", "")
    latency = int(res.get("latency_ms", 0))

    evidence_ok = max_sim >= 0.35 and len(citations) > 0

    return {
        "question": question,
        "variant": variant,
        "max_similarity": max_sim,
        "confidence_label": res.get("confidence_label", "low"),
        "hallucination_flag": hall_flag,
        "hallucination_risk": risk,
        "citations_count": len(citations),
        "latency_ms": latency,
        "used_query": used_query,
        "answer_preview": (res.get("answer", "") or "")[:220].replace("\n", " "),
        "evidence_ok": evidence_ok,
    }


def main() -> None:
    os.makedirs("results", exist_ok=True)

    kb_files = []
    if os.path.exists(KB_RAW_DIR):
        kb_files = [
            os.path.join(KB_RAW_DIR, f)
            for f in os.listdir(KB_RAW_DIR)
            if os.path.isfile(os.path.join(KB_RAW_DIR, f))
        ]

    if kb_files:
        print(f"[AB] Found {len(kb_files)} file(s) in {KB_RAW_DIR}. Re-indexing for reproducibility...")
        ingest_and_index_documents(kb_files)
    else:
        print(f"[AB] WARNING: No files found in {KB_RAW_DIR}. Results may be weak.")

    rows: List[Dict[str, Any]] = []
    for q in QUESTIONS:
        print(f"\n[Q] {q}")

        r0 = run_one(q, "baseline")
        print(f"  baseline: max_sim={r0['max_similarity']:.3f} citations={r0['citations_count']} hall={r0['hallucination_flag']}")

        r1 = run_one(q, "improved")
        print(f"  improved: max_sim={r1['max_similarity']:.3f} citations={r1['citations_count']} hall={r1['hallucination_flag']}")

        rows.append(r0)
        rows.append(r1)

    # Write CSV (append-safe)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    def agg(variant: str) -> Dict[str, float]:
        vs = [r for r in rows if r["variant"] == variant]
        return {
            "n": float(len(vs)),
            "avg_max_similarity": sum(r["max_similarity"] for r in vs) / len(vs),
            "avg_latency_ms": sum(r["latency_ms"] for r in vs) / len(vs),
            "hallucination_rate": sum(1 for r in vs if r["hallucination_flag"]) / len(vs),
            "avg_citations_count": sum(r["citations_count"] for r in vs) / len(vs),
            "evidence_ok_rate": sum(1 for r in vs if r["evidence_ok"]) / len(vs),
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "questions": QUESTIONS,
        "baseline": agg("baseline"),
        "improved": agg("improved"),
        "notes": {
            "baseline_config": "rewrite_query=False, use_mmr=False, top_k_raw=5, k=5",
            "improved_config": "rewrite_query=True, use_mmr=True, top_k_raw=20, k=5",
            "evidence_ok_definition": "max_similarity>=0.35 and citations_count>0",
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Wrote:")
    print(f" - {OUT_CSV}")
    print(f" - {OUT_JSON}")


if __name__ == "__main__":
    main()
