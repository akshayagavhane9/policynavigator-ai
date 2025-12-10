import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Add project root to path when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

try:
    from ..main import answer_question
    from .metrics import exact_match, token_f1, compute_aggregate_metrics, save_metrics_summary
except ImportError:
    # Fallback for when running directly
    from src.main import answer_question
    from src.evaluation.metrics import exact_match, token_f1, compute_aggregate_metrics, save_metrics_summary

load_dotenv()

DATA_DIR = os.path.join("data", "synthetic_eval")
DATA_PATH = os.path.join(DATA_DIR, "synthetic_qa_augmented.jsonl")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.jsonl")
SUMMARY_PATH = os.path.join(DATA_DIR, "metrics_summary.json")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_evaluation(
    data_path: str = DATA_PATH,
    results_path: str = RESULTS_PATH,
    summary_path: str = SUMMARY_PATH,
    limit: int | None = None,
) -> None:
    """
    Evaluate the live system on the synthetic dataset.
    """
    dataset = read_jsonl(data_path)
    if limit is not None:
        dataset = dataset[:limit]

    print(f"[EVAL] Loaded {len(dataset)} examples from {data_path}")

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, start=1):
        q = item["question"]
        gold_answer = item["answer"]

        print(f"[EVAL] ({idx}/{len(dataset)}) Q: {q[:80]}...")

        start = time.perf_counter()
        model_out = answer_question(q, use_rewrite=True, top_k=5, collection_name="policies")
        end = time.perf_counter()

        pred_answer = model_out.get("answer", "")
        confidence = model_out.get("confidence", "unknown")
        citations = model_out.get("citations", [])

        em = exact_match(gold_answer, pred_answer)
        f1 = token_f1(gold_answer, pred_answer)
        latency_ms = (end - start) * 1000.0

        result_record = {
            **item,
            "pred_answer": pred_answer,
            "pred_confidence": confidence,
            "pred_citations": citations,
            "is_exact_match": em,
            "f1": f1,
            "latency_ms": latency_ms,
        }
        results.append(result_record)

    write_jsonl(results, results_path)
    metrics = compute_aggregate_metrics(results)
    save_metrics_summary(metrics, summary_path)

    print(f"[EVAL] Wrote detailed results to {results_path}")
    print(f"[EVAL] Metrics summary:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    # For first runs, you may want to limit to e.g. 10 examples to save tokens.
    run_evaluation(limit=10)
