import json
import os
import re
from typing import List, Dict, Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(gold: str, pred: str) -> bool:
    return normalize_text(gold) == normalize_text(pred)


def token_f1(gold: str, pred: str) -> float:
    """
    Very simple token-level F1 for qualitative comparison.
    """
    gold_tokens = normalize_text(gold).split()
    pred_tokens = normalize_text(pred).split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)

    common = gold_set & pred_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    results: list of dicts, each containing:
      - is_exact_match: bool
      - f1: float
      - latency_ms: float
      - confidence: str
    """
    n = len(results)
    if n == 0:
        return {}

    exact = sum(1 for r in results if r.get("is_exact_match"))
    avg_f1 = sum(r.get("f1", 0.0) for r in results) / n
    avg_latency = sum(r.get("latency_ms", 0.0) for r in results) / n

    conf_counts: Dict[str, int] = {}
    for r in results:
        c = str(r.get("confidence", "unknown")).lower()
        conf_counts[c] = conf_counts.get(c, 0) + 1

    # Simple hallucination heuristic:
    # high confidence but not exact match
    hallucinations = sum(
        1
        for r in results
        if not r.get("is_exact_match") and str(r.get("confidence", "")).lower() == "high"
    )

    metrics = {
        "num_examples": n,
        "accuracy_exact": exact / n,
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "confidence_distribution": conf_counts,
        "hallucination_rate_high_conf": hallucinations / n,
    }
    return metrics


def save_metrics_summary(metrics: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
