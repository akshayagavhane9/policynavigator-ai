import json
import os
import re
from typing import List, Dict, Any


# ---------------------------
# Normalization + core metrics
# ---------------------------

def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(gold: str, pred: str) -> bool:
    return normalize_text(gold) == normalize_text(pred)


def token_f1(gold: str, pred: str) -> float:
    """
    Token-level F1 (simple overlap). Better than exact match for long answers.
    """
    gold_tokens = normalize_text(gold).split()
    pred_tokens = normalize_text(pred).split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    # multiset overlap (counts) is better, but set overlap is okay for quick iteration
    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)
    common = gold_set & pred_set
    if not common:
        return 0.0

    precision = len(common) / max(len(pred_set), 1)
    recall = len(common) / max(len(gold_set), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------
# Hallucination proxy
# ---------------------------

ASSERTIVE_PATTERN = re.compile(
    r"\b(must|will|requires|required|prohibited|policy states|sanction|"
    r"consequence|disciplinary|violation|shall)\b",
    re.IGNORECASE,
)


def hallucination_flag(
    pred: str,
    citations: list,
    retrieval_failed: bool,
    abstained: bool,
) -> int:
    """
    Heuristic proxy for hallucination.

    We flag as hallucination if:
      - retrieval failed (no citations) AND model still produced an assertive answer, OR
      - model produced a non-empty answer with no citations and did not abstain
    """
    pred = (pred or "").strip()
    if not pred:
        return 0

    if abstained:
        return 0

    assertive = bool(ASSERTIVE_PATTERN.search(pred))

    # retrieval failed but still assertive => very risky
    if retrieval_failed and assertive:
        return 1

    # any non-empty answer without citations is risky too
    if retrieval_failed and pred:
        return 1

    return 0


# ---------------------------
# Row + Aggregate metrics
# ---------------------------

def compute_row_metrics(
    gold: str,
    pred: str,
    citations: list,
    retrieval_failed: bool,
    abstained: bool,
    confidence_label: str,
) -> Dict[str, Any]:
    em = exact_match(gold, pred)
    f1 = token_f1(gold, pred)

    hall = hallucination_flag(
        pred=pred,
        citations=citations,
        retrieval_failed=retrieval_failed,
        abstained=abstained,
    )

    return {
        "is_exact_match": em,
        "f1": f1,
        "hallucination_flag": hall,
        "confidence_label": (confidence_label or "unknown").lower(),
    }


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    exact = sum(1 for r in results if r.get("is_exact_match"))
    avg_f1 = sum(float(r.get("f1", 0.0)) for r in results) / n
    avg_latency = sum(float(r.get("latency_ms", 0.0)) for r in results) / n

    # confidence distribution
    conf_counts: Dict[str, int] = {}
    for r in results:
        c = str(r.get("confidence_label", r.get("pred_confidence_label", "unknown"))).lower()
        conf_counts[c] = conf_counts.get(c, 0) + 1

    # retrieval success
    retrieval_failed_count = sum(1 for r in results if r.get("retrieval_failed") is True)
    retrieval_success_rate = 1.0 - (retrieval_failed_count / n)

    # abstain rate
    abstained_count = sum(1 for r in results if r.get("abstained") is True)
    abstain_rate = abstained_count / n

    # hallucination rates (overall + high-confidence)
    hallucinations = sum(1 for r in results if r.get("hallucination_flag") == 1)
    hallucination_rate = hallucinations / n

    high_conf_subset = [
        r for r in results
        if str(r.get("confidence_label", r.get("pred_confidence_label", ""))).lower() == "high"
    ]
    if high_conf_subset:
        high_conf_hall = sum(1 for r in high_conf_subset if r.get("hallucination_flag") == 1)
        hallucination_rate_high_conf = high_conf_hall / len(high_conf_subset)
    else:
        hallucination_rate_high_conf = 0.0

    metrics = {
        "num_examples": n,
        "accuracy_exact": exact / n,
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "confidence_distribution": conf_counts,
        "retrieval_success_rate": retrieval_success_rate,
        "abstain_rate": abstain_rate,
        "hallucination_rate": hallucination_rate,
        "hallucination_rate_high_conf": hallucination_rate_high_conf,
    }
    return metrics


def save_metrics_summary(metrics: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
