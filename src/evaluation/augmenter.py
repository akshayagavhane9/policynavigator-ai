import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Add project root to path when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

try:
    from ..llm.client import LLMClient
except ImportError:
    from src.llm.client import LLMClient

load_dotenv()

DATA_DIR = os.path.join("data", "synthetic_eval")
INPUT_PATH = os.path.join(DATA_DIR, "synthetic_qa.jsonl")
OUTPUT_PATH = os.path.join(DATA_DIR, "synthetic_qa_augmented.jsonl")


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_good_variant(original: str, candidate: str) -> bool:
    o = _normalize(original)
    c = _normalize(candidate)

    if not c or c == o:
        return False

    # Avoid super short low-signal questions
    if len(c.split()) < 6:
        return False

    # Must be meaning-preserving-ish: simple overlap guardrail
    o_set = set(o.split())
    c_set = set(c.split())
    overlap = len(o_set & c_set) / max(len(o_set), 1)
    if overlap < 0.25:  # too different → likely meaning drift
        return False

    return True


def build_paraphrase_prompt(question: str) -> str:
    return f"""
Paraphrase the question below so that:
- Meaning stays EXACTLY the same.
- The correct policy answer does NOT change.
- Wording is clearly different.
- Keep it specific (avoid making it too short).
Return ONLY the paraphrased question text.

Original question:
\"\"\"{question}\"\"\"
""".strip()


def build_search_query_prompt(question: str) -> str:
    return f"""
Rewrite the question as a SEARCH-OPTIMIZED query for retrieving policy clauses.
Rules:
- Keep meaning the same.
- Use key terms / keywords.
- Keep it short (6–14 words).
Return ONLY the rewritten search query.

Original question:
\"\"\"{question}\"\"\"
""".strip()


def augment_dataset_with_paraphrases(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    paraphrases_per_question: int = 1,
    include_search_variants: bool = True,
) -> str:
    base_records = read_jsonl(input_path)
    client = LLMClient()

    augmented: List[Dict[str, Any]] = []

    for rec in base_records:
        q0 = rec.get("question", "").strip()
        if not q0:
            continue

        # Keep original
        augmented.append({**rec, "augmentation": "original"})

        # Natural paraphrases
        for i in range(paraphrases_per_question):
            prompt = build_paraphrase_prompt(q0)
            new_q = client.run(prompt).strip()

            if not _is_good_variant(q0, new_q):
                continue

            new_rec = {
                **rec,
                "id": rec["id"] + f"_paraN{i+1}",
                "question": new_q,
                "augmentation": "paraphrase_natural",
            }
            augmented.append(new_rec)

        # Search-optimized variants (optional but strong for RAG)
        if include_search_variants:
            prompt = build_search_query_prompt(q0)
            search_q = client.run(prompt).strip()

            if _is_good_variant(q0, search_q):
                new_rec = {
                    **rec,
                    "id": rec["id"] + "_paraS1",
                    "question": search_q,
                    "augmentation": "paraphrase_search",
                }
                augmented.append(new_rec)

    write_jsonl(augmented, output_path)
    print(f"[AUG] Input records: {len(base_records)}  →  Output records: {len(augmented)}")
    print(f"[AUG] Wrote augmented dataset to {output_path}")
    return output_path


if __name__ == "__main__":
    augment_dataset_with_paraphrases()
