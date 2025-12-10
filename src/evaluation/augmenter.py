import json
import os
import sys
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
    # Fallback for when running directly
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
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_paraphrase_prompt(question: str) -> str:
    return f"""
You are helping to create robust evaluation data for a Q&A system.

Paraphrase the question below so that:
- The meaning stays EXACTLY the same.
- The correct answer does not change.
- The style and wording are clearly different.

Return ONLY the new paraphrased question text. No quotes, no explanations.

Original question:
\"\"\"{question}\"\"\"
""".strip()


def augment_dataset_with_paraphrases(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    paraphrases_per_question: int = 1,
) -> str:
    base_records = read_jsonl(input_path)
    client = LLMClient()

    augmented: List[Dict[str, Any]] = []

    for rec in base_records:
        augmented.append(rec)  # keep original

        for i in range(paraphrases_per_question):
            prompt = build_paraphrase_prompt(rec["question"])
            new_q = client.run(prompt).strip()

            if not new_q or new_q == rec["question"]:
                continue

            new_rec = {
                **rec,
                "id": rec["id"] + f"_para{i+1}",
                "question": new_q,
                "augmentation": "paraphrase",
            }
            augmented.append(new_rec)

    write_jsonl(augmented, output_path)
    print(f"[AUG] Input records: {len(base_records)}  →  Output records: {len(augmented)}")
    print(f"[AUG] Wrote augmented dataset to {output_path}")
    return output_path


if __name__ == "__main__":
    augment_dataset_with_paraphrases()
