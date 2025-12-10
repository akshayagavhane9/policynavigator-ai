import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Add project root to path when running directly
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.rag.loaders.pdf_loader import load_pdf
from src.rag.loaders.docx_loader import load_docx
from src.rag.loaders.text_loader import load_text
from src.rag.preprocessors.cleaner import clean_text
from src.llm.client import LLMClient

load_dotenv()

DATA_DIR = os.path.join("data", "synthetic_eval")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "synthetic_qa.jsonl")


def _load_doc(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in [".txt", ".md"]:
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type for synthetic generation: {ext}")


def _build_qg_prompt(cleaned_text: str, source_name: str, num_questions: int) -> str:
    """
    Prompt that asks the LLM to create Q&A pairs in JSON.
    """
    return f"""
You are helping build an evaluation dataset for a university policy assistant.

Read the policy text below and create {num_questions} diverse question–answer pairs
that a student might realistically ask. Vary the style and difficulty of questions.

Policy source: {source_name}

Return STRICTLY valid JSON ONLY, with no backticks, no explanations, and no text
before or after it. The JSON must be a single array of objects with this exact schema:

[
  {{
    "question": "...",
    "answer": "...",
    "section": "short section title or topic",
    "difficulty": "easy" | "medium" | "hard",
    "q_type": "definition" | "deadline" | "procedure" | "exception" | "other"
  }},
  ...
]

If you cannot create the questions, return [].

Policy text:
\"\"\"{cleaned_text[:4000]}\"\"\"
""".strip()


def _extract_json_array(raw: str) -> str:
    """
    Try to extract the JSON array part from the model output.
    Handles cases like: "Here is the JSON:\\n[ ... ]"
    """
    raw = raw.strip()
    # If it already starts with '[' just return
    if raw.startswith("["):
        return raw

    start = raw.find("[")
    end = raw.rfind("]")

    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]

    # No brackets found
    raise ValueError("Could not find JSON array in model output.")


def _parse_qas_from_raw(raw: str) -> List[Dict[str, Any]]:
    """
    Robust parsing: try to extract array, then json.loads.
    """
    json_str = _extract_json_array(raw)
    parsed = json.loads(json_str)
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON is not a list.")
    return parsed


def generate_qa_for_document(
    path: str,
    num_questions: int = 8,
    client: LLMClient | None = None,
) -> List[Dict[str, Any]]:
    if client is None:
        client = LLMClient()

    docs = _load_doc(path)
    all_qas: List[Dict[str, Any]] = []

    for doc in docs:
        cleaned = clean_text(doc["text"])
        if not cleaned:
            continue

        prompt = _build_qg_prompt(cleaned, os.path.basename(path), num_questions)
        raw = client.run(prompt)

        try:
            parsed = _parse_qas_from_raw(raw)
        except Exception as e:
            # Debug print for you while developing
            print(f"[WARN] Failed to parse JSON for {path}: {e}")
            print("[DEBUG] First 400 chars of model output:")
            print(raw[:400])
            # Try a simple repair prompt once
            repair_prompt = f"""
You are given an invalid attempt at JSON for question–answer pairs.

Fix it and return STRICTLY valid JSON array (no explanations, no backticks, no comments).

Here is the text:
\"\"\"{raw.strip()[:4000]}\"\"\"
"""
            repair_raw = client.run(repair_prompt)
            try:
                parsed = _parse_qas_from_raw(repair_raw)
            except Exception as e2:
                print(f"[WARN] Repair also failed for {path}: {e2}")
                continue  # skip this chunk / doc

        for idx, item in enumerate(parsed):
            qa_id = f"{os.path.basename(path)}_{idx}_{uuid.uuid4().hex[:8]}"
            qa = {
                "id": qa_id,
                "question": item.get("question", "").strip(),
                "answer": item.get("answer", "").strip(),
                "source_doc": os.path.basename(path),
                "section": item.get("section", "").strip() or "unknown",
                "difficulty": item.get("difficulty", "medium"),
                "q_type": item.get("q_type", "other"),
            }
            if qa["question"] and qa["answer"]:
                all_qas.append(qa)

    return all_qas


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_synthetic_dataset(
    kb_dir: str = os.path.join("data", "kb_raw"),
    num_questions_per_doc: int = 8,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> str:
    """
    Iterate over all files in kb_dir and generate synthetic Q&A.
    """
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"Knowledge base dir not found: {kb_dir}")

    client = LLMClient()
    all_records: List[Dict[str, Any]] = []

    files = [
        os.path.join(kb_dir, f)
        for f in os.listdir(kb_dir)
        if os.path.isfile(os.path.join(kb_dir, f))
    ]

    print(f"[GEN] Found {len(files)} file(s) in {kb_dir}")

    for path in files:
        print(f"[GEN] Generating Q&A for {path} ...")
        doc_records = generate_qa_for_document(
            path, num_questions=num_questions_per_doc, client=client
        )
        print(f"[GEN] Generated {len(doc_records)} Q&A for {path}")
        all_records.extend(doc_records)

    write_jsonl(all_records, output_path)
    print(f"[GEN] Wrote {len(all_records)} synthetic Q&A to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_synthetic_dataset()
