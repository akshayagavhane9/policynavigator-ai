import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

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


# -----------------------------
# Loading
# -----------------------------

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


# -----------------------------
# Prompting
# -----------------------------

def _build_qg_prompt(cleaned_text: str, source_name: str, num_questions: int) -> str:
    """
    Generates EXTRACTIVE Q&A pairs.
    Key improvement: gold answers are copied (verbatim) from policy text.
    """
    return f"""
You are generating an evaluation dataset for a Retrieval-Augmented Generation (RAG) university policy assistant.

TASK:
From the policy text below, create {num_questions} realistic student questions and EXTRACTIVE answers.

CRITICAL RULES:
- The "answer" MUST be a short extractive span copied VERBATIM from the policy text.
- Maximum 25 words in "answer".
- Also provide "evidence_quote" which is the exact same as "answer".
- Provide "evidence_span" describing where you found it (e.g., "near paragraph about plagiarism" or "section heading if visible").
- If the text does not contain a clear answer, DO NOT create that QA item.

Return STRICTLY valid JSON ONLY (no backticks, no explanations), as a single JSON array of objects with this schema:

[
  {{
    "question": "string",
    "answer": "verbatim extract <= 25 words",
    "evidence_quote": "same as answer",
    "evidence_span": "where it was found (short)",
    "section": "short topic label",
    "difficulty": "easy" | "medium" | "hard",
    "q_type": "definition" | "procedure" | "sanction" | "exception" | "other"
  }},
  ...
]

If you cannot create questions, return [].

Policy source: {source_name}

Policy text:
\"\"\"{cleaned_text}\"\"\"
""".strip()


def _extract_json_array(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("["):
        return raw

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]

    raise ValueError("Could not find JSON array in model output.")


def _parse_qas_from_raw(raw: str) -> List[Dict[str, Any]]:
    json_str = _extract_json_array(raw)
    parsed = json.loads(json_str)
    if not isinstance(parsed, list):
        raise ValueError("Top-level JSON is not a list.")
    return parsed


# -----------------------------
# Windowing / sampling
# -----------------------------

def _make_windows(text: str, window_chars: int = 4500, stride_chars: int = 3500) -> List[str]:
    """
    Instead of always using the same beginning of the doc, generate overlapping windows.
    This increases coverage and improves dataset diversity.
    """
    text = (text or "").strip()
    if not text:
        return []

    windows: List[str] = []
    n = len(text)
    start = 0
    while start < n and len(windows) < 6:  # cap windows to control token/cost
        end = min(start + window_chars, n)
        win = text[start:end].strip()
        if win:
            windows.append(win)
        start += stride_chars
    return windows


# -----------------------------
# Core generation
# -----------------------------

def generate_qa_for_document(
    path: str,
    num_questions: int = 8,
    client: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    """
    Generates QAs by sampling multiple windows across the document/chunks.
    Produces extractive answers for meaningful evaluation.
    """
    if client is None:
        client = LLMClient()

    docs = _load_doc(path)
    all_qas: List[Dict[str, Any]] = []

    # If loaders already split into pages/chunks, we still window each cleaned block
    for doc in docs:
        cleaned_full = clean_text(doc.get("text", ""))
        if not cleaned_full:
            continue

        windows = _make_windows(cleaned_full, window_chars=4500, stride_chars=3500)
        if not windows:
            continue

        # Spread questions across windows rather than all in one place
        # Example: if num_questions=12 and windows=3 -> 4 each
        per_win = max(2, num_questions // max(len(windows), 1))

        for w_i, win_text in enumerate(windows):
            prompt = _build_qg_prompt(
                win_text,
                source_name=f"{os.path.basename(path)} (window {w_i+1}/{len(windows)})",
                num_questions=per_win,
            )

            raw = client.run(prompt)

            try:
                parsed = _parse_qas_from_raw(raw)
            except Exception as e:
                print(f"[WARN] Failed to parse JSON for {path}: {e}")
                print("[DEBUG] First 400 chars of model output:")
                print(raw[:400])

                repair_prompt = f"""
You are given an invalid attempt at JSON for question–answer pairs.
Fix it and return STRICTLY valid JSON array only.

Schema reminder:
[
  {{
    "question": "...",
    "answer": "...",
    "evidence_quote": "...",
    "evidence_span": "...",
    "section": "...",
    "difficulty": "easy|medium|hard",
    "q_type": "definition|procedure|sanction|exception|other"
  }}
]

Here is the invalid text:
\"\"\"{raw.strip()[:4000]}\"\"\"
"""
                repair_raw = client.run(repair_prompt)
                try:
                    parsed = _parse_qas_from_raw(repair_raw)
                except Exception as e2:
                    print(f"[WARN] Repair also failed for {path}: {e2}")
                    continue

            for idx, item in enumerate(parsed):
                q = (item.get("question", "") or "").strip()
                a = (item.get("answer", "") or "").strip()

                # Hard constraints: extractive + short
                if not q or not a:
                    continue
                if len(a.split()) > 25:
                    continue

                qa_id = f"{os.path.basename(path)}_{w_i}_{idx}_{uuid.uuid4().hex[:8]}"
                qa = {
                    "id": qa_id,
                    "question": q,
                    "answer": a,
                    "evidence_quote": (item.get("evidence_quote", "") or a).strip(),
                    "evidence_span": (item.get("evidence_span", "") or "unknown").strip(),
                    "source_doc": os.path.basename(path),
                    "section": (item.get("section", "") or "unknown").strip(),
                    "difficulty": (item.get("difficulty", "medium") or "medium").strip(),
                    "q_type": (item.get("q_type", "other") or "other").strip(),
                }
                all_qas.append(qa)

    return all_qas


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_synthetic_dataset(
    kb_dir: str = os.path.join("data", "kb_raw"),
    num_questions_per_doc: int = 24,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> str:
    """
    Iterate over all files in kb_dir and generate synthetic Q&A.
    Recommended num_questions_per_doc: 20-40 for stable metrics.
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
