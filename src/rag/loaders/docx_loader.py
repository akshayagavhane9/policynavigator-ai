import os
from typing import List, Dict, Any

from docx import Document


def load_docx(path: str) -> List[Dict[str, Any]]:
    """
    Load a DOCX file and return a list with a single document dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DOCX not found: {path}")

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    full_text = "\n".join(paragraphs).strip()

    return [{
        "id": os.path.basename(path),
        "text": full_text,
        "metadata": {
            "source": path,
            "type": "docx",
        },
    }]
