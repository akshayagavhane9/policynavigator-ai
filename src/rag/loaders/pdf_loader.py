import os
from typing import List, Dict, Any

from pypdf import PdfReader
from pypdf.errors import PdfStreamError


def load_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF file and return a list with a single document dict.

    Each document dict has:
    - 'id': unique id for the document
    - 'text': full concatenated text
    - 'metadata': source info

    This version is defensive: if the PDF is partially corrupted, it will
    try to read as much as possible instead of crashing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        # strict=False makes PdfReader more tolerant of malformed PDFs
        reader = PdfReader(path, strict=False)
    except PdfStreamError as e:
        # Log & return empty content instead of crashing the pipeline
        print(f"[WARN] Failed to fully parse PDF '{path}': {e}")
        return [{
            "id": os.path.basename(path),
            "text": "",
            "metadata": {
                "source": path,
                "type": "pdf",
                "warning": "PdfStreamError while parsing; no text extracted",
            },
        }]

    pages_text = []

    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages_text.append(text)
        except Exception as e:
            print(f"[WARN] Failed to extract text from page {page_idx} in '{path}': {e}")
            continue

    full_text = "\n".join(pages_text).strip()

    return [{
        "id": os.path.basename(path),
        "text": full_text,
        "metadata": {
            "source": path,
            "type": "pdf",
        },
    }]
