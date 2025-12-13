import os
from typing import List, Dict, Any

from pypdf import PdfReader
from pypdf.errors import PdfStreamError


def load_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF file and return a list of page-level document dicts.

    Each dict has:
      - 'id': unique id for the page
      - 'text': extracted text for that page
      - 'metadata': includes source + page number

    Defensive behavior:
      - strict=False for tolerance
      - if parsing fails, returns a single empty doc with warning metadata
      - if a page fails extraction, skips that page and continues
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(path, strict=False)
    except PdfStreamError as e:
        print(f"[WARN] Failed to parse PDF '{path}': {e}")
        return [{
            "id": f"{os.path.basename(path)}_page_0",
            "text": "",
            "metadata": {
                "source": os.path.basename(path),
                "source_path": path,
                "type": "pdf",
                "page": None,
                "warning": "PdfStreamError while parsing; no text extracted",
            },
        }]
    except Exception as e:
        print(f"[WARN] Unexpected error opening PDF '{path}': {e}")
        return [{
            "id": f"{os.path.basename(path)}_page_0",
            "text": "",
            "metadata": {
                "source": os.path.basename(path),
                "source_path": path,
                "type": "pdf",
                "page": None,
                "warning": f"Unexpected error; no text extracted: {e}",
            },
        }]

    docs: List[Dict[str, Any]] = []
    base = os.path.basename(path)

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-based for UI friendliness
        try:
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            docs.append({
                "id": f"{base}_page_{page_num}",
                "text": text,
                "metadata": {
                    "source": base,        # IMPORTANT: filename only, matches kb_raw usage
                    "source_path": path,   # full path for backend debugging
                    "type": "pdf",
                    "page": page_num,
                },
            })

        except Exception as e:
            print(f"[WARN] Failed to extract text from page {page_num} in '{path}': {e}")
            continue

    # If every page was empty/unreadable, return a single empty doc with warning.
    if not docs:
        return [{
            "id": f"{base}_page_0",
            "text": "",
            "metadata": {
                "source": base,
                "source_path": path,
                "type": "pdf",
                "page": None,
                "warning": "No extractable text found in any page",
            },
        }]

    return docs
