import os
from typing import List, Dict, Any


def load_text(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """
    Load a plain text/markdown file and return a list with a single document dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")

    with open(path, "r", encoding=encoding) as f:
        full_text = f.read().strip()

    return [{
        "id": os.path.basename(path),
        "text": full_text,
        "metadata": {
            "source": path,
            "type": "text",
        },
    }]
