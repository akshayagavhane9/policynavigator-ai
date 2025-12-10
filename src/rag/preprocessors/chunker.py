from typing import List, Dict, Any


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 200,
    doc_id: str = "",
    base_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks using a fixed step.

    The step size is (max_chars - overlap).
    For example: max_chars=800, overlap=200 → step=600.

    This implementation avoids infinite loops even when the text is shorter
    than max_chars.
    """
    if base_metadata is None:
        base_metadata = {}

    chunks: List[Dict[str, Any]] = []
    if not text:
        return chunks

    n = len(text)

    # Ensure sensible parameters
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        # No progress would be made → infinite loop risk
        overlap = max_chars // 4  # fallback to something safe

    step = max_chars - overlap

    chunk_index = 0
    for start in range(0, n, step):
        end = min(start + max_chars, n)
        chunk_str = text[start:end].strip()
        if not chunk_str:
            continue

        chunk_metadata = {
            **base_metadata,
            "chunk_index": chunk_index,
        }
        chunks.append({
            "id": f"{doc_id}_chunk_{chunk_index}",
            "text": chunk_str,
            "metadata": chunk_metadata,
        })
        chunk_index += 1

    return chunks
