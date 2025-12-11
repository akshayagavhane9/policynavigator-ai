from typing import List, Union


def chunk_text(
    text: Union[str, list],
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[str]:
    """
    Simple word-based chunking.

    - Accepts a string or list of strings.
    - Returns a list of chunk strings.
    """
    # If it’s a list, join into one string
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)

    if not isinstance(text, str):
        text = str(text)

    words = text.split()
    chunks: List[str] = []

    if not words:
        return chunks

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward with overlap
        start = end - overlap
        if start < 0:
            start = 0

    return chunks
