import re


def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - normalize whitespace
    - remove extra blank lines
    - strip leading/trailing spaces
    """
    if not text:
        return ""

    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines and remove consecutive blank lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
