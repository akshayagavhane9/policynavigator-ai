import re
from typing import Union


def clean_text(text: Union[str, list]) -> str:
    """
    Normalize whitespace and basic artifacts.

    Accepts either a string or a list of strings and always returns a single
    cleaned string.
    """
    # If someone passed a list of strings, join them.
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)

    # As a last resort, coerce to string
    if not isinstance(text, str):
        text = str(text)

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
