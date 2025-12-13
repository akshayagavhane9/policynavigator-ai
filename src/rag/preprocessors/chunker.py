# src/rag/preprocessors/chunker.py
from __future__ import annotations

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1100,
    chunk_overlap: int = 180,
    separators: List[str] | None = None,
) -> List[str]:
    """
    High-quality policy chunker (no external deps).

    Goals:
    - Keep policy clauses/paragraphs intact where possible.
    - Create enough chunks for good retrieval (not just ~8).
    - Add overlap so key definitions don't get split away from context.

    Args:
        text: cleaned text
        chunk_size: target max chars per chunk
        chunk_overlap: chars overlapped between chunks
        separators: preferred split boundaries (in order)

    Returns:
        List of chunk strings
    """
    if not text or not str(text).strip():
        return []

    t = str(text).strip()

    if separators is None:
        # Prefer splitting on paragraph boundaries, then sentences, then spaces.
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    # --- Step 1: split into "units" using paragraph boundaries first ---
    units: List[str] = [t]
    for sep in separators[:2]:  # only paragraph-ish separators here
        new_units: List[str] = []
        for u in units:
            if len(u) <= chunk_size:
                new_units.append(u)
            else:
                parts = [p.strip() for p in u.split(sep) if p.strip()]
                # Re-add the separator lightly for readability
                if sep.strip():
                    parts = [p + sep.strip() for p in parts]
                new_units.extend(parts)
        units = new_units

    # --- Step 2: greedy pack into chunks up to chunk_size ---
    chunks: List[str] = []
    buf = ""

    def flush_buffer() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for u in units:
        if not u.strip():
            continue

        if len(buf) + len(u) + 1 <= chunk_size:
            buf = (buf + "\n" + u).strip() if buf else u.strip()
            continue

        # Buffer would overflow -> flush current
        flush_buffer()

        # If single unit is still too large, split it further (sentence/space)
        if len(u) > chunk_size:
            chunks.extend(_split_long_unit(u, chunk_size=chunk_size, separators=separators[2:]))
        else:
            buf = u.strip()

    flush_buffer()

    # --- Step 3: add overlap window ---
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = []
    prev = ""
    for c in chunks:
        if not prev:
            overlapped.append(c)
            prev = c
            continue
        prefix = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        merged = (prefix + "\n" + c).strip()
        # Avoid absurdly large overlap chunks
        if len(merged) > chunk_size + chunk_overlap:
            merged = merged[-(chunk_size + chunk_overlap) :]
        overlapped.append(merged)
        prev = c

    return overlapped


def _split_long_unit(unit: str, chunk_size: int, separators: List[str]) -> List[str]:
    u = unit.strip()
    if len(u) <= chunk_size:
        return [u]

    # Try progressively finer separators
    for sep in separators:
        parts = [p.strip() for p in u.split(sep) if p.strip()]
        if len(parts) <= 1:
            continue

        # Greedy pack parts
        out: List[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + sep + p).strip() if buf else p
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = p
                if len(buf) > chunk_size:
                    # fallback to hard split
                    out.extend(_hard_split(buf, chunk_size))
                    buf = ""
        if buf:
            out.append(buf)

        # If split improved, return it
        if out and all(len(x) <= chunk_size for x in out):
            return out

    # Fallback: hard split
    return _hard_split(u, chunk_size)


def _hard_split(text: str, chunk_size: int) -> List[str]:
    t = text.strip()
    return [t[i : i + chunk_size].strip() for i in range(0, len(t), chunk_size) if t[i : i + chunk_size].strip()]
