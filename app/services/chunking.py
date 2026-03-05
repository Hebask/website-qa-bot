id="3v1m5d"
import re
from typing import List


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    text = (text or "").strip()
    if not text:
        return []

    # normalize whitespace a bit
    text = re.sub(r"\n{3,}", "\n\n", text)

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks