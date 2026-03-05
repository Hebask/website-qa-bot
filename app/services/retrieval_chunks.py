id="cvzj1n"
import re
from collections import Counter
from math import log
from typing import List, Dict


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _build_idf(items: List[Dict], field: str = "text") -> Dict[str, float]:
    N = max(1, len(items))
    df = Counter()

    for it in items:
        tokens = set(_tokenize(it.get(field, "")))
        for t in tokens:
            df[t] += 1

    idf = {}
    for t, d in df.items():
        idf[t] = log((N + 1) / (d + 1)) + 1.0
    return idf


def score_chunks(question: str, chunks: List[Dict]) -> List[Dict]:
    q_tokens = _tokenize(question)
    if not q_tokens or not chunks:
        return []

    idf = _build_idf(chunks, field="text")
    q_counts = Counter(q_tokens)

    scored = []
    for ch in chunks:
        c_counts = Counter(_tokenize(ch.get("text", "")))
        score = 0.0
        for t, qt in q_counts.items():
            tf = c_counts.get(t, 0)
            if tf:
                score += (1 + log(tf)) * (idf.get(t, 1.0)) * qt
        scored.append({**ch, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_top_k_chunks(question: str, chunks: List[Dict], k: int = 6) -> List[Dict]:
    scored = score_chunks(question, chunks)
    return [c for c in scored[:k] if c.get("score", 0) > 0]