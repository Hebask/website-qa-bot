import re
from collections import Counter
from math import log
from typing import List, Dict


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _build_idf(pages: List[Dict]) -> Dict[str, float]:
    """
    IDF per token across pages: log((N+1)/(df+1)) + 1
    """
    N = max(1, len(pages))
    df = Counter()

    for p in pages:
        tokens = set(_tokenize((p.get("title") or "") + " " + (p.get("text") or "")))
        for t in tokens:
            df[t] += 1

    idf = {}
    for t, d in df.items():
        idf[t] = log((N + 1) / (d + 1)) + 1.0
    return idf


def score_pages(question: str, pages: List[Dict]) -> List[Dict]:
    """
    Returns pages with an added 'score' field, sorted desc.
    Uses TF-IDF-ish overlap between question tokens and page tokens.
    """
    q_tokens = _tokenize(question)
    if not q_tokens or not pages:
        return []

    idf = _build_idf(pages)
    q_counts = Counter(q_tokens)

    scored = []
    for p in pages:
        text = (p.get("title") or "") + " " + (p.get("text") or "")
        p_counts = Counter(_tokenize(text))

        score = 0.0
        for t, qt in q_counts.items():
            tf = p_counts.get(t, 0)
            if tf:
                score += (1 + log(tf)) * (idf.get(t, 1.0)) * qt

        scored.append({**p, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_top_k(question: str, pages: List[Dict], k: int = 4) -> List[Dict]:
    scored = score_pages(question, pages)
    return [p for p in scored[:k] if p.get("score", 0) > 0]