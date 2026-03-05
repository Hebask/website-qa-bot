import numpy as np
from typing import List, Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_by_embedding(query_emb: list[float], chunks: List[Dict], k: int = 6) -> List[Dict]:
    q = np.array(query_emb, dtype=np.float32)

    scored = []
    for ch in chunks:
        emb = ch.get("embedding")
        if not emb:
            continue
        v = np.array(emb, dtype=np.float32)
        score = cosine_similarity(q, v)
        scored.append({**ch, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]