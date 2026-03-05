import numpy as np
from typing import List, Dict
from app.services.retrieval_embed import top_k_by_embedding
from app.services.retrieval_chunks import score_chunks


def _normalize_scores(items: List[Dict], key: str) -> List[Dict]:
    vals = [it.get(key, 0.0) for it in items]
    if not vals:
        return items
    mn, mx = min(vals), max(vals)
    if mx == mn:
        for it in items:
            it[key] = 0.0
        return items
    for it in items:
        it[key] = (it.get(key, 0.0) - mn) / (mx - mn)
    return items


def hybrid_top_k(question: str, query_emb: list[float], chunks: List[Dict], k: int = 6,
                 alpha: float = 0.7) -> List[Dict]:
    """
    alpha: weight for embeddings (0..1). (1-alpha) for keyword score.
    """
    # embedding scores
    emb_scored = top_k_by_embedding(query_emb, chunks, k=max(30, k * 5))

    # keyword scores (TF-IDF-ish)
    kw_scored = score_chunks(question, chunks)

    # index by chunk id
    by_id = {}

    for it in emb_scored:
        by_id[it["id"]] = {**it, "emb_score": it.get("score", 0.0), "kw_score": 0.0}

    for it in kw_scored[:max(50, k * 8)]:
        cid = it["id"]
        if cid in by_id:
            by_id[cid]["kw_score"] = it.get("score", 0.0)
        else:
            by_id[cid] = {**it, "embedding": it.get("embedding"), "emb_score": 0.0, "kw_score": it.get("score", 0.0)}

    items = list(by_id.values())
    items = _normalize_scores(items, "emb_score")
    items = _normalize_scores(items, "kw_score")

    for it in items:
        it["score"] = alpha * it["emb_score"] + (1 - alpha) * it["kw_score"]

    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:k]