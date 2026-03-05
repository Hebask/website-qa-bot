from collections import deque
from openai import OpenAI
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.services.scraper import fetch_page
from app.services.parser import extract_page_data, normalize_url
from app.services.storage import load_pages, save_chunks, load_chunks, save_embedded_chunks, load_embedded_chunks
from app.services.retrieval import select_top_k
from app.services.chunking import chunk_text
from app.services.retrieval_chunks import select_top_k_chunks
from app.services.embeddings import embed_texts, embed_query
from app.services.retrieval_embed import top_k_by_embedding
from app.services.retrieval_hybrid import hybrid_top_k
from app.services.hash_utils import sha256_text
import json
import re

client = OpenAI(api_key=OPENAI_API_KEY)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_object(text: str) -> dict:
    """
    Robustly extract JSON object from model output.
    Handles:
    - normal JSON: {"answer":"...", "evidence_chunk_ids":[1,2]}
    - JSON embedded as an escaped string: "{\"answer\":\"...\",\"evidence_chunk_ids\":[1,2]}"
    - extra text around JSON
    """
    s = (text or "").strip()

    # Case 1: direct JSON object somewhere in the text
    m = _JSON_OBJ_RE.search(s)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Case 2: response is an escaped JSON string inside quotes
    # Try to load it as a JSON string first, then parse again
    try:
        # Example: "\"{\\\"answer\\\":...}\"" -> "{ \"answer\": ... }"
        unquoted = json.loads(s)
        if isinstance(unquoted, str):
            unquoted = unquoted.strip()
            m2 = _JSON_OBJ_RE.search(unquoted)
            if m2:
                return json.loads(m2.group(0))
            return json.loads(unquoted)
    except Exception:
        pass

    # Case 3: try to find an escaped JSON object pattern in raw text
    # e.g. {\"answer\": ...}
    escaped = s.replace('\\"', '"')
    m3 = _JSON_OBJ_RE.search(escaped)
    if m3:
        return json.loads(m3.group(0))

    raise ValueError("Could not extract a valid JSON object from model output.")

def crawl_site(seed_url: str, max_pages: int = 10) -> list[dict]:
    seed_url = normalize_url(seed_url)

    visited = set()
    queue = deque([seed_url])

    pages: list[dict] = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        try:
            final_url, html = fetch_page(url)
            final_url = normalize_url(final_url)

            data = extract_page_data(html, final_url)

            pages.append({
                "url": data["url"],
                "title": data["title"],
                "text": data["text"],
            })

            for link in data["internal_links"]:
                if link not in visited:
                    queue.append(link)
        
            print(f"[OK] {final_url} (pages={len(pages)}/{max_pages})")

        except Exception as e:
            print(f"[SKIP] {url} -> {e}")

    return pages


def build_website_context(max_chars: int = 20000) -> str:
    pages = load_pages()

    if not pages:
        return "No website pages have been scraped yet."

    parts = []

    for page in pages:
        block = (
            f"URL: {page['url']}\n"
            f"TITLE: {page['title']}\n"
            f"CONTENT:\n{page['text']}\n"
            f"{'-' * 80}\n"
        )
        parts.append(block)

    full_context = "\n".join(parts)
    return full_context[:max_chars]


def summarize_website() -> str:
    context = build_website_context()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Summarize the scraped website content clearly and accurately. "
                    "Use only the provided content."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this website content:\n\n{context}",
            },
        ],
    )

    return response.output_text


def build_and_save_chunks() -> int:
    pages = load_pages()
    all_chunks = []
    idx = 0

    for p in pages:
        chunks = chunk_text(p.get("text", ""), max_chars=1200, overlap=200)
        for c in chunks:
            idx += 1
            all_chunks.append({
                "id": idx,
                "url": p["url"],
                "title": p["title"],
                "text": c,
                "hash": sha256_text(c)
            })

    save_chunks(all_chunks)

    # ---- embedding cache ----
    old = load_embedded_chunks()
    cache = {c.get("hash"): c.get("embedding") for c in old if c.get("hash") and c.get("embedding")}

    embedded_chunks = []
    to_embed_texts = []
    to_embed_indices = []

    for i, ch in enumerate(all_chunks):
        h = ch["hash"]
        if h in cache:
            embedded_chunks.append({**ch, "embedding": cache[h]})
        else:
            embedded_chunks.append({**ch, "embedding": None})
            to_embed_texts.append(ch["text"])
            to_embed_indices.append(i)

    # embed only missing
    batch_size = 32
    for start in range(0, len(to_embed_texts), batch_size):
        batch_texts = to_embed_texts[start:start + batch_size]
        batch_embs = embed_texts(batch_texts)

        for j, emb in enumerate(batch_embs):
            idx_in_all = to_embed_indices[start + j]
            embedded_chunks[idx_in_all]["embedding"] = emb

    # filter just in case
    embedded_chunks = [c for c in embedded_chunks if c.get("embedding") is not None]

    save_embedded_chunks(embedded_chunks)
    return len(all_chunks)

def answer_question(question: str, top_k: int = 6) -> str:
    chunks = load_embedded_chunks()
    if not chunks:
        return "No embedded chunks found. Please run /crawl first (it will generate embeddings)."

    q_emb = embed_query(question)
    selected = hybrid_top_k(question, q_emb, chunks, k=top_k, alpha=0.7)

    if not selected:
        return "I could not find that information in the scraped website content.\n\nSources: None"

    context_parts = []
    for ch in selected:
        context_parts.append(
            f"CHUNK_ID: {ch['id']}\n"
            f"URL: {ch['url']}\n"
            f"TITLE: {ch['title']}\n"
            f"TEXT:\n{ch['text']}\n"
            f"{'-'*60}\n"
        )
    context = "\n".join(context_parts)[:20000]

    system_prompt = (
        "Return ONLY raw JSON object. Do not wrap it in quotes. "
        "Do not escape characters. No extra text."
        "You answer ONLY using the provided chunks. "
        "Return ONLY valid JSON (no extra text, no markdown). "
        "The JSON MUST match this schema exactly: "
        "{\"answer\": string, \"evidence_chunk_ids\": number[]}. "
        "IMPORTANT: evidence_chunk_ids MUST NOT be empty unless the answer is "
        "\"I could not find that information in the scraped website content.\" "
        "Only include chunk IDs that directly support the answer."
    )

    raw_resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Chunks:\n\n{context}\n\nQuestion: {question}"},
        ],
    )

    raw_text = raw_resp.output_text.strip()

    try:
        data = extract_json_object(raw_text)
        answer = (data.get("answer") or "").strip()
        evidence_ids = set(data.get("evidence_chunk_ids") or [])
    except Exception:
        # If model breaks the JSON rule, return raw output but no sources
        return f"{raw_text}\n\nSources: No evidence chunk IDs were returned by the model."

    # Evidence-only sources (no fallback to all retrieved)
    id_to_url = {ch["id"]: ch["url"] for ch in selected}
    used_urls = [id_to_url[i] for i in evidence_ids if i in id_to_url]
    sources = ", ".join(sorted(set(used_urls)))

    if not sources:
        sources = "No evidence chunk IDs were returned by the model."

        return {"answer": answer,"sources": [],"evidence_chunk_ids": [],}

    return {
    "answer": answer,
    "sources": sorted(set(used_urls)),
    "evidence_chunk_ids": sorted(list(evidence_ids)),
}