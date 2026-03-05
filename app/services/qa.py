from collections import deque
from openai import OpenAI
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL
from app.services.scraper import fetch_page
from app.services.parser import extract_page_data, normalize_url
from app.services.storage import load_pages, save_chunks, load_chunks
from app.services.retrieval import select_top_k
from app.services.chunking import chunk_text
from app.services.retrieval_chunks import select_top_k_chunks

client = OpenAI(api_key=OPENAI_API_KEY)


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
                "text": c
            })

    save_chunks(all_chunks)
    return len(all_chunks)

def answer_question(question: str, top_k: int = 6) -> str:
    chunks = load_chunks()
    if not chunks:
        return "No chunks found. Please run /crawl first (it will generate chunks)."

    selected = select_top_k_chunks(question, chunks, k=top_k)
    if not selected:
        return "I could not find that information in the scraped website content."

    context_parts = []
    used_urls = []
    for ch in selected:
        context_parts.append(
            f"URL: {ch['url']}\nTITLE: {ch['title']}\nCHUNK:\n{ch['text']}\n{'-'*60}\n"
        )
        used_urls.append(ch["url"])

    context = "\n".join(context_parts)[:20000]
    sources = ", ".join(sorted(set(used_urls)))

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the provided website chunks. "
                    "Do NOT use outside knowledge. "
                    "If the answer is not explicitly supported, say: "
                    "'I could not find that information in the scraped website content.'"
                ),
            },
            {"role": "user", "content": f"Website chunks:\n\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response.output_text + f"\n\nSources: {sources}"
