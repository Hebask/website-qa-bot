from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.storage import save_pages, load_pages
from app.services.qa import crawl_site, summarize_website, answer_question, build_and_save_chunks

router = APIRouter()


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 10


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@router.post("/crawl")
def crawl(req: CrawlRequest):
    pages = crawl_site(req.url, max_pages=req.max_pages)
    save_pages(pages)
    chunk_count = build_and_save_chunks()

    return {
        "message": "Crawl completed",
        "saved_pages": len(pages),
        "saved_chunks": chunk_count,
        "urls": [p["url"] for p in pages],
    }


@router.get("/pages")
def pages():
    pages = load_pages()
    return {"count": len(pages), "pages": [{"url": p["url"], "title": p["title"]} for p in pages]}


@router.get("/summary")
def summary():
    pages = load_pages()
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found. Run /crawl first.")

    return {"summary": summarize_website()}


@router.post("/ask")
def ask(req: AskRequest):
    pages = load_pages()
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found. Run /crawl first.")

    return {"question": req.question, "answer": answer_question(req.question, top_k=req.top_k)}