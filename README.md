# Website QA Bot (Scrape → Chunk → Ask)
1) crawls a website (internal links only),
2) extracts & cleans text,
3) chunks the content,
4) answers questions and summarizes using only the scraped website data.

## Features
- Crawl a site starting from a URL (max pages limit)
- Save scraped pages to `data/pages.json`
- Chunk content and save to `data/chunks.json`
- Question answering using chunk-level retrieval (RAG-lite)
- FastAPI endpoints with Swagger UI

## Tech Stack
- Python, FastAPI
- requests + BeautifulSoup for scraping
- JSON storage 
- OpenAI for summarization / answering