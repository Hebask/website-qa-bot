id="1y6e5l"
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PAGES_FILE = DATA_DIR / "pages.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"

CHUNKS_EMB_FILE = DATA_DIR / "chunks_embedded.json"

def save_pages(pages: list[dict]) -> None:
    payload = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "count": len(pages),
        "pages": pages,
    }
    PAGES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pages() -> list[dict]:
    if not PAGES_FILE.exists():
        return []
    payload = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    return payload.get("pages", [])


def save_chunks(chunks: list[dict]) -> None:
    payload = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "count": len(chunks),
        "chunks": chunks,
    }
    CHUNKS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        return []
    payload = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return payload.get("chunks", [])

def save_embedded_chunks(chunks: list[dict]) -> None:
    payload = {
        "count": len(chunks),
        "chunks": chunks,
    }
    CHUNKS_EMB_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_embedded_chunks() -> list[dict]:
    if not CHUNKS_EMB_FILE.exists():
        return []
    payload = json.loads(CHUNKS_EMB_FILE.read_text(encoding="utf-8"))
    return payload.get("chunks", [])