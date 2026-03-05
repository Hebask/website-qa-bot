from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag


SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".css", ".js", ".json", ".xml"
)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    parts = [line for line in lines if line]
    return "\n".join(parts)


def normalize_url(url: str) -> str:
    # remove #fragment
    url, _frag = urldefrag(url)
    # normalize trailing slash (keep root slash)
    if url.endswith("/") and len(url) > len("https://a.b/"):
        url = url.rstrip("/")
    return url


def is_probably_a_file(url: str) -> bool:
    u = url.lower()
    return any(u.endswith(ext) for ext in SKIP_EXTENSIONS)


def extract_page_data(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

    text = soup.get_text(separator="\n")
    text = clean_text(text)

    # remove common noisy lines
    noise = [
        "This page displays a fallback because interactive scripts did not run.",
        "Possible causes include disabled JavaScript",
    ]
    for n in noise:
        text = text.replace(n, "")
    text = clean_text(text)

    internal_links = []
    base_domain = urlparse(base_url).netloc

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        full_url = urljoin(base_url, href)
        full_url = normalize_url(full_url)

        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            continue

        if parsed.netloc != base_domain:
            continue

        if is_probably_a_file(full_url):
            continue

        if full_url not in internal_links:
            internal_links.append(full_url)

    return {
        "url": base_url,
        "title": title,
        "text": text,
        "internal_links": internal_links,
    }