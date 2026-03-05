import requests


def fetch_page(url: str, timeout: int = 15) -> tuple[str, str]:
    """
    Returns: (final_url, html)
    final_url matters when the site redirects.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebsiteQABot/1.0"
    }

    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/html" not in content_type:
        raise ValueError(f"Non-HTML content-type: {content_type}")

    return str(resp.url), resp.text