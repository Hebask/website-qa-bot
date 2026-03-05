from openai import OpenAI
from app.core.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns embeddings for a list of texts.
    """
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]