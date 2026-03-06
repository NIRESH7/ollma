import hashlib
import math
import os
import re
from typing import List, Tuple

from langchain_core.embeddings import Embeddings

HF_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"
DEFAULT_VECTOR_SIZE = 384
OPENAI_VECTOR_SIZE = 1536


class HashEmbeddings(Embeddings):
    """
    Deterministic offline embedding fallback.

    It is lower quality than transformer embeddings, but keeps ingest/query
    functional when model downloads are unavailable.
    """

    def __init__(self, size: int = DEFAULT_VECTOR_SIZE):
        self.size = size

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.size
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[idx] += sign
        return self._l2_normalize(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings() -> Tuple[Embeddings, int, str]:
    """
    Returns (embeddings, vector_size, backend_name).
    Order:
      1) OpenAI (if key configured)
      2) HuggingFace local cache only
      3) Deterministic offline hash fallback
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            return (
                OpenAIEmbeddings(model=OPENAI_MODEL_NAME),
                OPENAI_VECTOR_SIZE,
                "openai",
            )
        except Exception as e:
            print(f"--- [EMBED] OpenAI init failed, falling back: {e} ---")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return (
            HuggingFaceEmbeddings(
                model_name=HF_MODEL_NAME,
                model_kwargs={"local_files_only": True},
            ),
            DEFAULT_VECTOR_SIZE,
            "huggingface-local",
        )
    except Exception as e:
        print(f"--- [EMBED] HuggingFace local model unavailable, using hash fallback: {e} ---")

    return HashEmbeddings(size=DEFAULT_VECTOR_SIZE), DEFAULT_VECTOR_SIZE, "hash"
