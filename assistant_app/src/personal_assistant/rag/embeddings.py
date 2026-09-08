from __future__ import annotations

import hashlib
import math
from typing import Protocol

from personal_assistant.settings import Settings


class EmbeddingsLike(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddings:
    """Deterministic offline embeddings for tests and local wiring only."""

    def __init__(self, dimension: int = 32) -> None:
        if dimension < 2:
            raise ValueError("embedding dimension must be at least 2")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                number = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((number / 2**31) - 1.0)
                if len(values) == self.dimension:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> EmbeddingsLike:
    if settings.embedding_provider == "fake":
        return HashEmbeddings(settings.embedding_dimension)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
