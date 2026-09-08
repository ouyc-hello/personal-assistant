from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from personal_assistant.schemas import SearchHit


class MemoryIndex(Protocol):
    """pgvector adapter: semantic index only, never the memory source of truth."""

    def search(self, query: str, *, user_id: str, limit: int = 5) -> Sequence[SearchHit]: ...


class DocumentIndex(Protocol):
    """Milvus adapter for document chunks and source metadata."""

    def search(self, query: str, *, user_id: str, limit: int = 8) -> Sequence[SearchHit]: ...


class GraphIndex(Protocol):
    """Neo4j adapter for explainable entities and multi-hop relationships."""

    def search(self, query: str, *, user_id: str, limit: int = 8) -> Sequence[SearchHit]: ...
