from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IndexedChunk:
    """A LangChain document chunk plus stable IDs for PostgreSQL/Milvus joins."""

    id: str
    document_id: str
    user_id: str
    text: str
    vector: list[float]
    chunk_index: int
    version: int
    source_uri: str
    page: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    source_uri: str
    version: int
    chunk_count: int
    content_hash: str
