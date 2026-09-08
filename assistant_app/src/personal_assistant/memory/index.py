from __future__ import annotations

from personal_assistant.rag.embeddings import EmbeddingsLike
from personal_assistant.schemas import SearchHit
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.repositories import MemoryRepository


class PostgresMemoryIndex:
    """Text-query adapter over trusted memory records and their pgvector column."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingsLike,
        *,
        database: Database | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.database = database or Database(settings=settings)
        self._owns_database = database is None

    def search(self, query: str, *, user_id: str, limit: int = 5) -> list[SearchHit]:
        vector = self.embeddings.embed_query(query)
        with self.database.session() as session:
            return MemoryRepository(session).search_published(
                user_id=user_id,
                query_vector=vector,
                limit=limit,
            )

    def close(self) -> None:
        if self._owns_database:
            self.database.dispose()
