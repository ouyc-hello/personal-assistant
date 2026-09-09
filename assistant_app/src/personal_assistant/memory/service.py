from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from personal_assistant.rag.embeddings import EmbeddingsLike
from personal_assistant.storage.models import MemoryCandidate, MemoryRecord
from personal_assistant.storage.repositories import MemoryRepository
from personal_assistant.storage.state import MemoryStatus


class MemoryService:
    """Business rules around candidate extraction, trust and lifecycle transitions."""

    def __init__(
        self, session: Session, embeddings: EmbeddingsLike | None = None, *, withdraw_hours: int = 24
    ) -> None:
        self.repository = MemoryRepository(session)
        self.embeddings = embeddings
        self.withdraw_hours = withdraw_hours

    def propose(self, *, user_id: str, kind: str, content: str, source: str,
                confidence: float, evidence: Mapping[str, Any] | None = None,
                sensitive: bool = False) -> MemoryCandidate:
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.embeddings is None:
            raise ValueError("embeddings are required to propose a memory")
        vector = self.embeddings.embed_query(content)
        candidate = self.repository.create_candidate(
            user_id=user_id,
            kind=kind,
            content=content.strip(),
            source=source,
            confidence=confidence,
            evidence={**dict(evidence or {}), "sensitive": sensitive},
            embedding=vector,
            withdraw_deadline=datetime.now(UTC) + timedelta(hours=self.withdraw_hours),
        )
        conflicts = self.repository.find_conflicts(user_id=user_id, kind=kind, content=content)
        if conflicts:
            self.repository.mark_candidate(candidate.id, MemoryStatus.CONFLICT.value)
        elif self._can_auto_publish(candidate, sensitive=sensitive):
            self.repository.publish_candidate(candidate.id)
        return candidate

    def publish(self, candidate_id: str) -> MemoryRecord:
        return self.repository.publish_candidate(candidate_id)

    def resolve_conflict(self, candidate_id: str, supersede_record_id: str) -> MemoryRecord:
        return self.repository.resolve_conflict(candidate_id, supersede_record_id)

    def withdraw(self, record_id: str, *, actor: str = "user") -> MemoryRecord:
        return self.repository.withdraw(record_id, actor=actor)

    def expire_candidates(self, *, now: datetime | None = None) -> list[MemoryCandidate]:
        return self.repository.expire_candidates(now=now)

    @staticmethod
    def _can_auto_publish(candidate: MemoryCandidate, *, sensitive: bool) -> bool:
        return (
            not sensitive
            and candidate.confidence >= 0.85
            and candidate.source in {"USER", "TOOL", "MANUAL"}
        )
