from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from personal_assistant.memory.service import MemoryService
from personal_assistant.rag.embeddings import HashEmbeddings
from personal_assistant.schemas import SearchHit
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import MemoryCandidate, MemoryRecord


@pytest.fixture
def db():
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    yield database
    database.dispose()


def test_high_confidence_memory_is_published_and_searchable(db) -> None:
    with db.session() as session:
        service = MemoryService(session, HashEmbeddings(8))
        candidate = service.propose(
            user_id="user-1", kind="preference", content="报告使用中文",
            source="USER", confidence=1.0,
            evidence={"source": "user_message"},
        )
        assert candidate.status == "PUBLISHED"
        record = session.scalar(select(MemoryRecord).where(MemoryRecord.user_id == "user-1"))
        assert record is not None
        hits = service.repository.search_published(
            user_id="user-1", query_vector=HashEmbeddings(8).embed_query("报告使用中文")
        )
        assert isinstance(hits[0], SearchHit)
        assert hits[0].metadata["status"] == "PUBLISHED"


def test_conflicting_memory_waits_for_resolution(db) -> None:
    with db.session() as session:
        service = MemoryService(session, HashEmbeddings(8))
        first = service.propose(
            user_id="user-1", kind="location", content="住在上海",
            source="USER", confidence=1.0,
        )
        second = service.propose(
            user_id="user-1", kind="location", content="住在杭州",
            source="USER", confidence=1.0,
        )
        assert first.status == "PUBLISHED"
        assert second.status == "CONFLICT"
        old = session.scalar(select(MemoryRecord).where(MemoryRecord.status == "PUBLISHED"))
        assert old is not None
        replacement = service.resolve_conflict(second.id, old.id)
        assert replacement.status == "PUBLISHED"
        assert replacement.version == 2
        assert old.status == "SUPERSEDED"


def test_sensitive_or_low_confidence_memory_stays_candidate(db) -> None:
    with db.session() as session:
        service = MemoryService(session, HashEmbeddings(8))
        candidate = service.propose(
            user_id="user-1", kind="sensitive", content="敏感信息",
            source="LLM_INFERENCE", confidence=0.99, sensitive=True,
        )
        assert candidate.status == "CANDIDATE"
        with pytest.raises(LookupError):
            service.withdraw("missing")


def test_withdraw_published_memory_within_window(db) -> None:
    with db.session() as session:
        service = MemoryService(session, HashEmbeddings(8))
        candidate = service.propose(
            user_id="user-1", kind="preference", content="喜欢骑行",
            source="USER", confidence=1.0,
        )
        record = session.scalar(select(MemoryRecord).where(MemoryRecord.user_id == "user-1"))
        assert record is not None
        service.withdraw(record.id)
        assert record.status == "DELETED"
        assert session.scalars(select(MemoryCandidate)).first() is not None


def test_stale_candidate_can_expire(db) -> None:
    with db.session() as session:
        service = MemoryService(session, HashEmbeddings(8))
        candidate = service.propose(
            user_id="user-1", kind="inference", content="可能喜欢徒步",
            source="LLM_INFERENCE", confidence=0.4,
        )
        candidate.withdraw_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        expired = service.expire_candidates()
        assert [item.id for item in expired] == [candidate.id]
        assert candidate.status == "EXPIRED"
