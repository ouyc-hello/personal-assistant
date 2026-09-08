from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from personal_assistant.rag.models import IndexedChunk
from personal_assistant.schemas import SearchHit
from personal_assistant.storage.models import (
    ApprovalRequest,
    AuditEvent,
    KnowledgeChunk,
    KnowledgeDocument,
    Message,
    Task,
    Thread,
    ToolRun,
    MemoryCandidate,
    MemoryRecord,
    new_id,
)
from personal_assistant.storage.state import (
    APPROVAL_TRANSITIONS,
    TASK_TRANSITIONS,
    TOOL_TRANSITIONS,
)


class InvalidStateTransition(ValueError):
    pass


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _hash(canonical)


def _audit(session: Session, *, user_id: str, event_type: str, entity_type: str,
           entity_id: str, payload: Mapping[str, Any] | None = None) -> AuditEvent:
    event = AuditEvent(
        user_id=user_id,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        payload=dict(payload or {}),
    )
    session.add(event)
    return event


class ThreadRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: str, thread_id: str | None = None) -> Thread:
        thread = Thread(id=thread_id or new_id(), user_id=user_id)
        self.session.add(thread)
        self.session.flush()
        _audit(
            self.session,
            user_id=user_id,
            event_type="THREAD_CREATED",
            entity_type="thread",
            entity_id=thread.id,
        )
        return thread

    def get(self, thread_id: str) -> Thread | None:
        return self.session.get(Thread, thread_id)

    def append_message(self, thread_id: str, role: str, content: str) -> Message:
        thread = self.session.get(Thread, thread_id)
        if thread is None:
            raise LookupError(f"thread not found: {thread_id}")
        current = self.session.scalar(
            select(func.max(Message.event_seq)).where(Message.thread_id == thread_id)
        )
        message = Message(
            thread_id=thread_id,
            role=role,
            content=content,
            event_seq=int(current or 0) + 1,
        )
        self.session.add(message)
        thread.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(
            self.session,
            user_id=thread.user_id,
            event_type="MESSAGE_CREATED",
            entity_type="message",
            entity_id=message.id,
            payload={"thread_id": thread_id, "role": role, "content_sha256": _hash(content), "content_length": len(content)},
        )
        return message


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_for_entity(self, entity_type: str, entity_id: str) -> list[AuditEvent]:
        return list(
            self.session.scalars(
                select(AuditEvent)
                .where(AuditEvent.entity_type == entity_type, AuditEvent.entity_id == entity_id)
                .order_by(AuditEvent.id)
            )
        )


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_candidate(self, *, user_id: str, kind: str, content: str, source: str,
                         confidence: float, evidence: Mapping[str, Any] | None,
                         embedding: list[float] | None, withdraw_deadline: datetime | None) -> MemoryCandidate:
        candidate = MemoryCandidate(
            user_id=user_id,
            kind=kind,
            content=content,
            source=source,
            confidence=confidence,
            status="CANDIDATE",
            evidence=dict(evidence or {}),
            embedding=embedding,
            withdraw_deadline=withdraw_deadline,
        )
        self.session.add(candidate)
        self.session.flush()
        _audit(
            self.session,
            user_id=user_id,
            event_type="MEMORY_CANDIDATE_CREATED",
            entity_type="memory_candidate",
            entity_id=candidate.id,
            payload={"kind": kind, "source": source, "confidence": confidence},
        )
        return candidate

    def find_conflicts(self, *, user_id: str, kind: str, content: str) -> list[MemoryRecord]:
        normalized = _normalize_text(content)
        records = self.session.scalars(
            select(MemoryRecord).where(
                MemoryRecord.user_id == user_id,
                MemoryRecord.kind == kind,
                MemoryRecord.status == "PUBLISHED",
            )
        ).all()
        return [record for record in records if _normalize_text(record.content) != normalized]

    def mark_candidate(self, candidate_id: str, status: str) -> MemoryCandidate:
        candidate = self.session.get(MemoryCandidate, candidate_id)
        if candidate is None:
            raise LookupError(f"memory candidate not found: {candidate_id}")
        candidate.status = status
        self.session.flush()
        _audit(
            self.session,
            user_id=candidate.user_id,
            event_type=f"MEMORY_CANDIDATE_{status}",
            entity_type="memory_candidate",
            entity_id=candidate.id,
            payload={"status": status},
        )
        return candidate

    def publish_candidate(self, candidate_id: str) -> MemoryRecord:
        candidate = self.session.get(MemoryCandidate, candidate_id)
        if candidate is None:
            raise LookupError(f"memory candidate not found: {candidate_id}")
        if candidate.status != "CANDIDATE":
            raise InvalidStateTransition(f"cannot publish memory candidate in status {candidate.status}")
        return self._publish_candidate(candidate)

    def _publish_candidate(self, candidate: MemoryCandidate, *, version: int = 1) -> MemoryRecord:
        record = MemoryRecord(
            user_id=candidate.user_id,
            kind=candidate.kind,
            content=candidate.content,
            status="PUBLISHED",
            source=candidate.source,
            confidence=candidate.confidence,
            version=version,
            evidence=dict(candidate.evidence),
            withdraw_deadline=candidate.withdraw_deadline,
            embedding=candidate.embedding,
        )
        self.session.add(record)
        self.session.flush()
        candidate.status = "PUBLISHED"
        candidate.memory_record_id = record.id
        self.session.flush()
        _audit(
            self.session,
            user_id=candidate.user_id,
            event_type="MEMORY_PUBLISHED",
            entity_type="memory_record",
            entity_id=record.id,
            payload={"candidate_id": candidate.id, "kind": record.kind, "version": record.version},
        )
        return record

    def resolve_conflict(self, candidate_id: str, supersede_record_id: str) -> MemoryRecord:
        candidate = self.session.get(MemoryCandidate, candidate_id)
        old = self.session.get(MemoryRecord, supersede_record_id)
        if candidate is None or old is None:
            raise LookupError("memory candidate or record not found")
        if candidate.status not in {"CANDIDATE", "CONFLICT"}:
            raise InvalidStateTransition(f"cannot resolve candidate in status {candidate.status}")
        if old.user_id != candidate.user_id or old.kind != candidate.kind or old.status != "PUBLISHED":
            raise InvalidStateTransition("memory conflict does not match the published record")
        old.status = "SUPERSEDED"
        old.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(
            self.session,
            user_id=candidate.user_id,
            event_type="MEMORY_SUPERSEDED",
            entity_type="memory_record",
            entity_id=old.id,
            payload={"replacement_candidate_id": candidate.id},
        )
        return self._publish_candidate(candidate, version=old.version + 1)

    def expire_candidates(self, *, now: datetime | None = None) -> list[MemoryCandidate]:
        cutoff = now or datetime.now(timezone.utc)
        candidates = self.session.scalars(
            select(MemoryCandidate).where(
                MemoryCandidate.status.in_(("CANDIDATE", "CONFLICT")),
                MemoryCandidate.withdraw_deadline.is_not(None),
                MemoryCandidate.withdraw_deadline <= cutoff,
            )
        ).all()
        for candidate in candidates:
            candidate.status = "EXPIRED"
            _audit(
                self.session,
                user_id=candidate.user_id,
                event_type="MEMORY_CANDIDATE_EXPIRED",
                entity_type="memory_candidate",
                entity_id=candidate.id,
            )
        self.session.flush()
        return list(candidates)

    def withdraw(self, record_id: str, *, actor: str = "user") -> MemoryRecord:
        record = self.session.get(MemoryRecord, record_id)
        if record is None:
            raise LookupError(f"memory record not found: {record_id}")
        deadline = record.withdraw_deadline
        if deadline is not None:
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > deadline and actor == "user":
                raise InvalidStateTransition("memory withdrawal window has expired")
        if record.status != "PUBLISHED":
            raise InvalidStateTransition(f"cannot withdraw memory in status {record.status}")
        record.status = "DELETED"
        record.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(
            self.session,
            user_id=record.user_id,
            event_type="MEMORY_DELETED",
            entity_type="memory_record",
            entity_id=record.id,
            payload={"actor": actor},
        )
        return record

    def search_published(self, *, user_id: str, query_vector: list[float], limit: int = 5) -> list[SearchHit]:
        if limit <= 0:
            return []
        base = select(MemoryRecord).where(
            MemoryRecord.user_id == user_id,
            MemoryRecord.status == "PUBLISHED",
            MemoryRecord.embedding.is_not(None),
        )
        bind = self.session.get_bind()
        if bind is not None and bind.dialect.name == "postgresql":
            distance = MemoryRecord.embedding.op("<=>")(query_vector).label("distance")
            rows = self.session.execute(base.add_columns(distance).order_by(distance).limit(limit)).all()
            return [self._hit(record, 1.0 - float(distance)) for record, distance in rows]

        records = self.session.scalars(base).all()
        scored = [(record, _cosine(record.embedding or [], query_vector)) for record in records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [self._hit(record, score) for record, score in scored[:limit]]

    @staticmethod
    def _hit(record: MemoryRecord, score: float) -> SearchHit:
        return SearchHit(
            id=record.id,
            text=record.content,
            score=score,
            backend="postgresql+pgvector",
            metadata={
                "memory_id": record.id,
                "status": record.status,
                "kind": record.kind,
                "source": record.source,
                "confidence": record.confidence,
                "version": record.version,
            },
        )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class KnowledgeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def register_document(self, *, document_id: str, user_id: str, source_uri: str,
                          content_hash: str, version: int) -> None:
        document = self.session.get(KnowledgeDocument, document_id)
        if document is None:
            document = KnowledgeDocument(
                id=document_id,
                user_id=user_id,
                source_uri=source_uri,
                content_hash=content_hash,
                version=version,
                status="INDEXING",
            )
            self.session.add(document)
        else:
            document.status = "INDEXING"
            document.version = version
        self.session.flush()
        _audit(
            self.session,
            user_id=user_id,
            event_type="KNOWLEDGE_DOCUMENT_INDEXING",
            entity_type="knowledge_document",
            entity_id=document_id,
            payload={"source_uri": source_uri, "content_hash": content_hash, "version": version},
        )

    def register_chunks(self, chunks: list[IndexedChunk]) -> None:
        for chunk in chunks:
            existing = self.session.get(KnowledgeChunk, chunk.id)
            if existing is None:
                self.session.add(
                    KnowledgeChunk(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        content_hash=_hash(chunk.text),
                        milvus_id=chunk.id,
                        metadata_json={**chunk.metadata, "source_uri": chunk.source_uri, "page": chunk.page},
                    )
                )
        self.session.flush()

    def set_document_status(self, document_id: str, status: str) -> None:
        document = self.session.get(KnowledgeDocument, document_id)
        if document is None:
            raise LookupError(f"knowledge document not found: {document_id}")
        document.status = status
        document.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(
            self.session,
            user_id=document.user_id,
            event_type=f"KNOWLEDGE_DOCUMENT_{status}",
            entity_type="knowledge_document",
            entity_id=document.id,
            payload={"status": status},
        )


class TaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: str, title: str, *, due_at: datetime | None = None,
               metadata: Mapping[str, Any] | None = None) -> Task:
        task = Task(
            user_id=user_id,
            title=title,
            status="CREATED",
            due_at=due_at,
            metadata_json=dict(metadata or {}),
        )
        self.session.add(task)
        self.session.flush()
        _audit(self.session, user_id=user_id, event_type="TASK_CREATED", entity_type="task", entity_id=task.id)
        return task

    def transition(self, task_id: str, new_status: str) -> Task:
        task = self.session.get(Task, task_id)
        if task is None:
            raise LookupError(f"task not found: {task_id}")
        allowed = TASK_TRANSITIONS.get(task.status, set())
        if new_status not in allowed:
            raise InvalidStateTransition(f"task: {task.status} -> {new_status} is not allowed")
        old_status = task.status
        task.status = new_status
        task.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(
            self.session,
            user_id=task.user_id,
            event_type="TASK_STATUS_CHANGED",
            entity_type="task",
            entity_id=task.id,
            payload={"from": old_status, "to": new_status},
        )
        return task


class ToolRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_idempotent(self, *, user_id: str, tool_name: str, invocation_id: str,
                          input_hash: str, idempotency_key: str, thread_id: str | None = None) -> ToolRun:
        existing = self.session.scalar(
            select(ToolRun).where(
                ToolRun.tool_name == tool_name,
                ToolRun.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return existing
        run = ToolRun(
            thread_id=thread_id,
            invocation_id=invocation_id,
            tool_name=tool_name,
            input_hash=input_hash,
            idempotency_key=idempotency_key,
            status="CREATED",
            trace=[{"event": "created", "invocation_id": invocation_id}],
        )
        self.session.add(run)
        self.session.flush()
        _audit(
            self.session,
            user_id=user_id,
            event_type="TOOL_RUN_CREATED",
            entity_type="tool_run",
            entity_id=run.id,
            payload={"tool_name": tool_name, "idempotency_key": idempotency_key},
        )
        return run

    def transition(self, run_id: str, new_status: str, *, user_id: str,
                   trace_event: Mapping[str, Any] | None = None) -> ToolRun:
        run = self.session.get(ToolRun, run_id)
        if run is None:
            raise LookupError(f"tool run not found: {run_id}")
        allowed = TOOL_TRANSITIONS.get(run.status, set())
        if new_status not in allowed:
            raise InvalidStateTransition(f"tool_run: {run.status} -> {new_status} is not allowed")
        old_status = run.status
        run.status = new_status
        run.updated_at = datetime.now(timezone.utc)
        run.trace = [*run.trace, dict(trace_event or {"event": "status_changed", "to": new_status})]
        self.session.flush()
        _audit(
            self.session,
            user_id=user_id,
            event_type=f"TOOL_RUN_{new_status}",
            entity_type="tool_run",
            entity_id=run.id,
            payload={"from": old_status, "to": new_status},
        )
        return run


class ApprovalRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, *, user_id: str, snapshot: Mapping[str, Any], workflow_version: str,
               thread_id: str | None = None) -> ApprovalRequest:
        data = dict(snapshot)
        request = ApprovalRequest(
            thread_id=thread_id,
            workflow_version=workflow_version,
            snapshot_hash=snapshot_hash(data),
            status="DRAFT",
            snapshot=data,
        )
        self.session.add(request)
        self.session.flush()
        _audit(self.session, user_id=user_id, event_type="APPROVAL_CREATED", entity_type="approval_request", entity_id=request.id,
               payload={"snapshot_hash": request.snapshot_hash, "workflow_version": workflow_version})
        return request

    def transition(self, request_id: str, new_status: str, *, user_id: str,
                   expected_snapshot_hash: str | None = None,
                   expected_workflow_version: str | None = None) -> ApprovalRequest:
        request = self.session.get(ApprovalRequest, request_id)
        if request is None:
            raise LookupError(f"approval request not found: {request_id}")
        if expected_snapshot_hash is not None and expected_snapshot_hash != request.snapshot_hash:
            raise InvalidStateTransition("approval snapshot changed; confirmation is invalid")
        if expected_workflow_version is not None and expected_workflow_version != request.workflow_version:
            raise InvalidStateTransition("approval workflow version changed; confirmation is invalid")
        if snapshot_hash(request.snapshot) != request.snapshot_hash:
            raise InvalidStateTransition("approval snapshot integrity check failed")
        allowed = APPROVAL_TRANSITIONS.get(request.status, set())
        if new_status not in allowed:
            raise InvalidStateTransition(f"approval: {request.status} -> {new_status} is not allowed")
        old_status = request.status
        request.status = new_status
        request.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        _audit(self.session, user_id=user_id, event_type="APPROVAL_STATUS_CHANGED", entity_type="approval_request", entity_id=request.id,
               payload={"from": old_status, "to": new_status, "snapshot_hash": request.snapshot_hash})
        return request
