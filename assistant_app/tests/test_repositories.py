from __future__ import annotations

import pytest

from sqlalchemy import select

from personal_assistant.storage.database import Database
from personal_assistant.storage.models import AuditEvent
from personal_assistant.storage.repositories import (
    ApprovalRepository,
    InvalidStateTransition,
    TaskRepository,
    ThreadRepository,
    ToolRunRepository,
)


@pytest.fixture
def db():
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    yield database
    database.dispose()


def test_thread_messages_are_sequenced_and_audited(db):
    with db.session() as session:
        threads = ThreadRepository(session)
        thread = threads.create("user-1", thread_id="thread-1")
        first = threads.append_message(thread.id, "user", "你好")
        second = threads.append_message(thread.id, "assistant", "你好！")
        assert (first.event_seq, second.event_seq) == (1, 2)
        events = session.scalars(select(AuditEvent)).all()
        assert [event.event_type for event in events] == ["THREAD_CREATED", "MESSAGE_CREATED", "MESSAGE_CREATED"]


def test_task_transition_rejects_skipping_states(db):
    with db.session() as session:
        repo = TaskRepository(session)
        task = repo.create("user-1", "提交报销")
        with pytest.raises(InvalidStateTransition):
            repo.transition(task.id, "COMPLETED")
        repo.transition(task.id, "SCHEDULED")
        assert task.status == "SCHEDULED"


def test_tool_idempotency_returns_same_run_and_audits_once(db):
    with db.session() as session:
        repo = ToolRunRepository(session)
        kwargs = dict(
            user_id="user-1", tool_name="weather", invocation_id="inv-1",
            input_hash="hash", idempotency_key="weather:user-1:today",
        )
        first = repo.create_idempotent(**kwargs)
        second = repo.create_idempotent(**{**kwargs, "invocation_id": "inv-2"})
        assert first.id == second.id
        repo.transition(first.id, "EXECUTING", user_id="user-1")
        repo.transition(first.id, "EXECUTED_SUCCESS_UNACK", user_id="user-1")
        repo.transition(first.id, "COMPLETED", user_id="user-1")
        assert len(first.trace) == 4


def test_approval_binds_confirmation_to_snapshot_and_workflow(db):
    with db.session() as session:
        repo = ApprovalRepository(session)
        request = repo.create(user_id="user-1", snapshot={"amount": 100}, workflow_version="v1")
        repo.transition(request.id, "PENDING_CONFIRMATION", user_id="user-1")
        with pytest.raises(InvalidStateTransition):
            repo.transition(request.id, "SUBMITTING", user_id="user-1", expected_snapshot_hash="wrong")
        repo.transition(
            request.id, "SUBMITTING", user_id="user-1",
            expected_snapshot_hash=request.snapshot_hash,
            expected_workflow_version="v1",
        )
        assert request.status == "SUBMITTING"
