from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from personal_assistant.agent.execution import (
    ApprovalService,
    ToolExecutor,
    ToolOutcome,
)
from personal_assistant.storage.database import Database


@dataclass
class FakeTool:
    calls: int = 0
    outcomes: dict[str, ToolOutcome] = field(default_factory=dict)

    def execute(self, arguments, *, idempotency_key):
        self.calls += 1
        outcome = ToolOutcome(f"external-{self.calls}", {"echo": dict(arguments)})
        self.outcomes[idempotency_key] = outcome
        return outcome

    def reconcile(self, *, idempotency_key):
        return self.outcomes.get(idempotency_key)


def test_tool_executor_is_idempotent_and_recoverable() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    tool = FakeTool()
    try:
        executor = ToolExecutor(database.session_factory, {"notify": tool})
        first = executor.execute(user_id="user-1", tool_name="notify", arguments={"text": "hi"}, idempotency_key="k1", acknowledge=False)
        second = executor.execute(user_id="user-1", tool_name="notify", arguments={"text": "hi"}, idempotency_key="k1")
        assert first.id == second.id
        assert tool.calls == 1
        assert second.status == "COMPLETED"
        assert len(second.trace) == 4
    finally:
        database.dispose()


def test_approval_confirmation_requires_snapshot_and_workflow() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        service = ApprovalService(database.session_factory)
        request = service.request(user_id="user-1", snapshot={"amount": 10}, workflow_version="v1")
        confirmed = service.confirm(request_id=request.id, user_id="user-1", snapshot_hash=request.snapshot_hash, workflow_version="v1")
        assert confirmed.status == "SUBMITTING"
    finally:
        database.dispose()


def test_approval_service_resolve_records_terminal_decision() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        service = ApprovalService(database.session_factory)
        request = service.request(user_id="user-1", snapshot={"amount": 10}, workflow_version="v1")
        resolved = service.resolve(request_id=request.id, user_id="user-1", approved=True)
        assert resolved.status == "APPROVED"
    finally:
        database.dispose()


def test_tool_executor_stops_retrying_after_max_attempts() -> None:
    @dataclass
    class AlwaysFailTool:
        calls: int = 0

        def execute(self, arguments, *, idempotency_key):
            self.calls += 1
            raise TimeoutError("remote timeout")

        def reconcile(self, *, idempotency_key):
            return None

    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    tool = AlwaysFailTool()
    try:
        executor = ToolExecutor(
            database.session_factory,
            {"unstable": tool},
            max_attempts=2,
        )
        first = executor.execute(
            user_id="user-1",
            tool_name="unstable",
            arguments={"x": 1},
            idempotency_key="retry-key",
        )
        second = executor.execute(
            user_id="user-1",
            tool_name="unstable",
            arguments={"x": 1},
            idempotency_key="retry-key",
        )
        third = executor.execute(
            user_id="user-1",
            tool_name="unstable",
            arguments={"x": 1},
            idempotency_key="retry-key",
        )
        assert first.status == "EXECUTED_FAILED"
        assert second.status == "EXECUTED_FAILED"
        assert third.status == "EXECUTED_FAILED"
        assert tool.calls == 2
        assert third.attempt == 2
        assert any(event["event"] == "retry_exhausted" for event in third.trace)
    finally:
        database.dispose()


def test_approval_service_checks_thread_scope() -> None:
    from personal_assistant.storage.repositories import ThreadRepository

    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        with database.session() as session:
            thread = ThreadRepository(session).create("user-1", thread_id="thread-1")
        service = ApprovalService(database.session_factory)
        request = service.request(
            user_id="user-1",
            snapshot={"amount": 10},
            workflow_version="v1",
            thread_id=thread.id,
        )
        try:
            service.resolve(
                request_id=request.id,
                user_id="user-1",
                approved=True,
                thread_id="other-thread",
            )
        except PermissionError as exc:
            assert "thread" in str(exc)
        else:
            raise AssertionError("expected thread scope validation")
    finally:
        database.dispose()


def test_approval_service_rejects_conflicting_terminal_decisions() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        service = ApprovalService(database.session_factory)
        request = service.request(user_id="user-1", snapshot={"amount": 10}, workflow_version="v1")
        assert service.resolve(request_id=request.id, user_id="user-1", approved=False).status == "REJECTED"
        with pytest.raises(ValueError, match="REJECTED"):
            service.resolve(request_id=request.id, user_id="user-1", approved=True)

        request = service.request(user_id="user-1", snapshot={"amount": 11}, workflow_version="v1")
        service.begin_resolution(request_id=request.id, user_id="user-1", approved=True)
        assert service.get(request_id=request.id).status == "SUBMITTING"
        with pytest.raises(ValueError, match="SUBMITTING"):
            service.begin_resolution(request_id=request.id, user_id="user-1", approved=False)
        unknown = service.mark_unknown(request_id=request.id, user_id="user-1")
        assert unknown.status == "UNKNOWN"
        assert service.resolve(request_id=request.id, user_id="user-1", approved=True).status == "APPROVED"
    finally:
        database.dispose()


def test_tool_retry_exhaustion_writes_audit_event() -> None:
    from sqlalchemy import select

    from personal_assistant.storage.models import AuditEvent

    @dataclass
    class AlwaysFailTool:
        def execute(self, arguments, *, idempotency_key):
            raise TimeoutError("remote timeout")

        def reconcile(self, *, idempotency_key):
            return None

    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        executor = ToolExecutor(
            database.session_factory,
            {"unstable": AlwaysFailTool()},
            max_attempts=1,
        )
        executor.execute(
            user_id="user-1",
            tool_name="unstable",
            arguments={"x": 1},
            idempotency_key="audit-retry-key",
        )
        executor.execute(
            user_id="user-1",
            tool_name="unstable",
            arguments={"x": 1},
            idempotency_key="audit-retry-key",
        )
        with database.session() as session:
            event = session.scalar(
                select(AuditEvent).where(AuditEvent.event_type == "TOOL_RUN_RETRY_EXHAUSTED")
            )
            assert event is not None
            assert event.payload["max_attempts"] == 1
    finally:
        database.dispose()


def test_approval_service_enforces_user_scope_without_thread() -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        service = ApprovalService(database.session_factory)
        request = service.request(user_id="user-1", snapshot={"amount": 10}, workflow_version="v1")
        with pytest.raises(PermissionError, match="configured user"):
            service.get(request_id=request.id, user_id="user-2")
        with pytest.raises(PermissionError, match="configured user"):
            service.resolve(request_id=request.id, user_id="user-2", approved=False)
    finally:
        database.dispose()
