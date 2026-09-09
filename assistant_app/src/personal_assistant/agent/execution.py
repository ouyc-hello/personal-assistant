from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from personal_assistant.storage.models import ApprovalRequest, Thread, ToolRun
from personal_assistant.storage.repositories import (
    ApprovalRepository,
    ToolRunRepository,
)


@dataclass(frozen=True)
class ToolOutcome:
    external_id: str
    result: Mapping[str, Any]


class IdempotentTool(Protocol):
    def execute(self, arguments: Mapping[str, Any], *, idempotency_key: str) -> ToolOutcome: ...

    def reconcile(self, *, idempotency_key: str) -> ToolOutcome | None: ...


class ToolExecutor:
    """Persist tool state around an idempotent external side effect."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        tools: Mapping[str, IdempotentTool],
        *,
        max_attempts: int = 4,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.session_factory = session_factory
        self.tools = dict(tools)
        self.max_attempts = max_attempts

    def execute(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        idempotency_key: str,
        thread_id: str | None = None,
        invocation_id: str | None = None,
        acknowledge: bool = True,
    ) -> ToolRun:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise LookupError(f"tool not configured: {tool_name}")
        input_hash = _stable_hash(arguments)
        with self.session_factory() as session:
            repo = ToolRunRepository(session)
            run = repo.create_idempotent(
                user_id=user_id,
                tool_name=tool_name,
                invocation_id=invocation_id or str(uuid4()),
                input_hash=input_hash,
                idempotency_key=idempotency_key,
                thread_id=thread_id,
            )
            if run.input_hash != input_hash:
                raise ValueError("idempotency key was reused with different arguments")
            if run.status == "COMPLETED":
                return run
            if run.status == "EXECUTED_SUCCESS_UNACK" and acknowledge:
                result = repo.transition(
                    run.id,
                    "COMPLETED",
                    user_id=user_id,
                    trace_event={"event": "acknowledged"},
                )
                session.commit()
                return result
            if run.status not in {"CREATED", "EXECUTED_FAILED"}:
                session.commit()
                return run
            if run.status == "EXECUTED_FAILED":
                if run.attempt >= self.max_attempts:
                    run = repo.append_trace(
                        run.id,
                        user_id=user_id,
                        trace_event={"event": "retry_exhausted", "max_attempts": self.max_attempts},
                        audit_event_type="TOOL_RUN_RETRY_EXHAUSTED",
                    )
                    session.commit()
                    return run
                run.attempt += 1
                run.trace = [*run.trace, {"event": "retry_scheduled", "attempt": run.attempt}]
                session.flush()
            repo.transition(run.id, "EXECUTING", user_id=user_id, trace_event={"event": "execution_started", "attempt": run.attempt})
            try:
                outcome = tool.execute(arguments, idempotency_key=idempotency_key)
            except Exception as exc:  # noqa: BLE001 - external tool boundary
                result = repo.transition(
                    run.id,
                    "EXECUTED_FAILED",
                    user_id=user_id,
                    trace_event={
                        "event": "execution_failed",
                        "error_type": type(exc).__name__,
                        "attempt": run.attempt,
                    },
                )
                session.commit()
                return result
            run = repo.transition(
                run.id,
                "EXECUTED_SUCCESS_UNACK",
                user_id=user_id,
                trace_event={"event": "external_success", "external_id": outcome.external_id, "result": dict(outcome.result)},
            )
            if acknowledge:
                run = repo.transition(run.id, "COMPLETED", user_id=user_id, trace_event={"event": "acknowledged"})
            session.commit()
            return run

    def recover(self, *, user_id: str, limit: int = 20) -> list[ToolRun]:
        recovered: list[ToolRun] = []
        with self.session_factory() as session:
            runs = session.scalars(select(ToolRun).where(ToolRun.status.in_(("EXECUTING", "EXECUTED_SUCCESS_UNACK"))).limit(limit)).all()
            for run in runs:
                tool = self.tools.get(run.tool_name)
                if tool is None:
                    continue
                if run.status == "EXECUTED_SUCCESS_UNACK":
                    recovered.append(ToolRunRepository(session).transition(run.id, "COMPLETED", user_id=user_id, trace_event={"event": "recovered_ack"}))
                    continue
                outcome = tool.reconcile(idempotency_key=run.idempotency_key)
                if outcome is None:
                    continue
                repo = ToolRunRepository(session)
                repo.transition(run.id, "EXECUTED_SUCCESS_UNACK", user_id=user_id, trace_event={"event": "reconciled_success", "external_id": outcome.external_id})
                recovered.append(repo.transition(run.id, "COMPLETED", user_id=user_id, trace_event={"event": "recovered_ack"}))
            session.commit()
        return recovered


class ApprovalService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def request(
        self,
        *,
        user_id: str,
        snapshot: Mapping[str, Any],
        workflow_version: str,
        thread_id: str | None = None,
    ) -> ApprovalRequest:
        with self.session_factory() as session:
            if thread_id is not None:
                thread = session.get(Thread, thread_id)
                if thread is None:
                    raise LookupError(f"thread not found: {thread_id}")
                if thread.user_id != user_id:
                    raise PermissionError("thread does not belong to the configured user")
            request = ApprovalRepository(session).create(
                user_id=user_id,
                snapshot=snapshot,
                workflow_version=workflow_version,
                thread_id=thread_id,
            )
            ApprovalRepository(session).transition(request.id, "PENDING_CONFIRMATION", user_id=user_id)
            session.commit()
            return request

    def get(
        self,
        *,
        request_id: str,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> ApprovalRequest:
        with self.session_factory() as session:
            request = ApprovalRepository(session).get(request_id)
            if request is None:
                raise LookupError(f"approval request not found: {request_id}")
            self._validate_scope(
                session, request, user_id=user_id, thread_id=thread_id
            )
            return request

    def confirm(self, *, request_id: str, user_id: str, snapshot_hash: str, workflow_version: str) -> ApprovalRequest:
        with self.session_factory() as session:
            request = ApprovalRepository(session).transition(
                request_id,
                "SUBMITTING",
                user_id=user_id,
                expected_snapshot_hash=snapshot_hash,
                expected_workflow_version=workflow_version,
            )
            session.commit()
            return request

    def begin_resolution(
        self,
        *,
        request_id: str,
        user_id: str,
        approved: bool,
        thread_id: str | None = None,
    ) -> ApprovalRequest:
        """Durably mark a decision before resuming a paused graph.

        ``SUBMITTING`` is a recovery boundary: if checkpoint resume fails after
        this commit, the request can be marked ``UNKNOWN`` and retried without
        silently treating an incomplete operation as approved.
        """
        with self.session_factory() as session:
            repo = ApprovalRepository(session)
            request = repo.get(request_id)
            if request is None:
                raise LookupError(f"approval request not found: {request_id}")
            self._validate_scope(session, request, user_id=user_id, thread_id=thread_id)
            if not approved:
                if request.status == "REJECTED":
                    return request
                if request.status == "APPROVED":
                    raise ValueError("approval request is already APPROVED")
                if request.status == "SUBMITTING":
                    raise ValueError("approval request is currently SUBMITTING; recover it before rejecting")
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN", "SUBMITTED"}:
                    repo.transition(request_id, "REJECTED", user_id=user_id)
                else:
                    raise ValueError(f"approval request cannot be rejected from {request.status}")
            else:
                if request.status == "APPROVED":
                    return request
                if request.status == "REJECTED":
                    raise ValueError("approval request is already REJECTED")
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN"}:
                    repo.transition(
                        request_id,
                        "SUBMITTING",
                        user_id=user_id,
                        expected_snapshot_hash=request.snapshot_hash,
                        expected_workflow_version=request.workflow_version,
                    )
                elif request.status not in {"SUBMITTING", "SUBMITTED"}:
                    raise ValueError(f"approval request cannot be approved from {request.status}")
            session.commit()
            return request

    def mark_unknown(
        self,
        *,
        request_id: str,
        user_id: str,
        thread_id: str | None = None,
    ) -> ApprovalRequest:
        """Record that checkpoint resume and business state are inconsistent."""
        with self.session_factory() as session:
            repo = ApprovalRepository(session)
            request = repo.get(request_id)
            if request is None:
                raise LookupError(f"approval request not found: {request_id}")
            self._validate_scope(session, request, user_id=user_id, thread_id=thread_id)
            if request.status == "UNKNOWN":
                return request
            if request.status not in {"SUBMITTING", "SUBMITTED"}:
                raise ValueError(f"approval request cannot be marked UNKNOWN from {request.status}")
            request = repo.transition(request_id, "UNKNOWN", user_id=user_id)
            session.commit()
            return request

    def resolve(
        self,
        *,
        request_id: str,
        user_id: str,
        approved: bool,
        thread_id: str | None = None,
    ) -> ApprovalRequest:
        """Record the terminal decision with explicit, idempotent state rules.

        This method remains convenient for callers that do not have a separate
        checkpoint resume step. The CLI uses ``begin_resolution`` first so a
        failed resume is recoverable as ``UNKNOWN`` rather than looking approved.
        """
        with self.session_factory() as session:
            repo = ApprovalRepository(session)
            request = repo.get(request_id)
            if request is None:
                raise LookupError(f"approval request not found: {request_id}")
            self._validate_scope(session, request, user_id=user_id, thread_id=thread_id)
            if approved:
                if request.status == "APPROVED":
                    return request
                if request.status == "REJECTED":
                    raise ValueError("approval request is already REJECTED")
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN"}:
                    repo.transition(
                        request_id,
                        "SUBMITTING",
                        user_id=user_id,
                        expected_snapshot_hash=request.snapshot_hash,
                        expected_workflow_version=request.workflow_version,
                    )
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN", "SUBMITTING"}:
                    repo.transition(request_id, "SUBMITTED", user_id=user_id)
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN", "SUBMITTING", "SUBMITTED"}:
                    repo.transition(request_id, "APPROVED", user_id=user_id)
            else:
                if request.status == "REJECTED":
                    return request
                if request.status == "APPROVED":
                    raise ValueError("approval request is already APPROVED")
                if request.status == "SUBMITTING":
                    raise ValueError("approval request is currently SUBMITTING; mark it UNKNOWN before rejecting")
                if request.status in {"PENDING_CONFIRMATION", "UNKNOWN", "SUBMITTED"}:
                    repo.transition(request_id, "REJECTED", user_id=user_id)
                else:
                    raise ValueError(f"approval request cannot be rejected from {request.status}")
            session.commit()
            return request


    @staticmethod
    def _validate_scope(
        session: Session,
        request: ApprovalRequest,
        *,
        user_id: str | None,
        thread_id: str | None,
    ) -> None:
        if thread_id is not None and request.thread_id != thread_id:
            raise PermissionError("approval request does not belong to the supplied thread")
        if user_id is not None and request.user_id != user_id:
            raise PermissionError("approval request does not belong to the configured user")
        if request.thread_id is not None:
            thread = session.get(Thread, request.thread_id)
            if thread is None or thread.user_id != request.user_id:
                raise PermissionError("approval request thread owner does not match its user scope")


def _stable_hash(arguments: Mapping[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
