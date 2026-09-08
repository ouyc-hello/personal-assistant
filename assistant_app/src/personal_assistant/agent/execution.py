from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from personal_assistant.storage.models import ApprovalRequest, ToolRun
from personal_assistant.storage.repositories import ApprovalRepository, ToolRunRepository


@dataclass(frozen=True)
class ToolOutcome:
    external_id: str
    result: Mapping[str, Any]


class IdempotentTool(Protocol):
    def execute(self, arguments: Mapping[str, Any], *, idempotency_key: str) -> ToolOutcome: ...

    def reconcile(self, *, idempotency_key: str) -> ToolOutcome | None: ...


class ToolExecutor:
    """Persist tool state around an idempotent external side effect."""

    def __init__(self, session_factory: sessionmaker[Session], tools: Mapping[str, IdempotentTool]) -> None:
        self.session_factory = session_factory
        self.tools = dict(tools)

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
                return repo.transition(run.id, "COMPLETED", user_id=user_id, trace_event={"event": "acknowledged"})
            if run.status not in {"CREATED", "EXECUTED_FAILED"}:
                return run
            repo.transition(run.id, "EXECUTING", user_id=user_id, trace_event={"event": "execution_started"})
            try:
                outcome = tool.execute(arguments, idempotency_key=idempotency_key)
            except Exception as exc:
                return repo.transition(
                    run.id,
                    "EXECUTED_FAILED",
                    user_id=user_id,
                    trace_event={"event": "execution_failed", "error_type": type(exc).__name__},
                )
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

    def request(self, *, user_id: str, snapshot: Mapping[str, Any], workflow_version: str, thread_id: str | None = None) -> ApprovalRequest:
        with self.session_factory() as session:
            request = ApprovalRepository(session).create(user_id=user_id, snapshot=snapshot, workflow_version=workflow_version, thread_id=thread_id)
            ApprovalRepository(session).transition(request.id, "PENDING_CONFIRMATION", user_id=user_id)
            session.commit()
            return request

    def confirm(self, *, request_id: str, user_id: str, snapshot_hash: str, workflow_version: str) -> ApprovalRequest:
        with self.session_factory() as session:
            request = ApprovalRepository(session).transition(request_id, "SUBMITTING", user_id=user_id, expected_snapshot_hash=snapshot_hash, expected_workflow_version=workflow_version)
            session.commit()
            return request


def _stable_hash(arguments: Mapping[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
