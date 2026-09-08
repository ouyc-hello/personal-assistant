from __future__ import annotations

from dataclasses import dataclass, field

from personal_assistant.agent.execution import ApprovalService, ToolExecutor, ToolOutcome
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
