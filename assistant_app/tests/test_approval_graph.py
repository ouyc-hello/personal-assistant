from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from personal_assistant.agent.graph import run_approval_resume, run_approval_start


def test_approval_graph_interrupts_then_resumes_and_executes_once() -> None:
    saver = MemorySaver()
    calls: list[dict[str, object]] = []

    def execute(state: dict[str, object]) -> dict[str, object]:
        calls.append(state)
        return {"sent": True}

    first = run_approval_start(
        thread_id="approval-thread",
        user_id="user-1",
        approval_request_id="request-1",
        snapshot={"tool": "notify", "text": "hello"},
        checkpointer=saver,
        execute=execute,
    )
    assert "__interrupt__" in first
    assert calls == []

    second = run_approval_resume(
        thread_id="approval-thread", approved=True, checkpointer=saver, execute=execute
    )
    assert second["approval_status"] == "APPROVED"
    assert second["result"] == {"sent": True}
    assert len(calls) == 1


def test_approval_graph_rejects_without_execution() -> None:
    saver = MemorySaver()
    calls: list[dict[str, object]] = []
    first = run_approval_start(
        thread_id="reject-thread",
        user_id="user-1",
        approval_request_id="request-2",
        snapshot={"tool": "notify"},
        checkpointer=saver,
        execute=lambda state: calls.append(state),
    )
    assert "__interrupt__" in first
    result = run_approval_resume(
        thread_id="reject-thread", approved=False, checkpointer=saver, execute=lambda state: calls.append(state)
    )
    assert result["approval_status"] == "REJECTED"
    assert result["result"] is None
    assert calls == []
