from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from personal_assistant.agent.graph import run_real
from personal_assistant.retrieval import UnifiedRetriever
from personal_assistant.schemas import SearchHit


@dataclass
class FakeResponse:
    content: str


class FakeChatModel:
    def __init__(self) -> None:
        self.inputs: list[Any] = []

    def invoke(self, input: Any) -> FakeResponse:
        self.inputs.append(input)
        return FakeResponse("根据 [S1]，采购流程需要先提交申请。")


class FakeBackend:
    def search(self, query: str, *, user_id: str, limit: int = 8) -> list[SearchHit]:
        assert query == "根据文档说明采购流程"
        assert user_id == "user-1"
        return [
            SearchHit(
                id="chunk-1",
                text="采购流程需要先提交申请。",
                score=0.91,
                backend="milvus",
                metadata={"source_uri": "/notes/procurement.md", "page": 2, "user_id": user_id},
            )
        ][:limit]


def test_real_graph_routes_retrieves_and_injects_context() -> None:
    model = FakeChatModel()
    retriever = UnifiedRetriever({"milvus": FakeBackend()})

    result = run_real(
        "根据文档说明采购流程",
        user_id="user-1",
        chat_model=model,
        retriever=retriever,
    )

    assert result.answer == "根据 [S1]，采购流程需要先提交申请。"
    assert result.route.value == "document"
    assert [hit.id for hit in result.retrieved_hits] == ["chunk-1"]
    assert "route:document" in result.trace
    assert "retrieval:milvus:raw=1:allowed=1:denied=0" in result.trace
    assert "answer:llm" in result.trace
    assert len(model.inputs) == 1
    messages = model.inputs[0]
    assert "采购流程需要先提交申请" in messages[0].content
    assert "[S1]" in messages[0].content
    assert messages[1].content == "根据文档说明采购流程"


def test_real_graph_without_retriever_still_answers_chat() -> None:
    model = FakeChatModel()

    result = run_real("你好", user_id="user-1", chat_model=model)

    assert result.route.value == "chat"
    assert result.retrieved_hits == []
    assert result.answer
    assert "answer:llm" in result.trace


def test_checkpointed_turn_replaces_previous_message_state() -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    class TwoTurnModel:
        def __init__(self) -> None:
            self.inputs = []
            self.responses = [AIMessage(content="第一轮"), AIMessage(content="第二轮")]

        def invoke(self, messages):
            self.inputs.append(messages)
            return self.responses.pop(0)

    model = TwoTurnModel()
    saver = MemorySaver()
    first = run_real(
        "根据文档说明采购流程",
        user_id="user-1",
        thread_id="same-thread",
        chat_model=model,
        retriever=UnifiedRetriever({"milvus": FakeBackend()}),
        checkpointer=saver,
    )
    second = run_real(
        "你好",
        user_id="user-1",
        thread_id="same-thread",
        chat_model=model,
        checkpointer=saver,
        conversation_history=[],
    )

    assert first.answer == "第一轮"
    assert second.answer == "第二轮"
    assert second.retrieved_hits == []
    assert [message.content for message in model.inputs[1] if hasattr(message, "content")] == [
        model.inputs[1][0].content,
        "你好",
    ]
    assert len(model.inputs[1]) == 2
