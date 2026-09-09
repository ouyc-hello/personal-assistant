from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from personal_assistant.agent.graph import run_real
from personal_assistant.agent.tools import build_runtime_tools
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import ToolRun


class ToolCallingModel:
    def __init__(self) -> None:
        self.bound_tools = []
        self.inputs = []
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "current_time",
                        "args": {"timezone": "UTC"},
                        "id": "call-time-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="现在是工具返回的 UTC 时间。"),
        ]

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        return self.responses.pop(0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'assistant.db'}")


def test_real_graph_executes_tool_then_returns_final_answer(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings=settings)
    database.create_schema_for_dev()
    database.dispose()
    model = ToolCallingModel()
    tools = build_runtime_tools(settings, user_id="user-1", thread_id="thread-1")

    result = run_real(
        "现在几点？",
        user_id="user-1",
        thread_id="thread-1",
        chat_model=model,
        tools=tools,
        max_graph_steps=3,
    )

    assert result.answer == "现在是工具返回的 UTC 时间。"
    assert result.loop_step == 2
    assert result.tool_calls[0]["id"] == "call-time-1"
    assert "answer:tool_calls:1" in result.trace
    assert "answer:llm" in result.trace
    assert len(model.inputs) == 2

    database = Database(settings=settings)
    try:
        with database.session() as session:
            runs = session.query(ToolRun).all()
            assert len(runs) == 1
            assert runs[0].status == "COMPLETED"
            assert runs[0].invocation_id == "call-time-1"
    finally:
        database.dispose()


def test_current_time_tool_reuses_call_id_without_duplicate_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings=settings)
    database.create_schema_for_dev()
    database.dispose()
    tool = build_runtime_tools(settings, user_id="user-1", thread_id="thread-1")["current_time"]
    call = {"name": "current_time", "args": {"timezone": "UTC"}, "id": "call-same", "type": "tool_call"}

    first = tool.invoke(call)
    second = tool.invoke(call)

    assert first == second
    database = Database(settings=settings)
    try:
        with database.session() as session:
            assert session.query(ToolRun).count() == 1
    finally:
        database.dispose()


def test_disabled_tools_are_not_exposed_to_the_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = Settings(
        database_url=settings.database_url,
        enabled_tools=(),
    )
    assert build_runtime_tools(settings, user_id="user-1") == {}
