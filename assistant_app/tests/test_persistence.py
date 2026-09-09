from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from personal_assistant.agent.persistence import (
    LOCAL_TIMEZONE,
    load_chat_run,
    parse_task_intent,
    persist_conversation_turn,
    persist_task_request,
)
from personal_assistant.schemas import AssistantState, Route
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import Message, Task


def test_parse_explicit_interview_time_in_shanghai() -> None:
    intent = parse_task_intent(
        "记住我今天下午4点有一场面试",
        now=datetime(2026, 9, 9, 2, 0, tzinfo=UTC),
    )
    assert intent is not None
    assert intent.title == "面试"
    assert intent.due_at == datetime(2026, 9, 9, 8, 0, tzinfo=UTC)


def test_normal_chat_is_not_converted_to_task() -> None:
    assert parse_task_intent("今天下午4点有什么安排？") is None


def test_task_request_persists_task_and_messages(tmp_path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'assistant.db'}")
    database.create_schema_for_dev()
    database.dispose()
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'assistant.db'}",
        default_user_id="user-1",
    )

    result = persist_task_request(
        settings,
        "记住我明天下午4点有一场面试",
        now=datetime(2026, 9, 9, 2, 0, tzinfo=UTC),
    )

    assert result is not None
    assert result.task.title == "面试"
    assert result.task.status == "SCHEDULED"
    assert result.task.due_at == datetime(2026, 9, 10, 8, 0, tzinfo=UTC)
    assert str(LOCAL_TIMEZONE) == "Asia/Shanghai"

    database = Database(settings=settings)
    with database.session() as session:
        task = session.scalar(select(Task).where(Task.id == result.task.id))
        messages = session.scalars(
            select(Message).where(Message.thread_id == result.thread_id).order_by(Message.event_seq)
        ).all()
        assert task is not None
        assert task.metadata_json["source"] == "USER"
        assert [message.role for message in messages] == ["user", "assistant"]
        assert "写入任务数据库" in messages[-1].content
    database.dispose()


def test_chat_run_trace_and_sources_are_persisted_and_loadable(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'assistant.db'}"
    database = Database(database_url)
    database.create_schema_for_dev()
    database.dispose()
    settings = Settings(database_url=database_url, default_user_id="user-1")
    run = AssistantState(
        user_id="user-1",
        thread_id="run-thread",
        user_message="根据文档说明流程",
        route=Route.DOCUMENT,
        answer="流程见 [S1]。",
        trace=["run:start", "route:document", "retrieval:milvus:raw=1", "answer:llm"],
        loop_step=1,
    )

    thread_id = persist_conversation_turn(
        settings, run.user_message, run.answer, thread_id=run.thread_id, run=run
    )
    assert thread_id == run.thread_id
    loaded = load_chat_run(settings, run.thread_id)
    assert loaded is not None
    assert loaded.route == Route.DOCUMENT
    assert loaded.trace == run.trace
    assert loaded.answer == run.answer
    latest = load_chat_run(settings, None)
    assert latest is not None
    assert latest.thread_id == run.thread_id
