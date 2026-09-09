from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from personal_assistant.agent.persistence import (
    LOCAL_TIMEZONE,
    parse_task_intent,
    persist_task_request,
)
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import Message, Task


def test_parse_explicit_interview_time_in_shanghai() -> None:
    intent = parse_task_intent(
        "记住我今天下午4点有一场面试",
        now=datetime(2026, 9, 9, 2, 0, tzinfo=timezone.utc),
    )
    assert intent is not None
    assert intent.title == "面试"
    assert intent.due_at == datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)


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
        now=datetime(2026, 9, 9, 2, 0, tzinfo=timezone.utc),
    )

    assert result is not None
    assert result.task.title == "面试"
    assert result.task.status == "SCHEDULED"
    assert result.task.due_at == datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
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
