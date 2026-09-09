from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select, text

from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import Message, Task
from personal_assistant.storage.repositories import TaskRepository, ThreadRepository

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
_TRIGGER_RE = re.compile(r"记住|帮我记|提醒我|备忘|安排", re.IGNORECASE)
_EVENT_RE = re.compile(r"面试|会议|预约|看医生|航班|考试|演讲|约会|截止|提交")
_DATE_RE = re.compile(r"(?:(\d{4})年)?\s*(\d{1,2})月\s*(\d{1,2})日?")
_TIME_RE = re.compile(r"(上午|早上|中午|下午|晚上)?\s*(\d{1,2})(?:[:：点](\d{1,2}))?\s*(?:分)?")


@dataclass(frozen=True)
class TaskIntent:
    """A deliberately conservative task extracted from an explicit user request."""

    title: str
    due_at: datetime | None
    source_text: str


@dataclass(frozen=True)
class PersistedTask:
    task: Task
    thread_id: str
    reply: str


def parse_task_intent(message: str, *, now: datetime | None = None) -> TaskIntent | None:
    """Parse only explicit reminder/task language; do not infer from normal chat."""
    if not _TRIGGER_RE.search(message):
        return None
    event_match = _EVENT_RE.search(message)
    if event_match is None:
        return None

    due_at = _parse_due_at(message, now=now)
    if due_at is None and not any(word in message for word in ("记住", "备忘")):
        return None

    return TaskIntent(
        title=event_match.group(0),
        due_at=due_at,
        source_text=message.strip(),
    )


def persist_task_request(
    settings: Settings,
    message: str,
    *,
    thread_id: str | None = None,
    now: datetime | None = None,
) -> PersistedTask | None:
    """Persist an explicitly requested task and its conversation messages atomically."""
    intent = parse_task_intent(message, now=now)
    if intent is None:
        return None

    database = Database(settings=settings)
    try:
        with database.session() as session:
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                return _persist_task_postgres(
                    session, settings, message, intent, thread_id=thread_id
                )

            threads = ThreadRepository(session)
            thread = threads.get(thread_id) if thread_id else None
            if thread is None:
                thread = threads.create(settings.default_user_id, thread_id=thread_id)
            if thread.user_id != settings.default_user_id:
                raise PermissionError("thread does not belong to the configured user")

            threads.append_message(thread.id, "user", message)
            task = TaskRepository(session).create(
                settings.default_user_id,
                intent.title,
                due_at=intent.due_at,
                metadata={
                    "kind": "appointment" if intent.title in {"面试", "会议", "预约"} else "task",
                    "source": "USER",
                    "source_text": intent.source_text,
                    "timezone": str(LOCAL_TIMEZONE),
                },
            )
            if intent.due_at is not None:
                TaskRepository(session).transition(task.id, "SCHEDULED")

            reply = _task_reply(task, timezone_name=str(LOCAL_TIMEZONE))
            threads.append_message(thread.id, "assistant", reply)
            return PersistedTask(task=task, thread_id=thread.id, reply=reply)
    finally:
        database.dispose()


def persist_conversation_turn(
    settings: Settings,
    user_message: str,
    assistant_message: str,
    *,
    thread_id: str | None = None,
) -> str:
    """Persist a normal user/assistant turn and return its durable thread ID."""
    database = Database(settings=settings)
    try:
        with database.session() as session:
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                return _persist_turn_postgres(
                    session, settings, user_message, assistant_message, thread_id=thread_id
                )

            threads = ThreadRepository(session)
            thread = threads.get(thread_id) if thread_id else None
            if thread is None:
                thread = threads.create(settings.default_user_id, thread_id=thread_id)
            if thread.user_id != settings.default_user_id:
                raise PermissionError("thread does not belong to the configured user")
            threads.append_message(thread.id, "user", user_message)
            threads.append_message(thread.id, "assistant", assistant_message)
            return thread.id
    finally:
        database.dispose()


def load_conversation_history(
    settings: Settings,
    thread_id: str | None,
    *,
    limit: int = 20,
) -> list[tuple[str, str]]:
    """Load prior user/assistant messages for a thread, newest window in order."""
    if not thread_id or limit <= 0:
        return []
    database = Database(settings=settings)
    try:
        with database.session() as session:
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                rows = session.execute(
                    text(
                        "SELECT role, content FROM messages "
                        "WHERE thread_id = CAST(:thread_id AS uuid) "
                        "ORDER BY created_at DESC, id DESC LIMIT :limit"
                    ),
                    {"thread_id": thread_id, "limit": limit},
                ).all()
                return [(str(role), str(content)) for role, content in reversed(rows)]
            rows = session.scalars(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.event_seq.desc())
                .limit(limit)
            ).all()
            return [(message.role, message.content) for message in reversed(rows)]
    finally:
        database.dispose()


def _get_or_create_postgres_thread(session, settings: Settings, thread_id: str | None) -> str:
    selected_id = thread_id or str(uuid4())
    row = session.execute(
        text("SELECT id::text, user_id FROM threads WHERE id = CAST(:thread_id AS uuid)"),
        {"thread_id": selected_id},
    ).one_or_none()
    if row is not None:
        if row.user_id != settings.default_user_id:
            raise PermissionError("thread does not belong to the configured user")
        return str(row.id)

    session.execute(
        text(
            "INSERT INTO threads (id, user_id, title) "
            "VALUES (CAST(:thread_id AS uuid), :user_id, :title)"
        ),
        {"thread_id": selected_id, "user_id": settings.default_user_id, "title": "Personal Assistant"},
    )
    return selected_id


def _append_postgres_message(session, *, thread_id: str, user_id: str, role: str, content: str) -> None:
    session.execute(
        text(
            "INSERT INTO messages (id, thread_id, user_id, role, content, metadata) "
            "VALUES (gen_random_uuid(), CAST(:thread_id AS uuid), :user_id, :role, :content, CAST(:metadata AS jsonb))"
        ),
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "metadata": "{}",
        },
    )


def _audit_postgres(session, *, user_id: str, event_type: str, entity_type: str, entity_id: str) -> None:
    session.execute(
        text(
            "INSERT INTO audit_events "
            "(id, user_id, aggregate_type, aggregate_id, event_type, actor_type, payload) "
            "VALUES (gen_random_uuid(), :user_id, :aggregate_type, :aggregate_id, :event_type, 'SYSTEM', '{}'::jsonb)"
        ),
        {
            "user_id": user_id,
            "aggregate_type": entity_type,
            "aggregate_id": entity_id,
            "event_type": event_type,
        },
    )


def _persist_task_postgres(session, settings: Settings, message: str, intent: TaskIntent, *, thread_id: str | None) -> PersistedTask:
    durable_thread_id = _get_or_create_postgres_thread(session, settings, thread_id)
    _append_postgres_message(
        session, thread_id=durable_thread_id, user_id=settings.default_user_id, role="user", content=message
    )

    task_id = str(uuid4())
    metadata = {
        "kind": "appointment" if intent.title in {"面试", "会议", "预约"} else "task",
        "source": "USER",
        "source_text": intent.source_text,
        "timezone": str(LOCAL_TIMEZONE),
    }
    status = "SCHEDULED" if intent.due_at is not None else "CREATED"
    session.execute(
        text(
            "INSERT INTO tasks (id, user_id, title, status, due_at, metadata, created_at, updated_at) "
            "VALUES (CAST(:task_id AS uuid), :user_id, :title, :status, :due_at, "
            "CAST(:metadata AS json), now(), now())"
        ),
        {
            "task_id": task_id,
            "user_id": settings.default_user_id,
            "title": intent.title,
            "status": status,
            "due_at": intent.due_at,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        },
    )
    _audit_postgres(
        session, user_id=settings.default_user_id, event_type="TASK_CREATED",
        entity_type="task", entity_id=task_id,
    )
    task = Task(
        id=task_id, user_id=settings.default_user_id, title=intent.title, status=status,
        due_at=intent.due_at, metadata_json=metadata,
    )
    reply = _task_reply(task, timezone_name=str(LOCAL_TIMEZONE))
    _append_postgres_message(
        session, thread_id=durable_thread_id, user_id=settings.default_user_id, role="assistant", content=reply
    )
    return PersistedTask(task=task, thread_id=durable_thread_id, reply=reply)


def _persist_turn_postgres(session, settings: Settings, user_message: str, assistant_message: str, *, thread_id: str | None) -> str:
    durable_thread_id = _get_or_create_postgres_thread(session, settings, thread_id)
    _append_postgres_message(
        session, thread_id=durable_thread_id, user_id=settings.default_user_id, role="user", content=user_message
    )
    _append_postgres_message(
        session, thread_id=durable_thread_id, user_id=settings.default_user_id, role="assistant", content=assistant_message
    )
    return durable_thread_id


def _parse_due_at(message: str, *, now: datetime | None) -> datetime | None:
    current = (now or datetime.now(timezone.utc)).astimezone(LOCAL_TIMEZONE)
    target_date = current.date()

    if "后天" in message:
        target_date += timedelta(days=2)
    elif "明天" in message:
        target_date += timedelta(days=1)
    elif "今天" in message:
        target_date = current.date()
    else:
        date_match = _DATE_RE.search(message)
        if date_match:
            year = int(date_match.group(1) or target_date.year)
            month = int(date_match.group(2))
            day = int(date_match.group(3))
            try:
                target_date = target_date.replace(year=year, month=month, day=day)
            except ValueError:
                return None

    time_match = _TIME_RE.search(message)
    if time_match is None:
        return None
    period, raw_hour, raw_minute = time_match.groups()
    hour = int(raw_hour)
    minute = int(raw_minute or 0)
    if minute > 59:
        return None
    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    elif period in {"上午", "早上"} and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=LOCAL_TIMEZONE,
    ).astimezone(timezone.utc)


def _task_reply(task: Task, *, timezone_name: str) -> str:
    if task.due_at is None:
        return f"好的，已将“{task.title}”写入任务数据库。"
    due_local = task.due_at.astimezone(LOCAL_TIMEZONE)
    return (
        f"好的，已将“{task.title}”写入任务数据库："
        f"{due_local:%Y年%-m月%-d日 %H:%M}（{timezone_name}）。"
    )
