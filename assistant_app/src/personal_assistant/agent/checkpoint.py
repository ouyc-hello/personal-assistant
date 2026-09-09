from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from personal_assistant.settings import Settings


@contextmanager
def postgres_checkpointer(settings: Settings) -> Iterator[Any | None]:
    """Yield a LangGraph PostgresSaver when enabled, otherwise no checkpointer.

    The saver owns its connection pool and must stay open for the graph lifetime.
    ``setup()`` creates only LangGraph's checkpoint tables and is safe to run
    repeatedly; business tables remain managed by this application's migrations.
    """
    if not settings.checkpoint_enabled:
        yield None
        return
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "Postgres checkpointing requires langgraph-checkpoint-postgres; "
            "install the assistant_app dependencies"
        ) from exc

    connection_string = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with PostgresSaver.from_conn_string(connection_string) as saver:
        saver.setup()
        yield saver
