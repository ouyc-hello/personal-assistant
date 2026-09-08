from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from personal_assistant.settings import Settings
from personal_assistant.storage.models import Base


class Database:
    """SQLAlchemy session boundary for the PostgreSQL business store."""

    def __init__(self, url: str | None = None, *, settings: Settings | None = None) -> None:
        database_url = url or (settings or Settings.from_env()).database_url
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_schema_for_dev(self) -> None:
        """Create portable tables for tests/dev; production uses SQL migrations."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
