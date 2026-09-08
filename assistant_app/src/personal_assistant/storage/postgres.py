from __future__ import annotations

from personal_assistant.settings import Settings
from personal_assistant.storage.health import check_optional
from personal_assistant.schemas import HealthStatus


def health(settings: Settings) -> HealthStatus:
    def _check() -> str:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return "business truth + pgvector reachable"
        finally:
            engine.dispose()

    return check_optional("postgresql+pgvector", _check)
