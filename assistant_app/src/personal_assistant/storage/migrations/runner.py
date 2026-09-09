from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import text

from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database

_MIGRATION_TABLE = "pa_schema_migrations"


def _statements(script: str) -> list[str]:
    """Split the project's simple SQL migrations without sending comments to PostgreSQL."""
    clean = re.sub(r"(?m)^\s*--.*(?:\n|$)", "", script)
    return [statement.strip() for statement in clean.split(";") if statement.strip()]


def run_migrations(settings: Settings, *, migrations_dir: Path | None = None) -> list[str]:
    """Apply unapplied PostgreSQL migrations in lexical order, preserving old data."""
    if not settings.database_url.startswith("postgresql"):
        raise ValueError("production migrations require a PostgreSQL PA_DATABASE_URL")
    directory = migrations_dir or Path(__file__).parent
    files = sorted(directory.glob("[0-9][0-9][0-9]_*.sql"))
    database = Database(settings=settings)
    applied: list[str] = []
    try:
        with database.session() as session:
            session.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS pa_schema_migrations "
                    "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
            )
            existing = {
                row[0]
                for row in session.execute(text("SELECT version FROM pa_schema_migrations")).all()
            }
            # A database created before the migration runner already has the
            # business tables. Treat the initial schema as a baseline so the
            # runner never replays 001 against incompatible legacy tables.
            if not existing:
                has_threads = session.execute(
                    text("SELECT to_regclass('public.threads') IS NOT NULL")
                ).scalar_one()
                if has_threads:
                    session.execute(
                        text("INSERT INTO pa_schema_migrations (version) VALUES ('001_initial.sql')")
                    )
                    existing.add("001_initial.sql")
            for path in files:
                if path.name in existing:
                    continue
                for statement in _statements(path.read_text(encoding="utf-8")):
                    session.execute(text(statement))
                session.execute(
                    text("INSERT INTO pa_schema_migrations (version) VALUES (:version)"),
                    {"version": path.name},
                )
                applied.append(path.name)
    finally:
        database.dispose()
    return applied
