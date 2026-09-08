from __future__ import annotations

from personal_assistant.settings import Settings
from personal_assistant.schemas import HealthStatus
from personal_assistant.storage.health import check_optional


def health(settings: Settings) -> HealthStatus:
    def _check() -> str:
        from neo4j import GraphDatabase

        if not settings.neo4j_password:
            raise ValueError("PA_NEO4J_PASSWORD is not configured")
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            driver.verify_connectivity()
            return "entity relationship service reachable"
        finally:
            driver.close()

    return check_optional("neo4j", _check)
