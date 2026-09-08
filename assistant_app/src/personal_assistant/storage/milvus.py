from __future__ import annotations

from personal_assistant.settings import Settings
from personal_assistant.schemas import HealthStatus
from personal_assistant.storage.health import check_optional


def health(settings: Settings) -> HealthStatus:
    def _check() -> str:
        from pymilvus import MilvusClient

        kwargs = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        client = MilvusClient(**kwargs)
        collections = client.list_collections()
        return f"document vector service reachable; collections={len(collections)}"

    return check_optional("milvus", _check)
