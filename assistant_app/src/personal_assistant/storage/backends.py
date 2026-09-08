from __future__ import annotations

from dataclasses import dataclass

from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import Route


@dataclass(frozen=True)
class BackendSelection:
    route: Route
    backends: tuple[str, ...]


def select_backends(message: str) -> BackendSelection:
    """Keep backend choice explicit so every retrieval is visible in the trace."""
    route = classify_route(message)
    mapping = {
        Route.CHAT: (),
        Route.MEMORY: ("postgresql+pgvector",),
        Route.DOCUMENT: ("milvus",),
        Route.GRAPH: ("neo4j",),
        Route.HYBRID: ("postgresql+pgvector", "milvus", "neo4j"),
    }
    return BackendSelection(route=route, backends=mapping[route])
