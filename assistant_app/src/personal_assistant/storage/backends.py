from __future__ import annotations

from dataclasses import dataclass

from personal_assistant.retrieval import route_backends
from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import Route


@dataclass(frozen=True)
class BackendSelection:
    route: Route
    backends: tuple[str, ...]


def select_backends(message: str) -> BackendSelection:
    """Keep backend choice explicit so every retrieval is visible in the trace."""
    route = classify_route(message)
    backends = route_backends(route)
    return BackendSelection(route=route, backends=backends)
