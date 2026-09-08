from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Route(StrEnum):
    CHAT = "chat"
    MEMORY = "memory"
    DOCUMENT = "document"
    GRAPH = "graph"
    HYBRID = "hybrid"


class SourceKind(StrEnum):
    USER = "USER"
    LLM_INFERENCE = "LLM_INFERENCE"
    TOOL = "TOOL"
    MANUAL = "MANUAL"


class SearchHit(BaseModel):
    id: str
    text: str
    score: float | None = None
    backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    name: str
    status: str
    detail: str


class AssistantState(BaseModel):
    user_id: str
    thread_id: str
    user_message: str
    route: Route = Route.CHAT
    memory_hits: list[SearchHit] = Field(default_factory=list)
    knowledge_hits: list[SearchHit] = Field(default_factory=list)
    graph_hits: list[SearchHit] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    trace: list[str] = Field(default_factory=list)
    loop_step: int = 0
