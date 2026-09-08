from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from personal_assistant.retrieval import UnifiedRetriever, route_backends
from personal_assistant.settings import Settings
from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import AssistantState, Route, SearchHit


class GraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    user_message: str
    route: str
    memory_hits: list[dict[str, Any]]
    knowledge_hits: list[dict[str, Any]]
    graph_hits: list[dict[str, Any]]
    trace: list[str]
    answer: str
    loop_step: int


def _route_node(state: dict[str, Any]) -> dict[str, Any]:
    route = classify_route(state["user_message"])
    return {"route": route.value, "trace": [*state.get("trace", []), f"route:{route.value}"]}


def _retrieve_node(
    state: dict[str, Any],
    retriever: UnifiedRetriever | None = None,
) -> dict[str, Any]:
    route = Route(state["route"])
    if retriever is None:
        trace = [
            *state.get("trace", []),
            *(f"retrieval:{backend}" for backend in route_backends(route)),
        ]
        return {"trace": trace}

    result = retriever.search(
        state["user_message"],
        user_id=state["user_id"],
        route=route,
    )
    return {
        "memory_hits": _serialize_hits(result.backend_hits.get("postgresql+pgvector", ())),
        "knowledge_hits": _serialize_hits(result.backend_hits.get("milvus", ())),
        "graph_hits": _serialize_hits(result.backend_hits.get("neo4j", ())),
        "trace": [*state.get("trace", []), *result.trace[1:]],
    }


def _answer_node(state: dict[str, Any]) -> dict[str, Any]:
    route = Route(state["route"])
    hit_count = sum(len(state.get(field, [])) for field in ("memory_hits", "knowledge_hits", "graph_hits"))
    answer = (
        "Fake 模式已收到请求。"
        f"\n路由：{route.value}"
        f"\n已检索 {hit_count} 条经过权限过滤的上下文。"
    )
    return {
        "answer": answer,
        "loop_step": state.get("loop_step", 0) + 1,
        "trace": [*state.get("trace", []), "answer:fake"],
    }


def build_graph(settings: Settings | None = None, *, retriever: UnifiedRetriever | None = None):
    """Build the LangGraph retrieval flow with optional real backend adapters."""
    del settings  # reserved for model/checkpointer wiring in later milestones
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("langgraph is required; install assistant_app dependencies") from exc

    graph = StateGraph(GraphState)
    graph.add_node("route", _route_node)
    graph.add_node("retrieve", lambda state: _retrieve_node(state, retriever))
    graph.add_node("answer", _answer_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_fake(
    message: str,
    user_id: str = "local-user",
    thread_id: str | None = None,
    *,
    retriever: UnifiedRetriever | None = None,
) -> AssistantState:
    """Run a deterministic turn, optionally against injected retrieval backends."""
    state: dict[str, Any] = {
        "user_id": user_id,
        "thread_id": thread_id or str(uuid4()),
        "user_message": message,
        "trace": ["run:start"],
    }
    state.update(_route_node(state))
    state.update(_retrieve_node(state, retriever))
    state.update(_answer_node(state))
    return AssistantState.model_validate(state)


def _serialize_hits(hits: Any) -> list[dict[str, Any]]:
    return [hit.model_dump() if isinstance(hit, SearchHit) else dict(hit) for hit in hits]
