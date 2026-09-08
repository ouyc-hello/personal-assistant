from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from personal_assistant.settings import Settings
from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import AssistantState, Route


class GraphState(TypedDict, total=False):
    user_id: str
    thread_id: str
    user_message: str
    route: str
    trace: list[str]
    answer: str
    loop_step: int


def _route_node(state: dict[str, Any]) -> dict[str, Any]:
    route = classify_route(state["user_message"])
    return {"route": route.value, "trace": [*state.get("trace", []), f"route:{route.value}"]}


def _retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    route = Route(state["route"])
    backends = {
        Route.MEMORY: ["postgresql+pgvector"],
        Route.DOCUMENT: ["milvus"],
        Route.GRAPH: ["neo4j"],
        Route.HYBRID: ["postgresql+pgvector", "milvus", "neo4j"],
        Route.CHAT: [],
    }[route]
    trace = [*state.get("trace", []), *(f"retrieval:{backend}" for backend in backends)]
    return {"trace": trace}


def _answer_node(state: dict[str, Any]) -> dict[str, Any]:
    route = Route(state["route"])
    answer = (
        "Fake 模式已收到请求。"
        f"\n路由：{route.value}"
        "\n当前只完成路由与检索后端选择，真实数据库/RAG/工具将在后续里程碑接入。"
    )
    return {
        "answer": answer,
        "loop_step": state.get("loop_step", 0) + 1,
        "trace": [*state.get("trace", []), "answer:fake"],
    }


def build_graph(settings: Settings | None = None):
    """Build the minimal LangGraph flow; imports LangGraph only when called."""
    del settings  # reserved for model/checkpointer wiring in M1+
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("langgraph is required; install assistant_app dependencies") from exc

    graph = StateGraph(GraphState)
    graph.add_node("route", _route_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("answer", _answer_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_fake(message: str, user_id: str = "local-user", thread_id: str | None = None) -> AssistantState:
    """Run the deterministic skeleton without LLMs or databases."""
    state: dict[str, Any] = {
        "user_id": user_id,
        "thread_id": thread_id or str(uuid4()),
        "user_message": message,
        "trace": ["run:start"],
    }
    for node in (_route_node, _retrieve_node, _answer_node):
        state.update(node(state))
    return AssistantState.model_validate(state)
