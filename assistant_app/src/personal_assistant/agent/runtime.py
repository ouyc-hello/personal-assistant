from __future__ import annotations

from typing import Any

from personal_assistant.memory.index import PostgresMemoryIndex
from personal_assistant.rag.embeddings import build_embeddings
from personal_assistant.retrieval import UnifiedRetriever
from personal_assistant.settings import Settings
from personal_assistant.storage.milvus_documents import MilvusDocumentStore
from personal_assistant.storage.neo4j_graph import Neo4jGraphStore


def build_chat_model(settings: Settings) -> Any:
    """Build the configured LangChain chat model without connecting in Fake mode."""
    if settings.llm_provider not in {"openai", "openai-compatible"}:
        raise ValueError(
            f"Unsupported PA_LLM_PROVIDER={settings.llm_provider!r}; use fake or openai."
        )
    if not settings.openai_api_key:
        raise ValueError("PA_OPENAI_API_KEY is required when PA_LLM_PROVIDER=openai.")
    if not settings.llm_model:
        raise ValueError("PA_LLM_MODEL is required when PA_LLM_PROVIDER=openai.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ValueError(
            "langchain-openai is required for real chat; install with: "
            "pip install -e '.[openai]'"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 1,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def build_runtime_retriever(settings: Settings) -> UnifiedRetriever:
    """Construct the configured memory, document and graph retrieval adapters.

    Adapters are lazy where possible: creating this object does not connect to
    Milvus or Neo4j until a routed query actually uses them.
    """
    embeddings = build_embeddings(settings)
    memory = PostgresMemoryIndex(settings, embeddings)
    document = MilvusDocumentStore(settings, embeddings)
    graph = Neo4jGraphStore(settings)
    return UnifiedRetriever({
        "postgresql+pgvector": memory,
        "milvus": document,
        "neo4j": graph,
    })


def close_runtime_retriever(retriever: UnifiedRetriever | None) -> None:
    """Release adapters that own database/network resources."""
    if retriever is None:
        return
    for backend in retriever.backends.values():
        close = getattr(backend, "close", None)
        if callable(close):
            close()
