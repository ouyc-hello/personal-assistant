from personal_assistant.agent.graph import run_fake
from personal_assistant.schemas import Route


def test_fake_graph_has_trace() -> None:
    result = run_fake("我住在哪里？", thread_id="thread-test")
    assert result.thread_id == "thread-test"
    assert result.route == Route.MEMORY
    assert "retrieval:postgresql+pgvector" in result.trace
    assert result.answer


def test_fake_graph_populates_backend_hits_from_unified_retriever() -> None:
    from personal_assistant.retrieval import UnifiedRetriever
    from personal_assistant.schemas import SearchHit

    class MemoryBackend:
        def search(self, query: str, *, user_id: str, limit: int = 8):
            del query, limit
            return [SearchHit(
                id="memory-1",
                text="住在上海",
                score=1.0,
                backend="postgresql+pgvector",
                metadata={"user_id": user_id, "status": "PUBLISHED"},
            )]

    retriever = UnifiedRetriever({"postgresql+pgvector": MemoryBackend()})
    result = run_fake("我住在哪里？", user_id="user-1", retriever=retriever)
    assert [hit.id for hit in result.memory_hits] == ["memory-1"]
    assert result.knowledge_hits == []
    assert "retrieval:fused=1" in result.trace
    assert "已检索 1 条" in result.answer
