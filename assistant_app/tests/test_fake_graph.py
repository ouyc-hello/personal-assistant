from personal_assistant.agent.graph import run_fake
from personal_assistant.schemas import Route


def test_fake_graph_has_trace() -> None:
    result = run_fake("我住在哪里？", thread_id="thread-test")
    assert result.thread_id == "thread-test"
    assert result.route == Route.MEMORY
    assert "retrieval:postgresql+pgvector" in result.trace
    assert result.answer
