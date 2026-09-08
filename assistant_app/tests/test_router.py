from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import Route


def test_memory_route() -> None:
    assert classify_route("请记住我喜欢骑行") == Route.MEMORY


def test_document_route() -> None:
    assert classify_route("根据项目手册总结发布流程") == Route.DOCUMENT


def test_hybrid_route() -> None:
    assert classify_route("根据文档说明我和项目的关系") == Route.HYBRID
