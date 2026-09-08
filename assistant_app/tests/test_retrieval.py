from __future__ import annotations

from dataclasses import dataclass

import pytest

from personal_assistant.retrieval import UnifiedRetriever, reciprocal_rank_fusion
from personal_assistant.schemas import Route, SearchHit


@dataclass
class FakeBackend:
    hits: list[SearchHit]
    error: Exception | None = None
    calls: int = 0

    def search(self, query: str, *, user_id: str, limit: int = 8) -> list[SearchHit]:
        del query, user_id
        self.calls += 1
        if self.error:
            raise self.error
        return self.hits[:limit]


def hit(backend: str, item_id: str, score: float, **metadata: object) -> SearchHit:
    return SearchHit(id=item_id, text=item_id, score=score, backend=backend, metadata=metadata)


def test_memory_route_only_queries_memory_backend() -> None:
    memory = FakeBackend([hit("postgresql+pgvector", "m1", 0.9)])
    document = FakeBackend([hit("milvus", "d1", 0.9)])
    graph = FakeBackend([hit("neo4j", "g1", 0.9)])
    result = UnifiedRetriever({
        "postgresql+pgvector": memory,
        "milvus": document,
        "neo4j": graph,
    }).search("我住在哪里", user_id="user-1")
    assert result.route == Route.MEMORY
    assert [item.id for item in result.hits] == ["m1"]
    assert memory.calls == 1 and document.calls == 0 and graph.calls == 0
    assert "retrieval:postgresql+pgvector:raw=1:allowed=1:denied=0" in result.trace


def test_hybrid_fuses_deduplicates_and_applies_permission_filter() -> None:
    memory = FakeBackend([
        hit("postgresql+pgvector", "same", 0.95, canonical_id="source-1", user_id="user-1"),
        hit("postgresql+pgvector", "private", 0.8, user_id="other"),
    ])
    document = FakeBackend([
        hit("milvus", "same-doc", 0.92, canonical_id="source-1", user_id="user-1"),
        hit("milvus", "doc-2", 0.7, user_id="user-1"),
    ])
    graph = FakeBackend([hit("neo4j", "entity-1", 1.0, user_id="user-1")])
    result = UnifiedRetriever(
        {"postgresql+pgvector": memory, "milvus": document, "neo4j": graph},
        backend_weights={"neo4j": 2.0},
    ).search("根据文档说明我的关系", user_id="user-1", limit=4)
    assert result.route == Route.HYBRID
    assert len(result.hits) == 3
    merged = next(item for item in result.hits if item.metadata.get("canonical_id") == "source-1")
    assert merged.metadata["retrieved_from"] == ["postgresql+pgvector", "milvus"]
    assert merged.metadata["fusion_contributions"]["milvus"] > 0
    assert all(item.id != "private" for item in result.hits)
    assert any("denied=1" in line for line in result.trace)


def test_backend_failure_is_traced_and_other_backend_survives() -> None:
    memory = FakeBackend([], error=RuntimeError("offline"))
    document = FakeBackend([hit("milvus", "d1", 0.6, user_id="user-1")])
    graph = FakeBackend([])
    result = UnifiedRetriever({
        "postgresql+pgvector": memory,
        "milvus": document,
        "neo4j": graph,
    }).search("根据资料说明我的关系", user_id="user-1")
    assert [item.id for item in result.hits] == ["d1"]
    assert "retrieval:postgresql+pgvector:error=RuntimeError" in result.trace


def test_rrf_is_deterministic_and_keeps_source_provenance() -> None:
    fused = reciprocal_rank_fusion({
        "a": [hit("a", "one", 1.0, canonical_id="same")],
        "b": [hit("b", "two", 1.0, canonical_id="same"), hit("b", "other", 1.0)],
    }, rrf_k=1)
    assert len(fused) == 2
    assert fused[0].hit.metadata["retrieved_from"] == ["a", "b"]
    assert fused[0].fusion_score > fused[1].fusion_score


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        UnifiedRetriever({}, rrf_k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, rrf_k=0)
