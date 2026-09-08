from __future__ import annotations

import pytest

from personal_assistant.storage.neo4j_graph import (
    GraphEntity,
    GraphRelationship,
    GraphSource,
    InMemoryGraphStore,
)


@pytest.fixture
def graph() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    source = GraphSource("knowledge_document", "doc-1", "/notes/project.md")
    store.upsert(
        [
            GraphEntity("project:alpha", "Alpha", "Project", "user-1", sources=(source,)),
            GraphEntity("person:alice", "Alice", "Person", "user-1", sources=(GraphSource("memory_record", "mem-1"),)),
            GraphEntity("team:platform", "Platform", "Team", "user-1", sources=(source,)),
            GraphEntity("service:api", "API", "Service", "user-1", sources=(source,)),
            GraphEntity("person:bob", "Bob", "Person", "user-2"),
        ],
        [
            GraphRelationship("project:alpha", "person:alice", "owns", "user-1", sources=(source,)),
            GraphRelationship("person:alice", "team:platform", "member_of", "user-1", sources=(source,)),
            GraphRelationship("team:platform", "service:api", "operates", "user-1", sources=(source,)),
        ],
    )
    return store


def test_upsert_is_idempotent_and_keeps_source_backlinks(graph: InMemoryGraphStore) -> None:
    result = graph.upsert(
        [GraphEntity("project:alpha", "Alpha", "Project", "user-1")],
        [GraphRelationship("project:alpha", "person:alice", "owns", "user-1")],
    )
    assert result.entities == 1
    assert result.relationships == 1
    assert len(graph.entities) == 5
    assert len(graph.relationships) == 3
    hit = graph.find_entities("alpha", user_id="user-1")[0]
    assert hit.metadata["source_refs"] == []


def test_entity_lookup_is_scoped_by_user_and_type(graph: InMemoryGraphStore) -> None:
    assert [hit.id for hit in graph.find_entities("bob", user_id="user-1")] == []
    hits = graph.find_entities("ali", user_id="user-1", entity_type="Person")
    assert [hit.id for hit in hits] == ["person:alice"]
    assert hits[0].metadata["postgres_ids"] == ["mem-1"]


def test_multi_hop_neighborhood_and_source_backlink(graph: InMemoryGraphStore) -> None:
    hits = graph.neighborhood("project:alpha", user_id="user-1", max_hops=3)
    assert [hit.id for hit in hits] == ["person:alice", "team:platform", "service:api"]
    assert hits[-1].metadata["hops"] == 3
    assert hits[-1].metadata["path"] == ["project:alpha", "person:alice", "team:platform", "service:api"]
    assert hits[0].metadata["relationship_sources"][0]["source_id"] == "doc-1"


def test_search_returns_direct_entity_then_graph_context(graph: InMemoryGraphStore) -> None:
    hits = graph.search("Alpha", user_id="user-1", limit=4, max_hops=2)
    assert hits[0].id == "project:alpha"
    assert hits[0].metadata["entity_type"] == "Project"
    assert any(hit.id == "person:alice" for hit in hits)
    assert all(hit.backend == "neo4j" for hit in hits)


def test_hop_limit_is_validated(graph: InMemoryGraphStore) -> None:
    with pytest.raises(ValueError):
        graph.neighborhood("project:alpha", user_id="user-1", max_hops=6)
