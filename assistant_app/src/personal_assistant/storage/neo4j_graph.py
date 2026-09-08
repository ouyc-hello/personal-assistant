from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from personal_assistant.schemas import SearchHit
from personal_assistant.settings import Settings


@dataclass(frozen=True)
class GraphSource:
    """A source-system reference retained on a graph entity or relationship."""

    source_type: str
    source_id: str
    source_uri: str | None = None

    def as_dict(self) -> dict[str, str]:
        value = {"source_type": self.source_type, "source_id": self.source_id}
        if self.source_uri:
            value["source_uri"] = self.source_uri
        return value


@dataclass(frozen=True)
class GraphEntity:
    """An idempotently addressable entity owned by one user."""

    key: str
    name: str
    entity_type: str
    user_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    sources: tuple[GraphSource, ...] = ()


@dataclass(frozen=True)
class GraphRelationship:
    """A typed edge represented as a stable relation property on RELATED_TO."""

    source_key: str
    target_key: str
    relation: str
    user_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    sources: tuple[GraphSource, ...] = ()


@dataclass(frozen=True)
class GraphUpsertResult:
    entities: int
    relationships: int


def _sources_json(sources: Iterable[GraphSource]) -> str:
    return json.dumps([source.as_dict() for source in sources], ensure_ascii=False, sort_keys=True)


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_sources(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [dict(item) for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    return []


def _hit_metadata(*, entity: GraphEntity, hops: int = 0, path: list[str] | None = None) -> dict[str, Any]:
    source_refs = [source.as_dict() for source in entity.sources]
    metadata: dict[str, Any] = {
        "entity_key": entity.key,
        "entity_type": entity.entity_type,
        "name": entity.name,
        "properties": dict(entity.properties),
        "source_refs": source_refs,
        "postgres_ids": [
            source.source_id
            for source in entity.sources
            if source.source_type.startswith("postgres") or source.source_type in {"memory_record", "knowledge_document", "task"}
        ],
    }
    if hops:
        metadata["hops"] = hops
    if path:
        metadata["path"] = path
    return metadata


class InMemoryGraphStore:
    """Deterministic graph backend for tests and offline CLI development."""

    backend_name = "neo4j"

    def __init__(self) -> None:
        self.entities: dict[tuple[str, str], GraphEntity] = {}
        self.relationships: dict[tuple[str, str, str, str], GraphRelationship] = {}

    def upsert(
        self,
        entities: Iterable[GraphEntity],
        relationships: Iterable[GraphRelationship],
    ) -> GraphUpsertResult:
        entity_count = 0
        for entity in entities:
            self.entities[(entity.user_id, entity.key)] = entity
            entity_count += 1
        relationship_count = 0
        for relationship in relationships:
            if (relationship.user_id, relationship.source_key) not in self.entities:
                raise LookupError(f"graph source entity not found: {relationship.source_key}")
            if (relationship.user_id, relationship.target_key) not in self.entities:
                raise LookupError(f"graph target entity not found: {relationship.target_key}")
            key = (relationship.user_id, relationship.source_key, relationship.relation, relationship.target_key)
            self.relationships[key] = relationship
            relationship_count += 1
        return GraphUpsertResult(entities=entity_count, relationships=relationship_count)

    def find_entities(
        self,
        name: str,
        *,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 8,
    ) -> list[SearchHit]:
        if limit <= 0:
            return []
        needle = name.strip().casefold()
        matches = [
            entity
            for (owner, _), entity in self.entities.items()
            if owner == user_id
            and (not entity_type or entity.entity_type == entity_type)
            and (needle in entity.name.casefold() or needle in entity.key.casefold())
        ]
        matches.sort(key=lambda entity: (entity.name.casefold(), entity.key))
        return [
            SearchHit(
                id=entity.key,
                text=f"{entity.entity_type}: {entity.name}",
                score=1.0,
                backend=self.backend_name,
                metadata=_hit_metadata(entity=entity),
            )
            for entity in matches[:limit]
        ]

    def neighborhood(
        self,
        entity_key: str,
        *,
        user_id: str,
        max_hops: int = 2,
        limit: int = 8,
    ) -> list[SearchHit]:
        if limit <= 0:
            return []
        max_hops = _validate_hops(max_hops)
        start = self.entities.get((user_id, entity_key))
        if start is None:
            return []
        adjacency: dict[str, list[tuple[str, GraphRelationship]]] = {}
        for relationship in self.relationships.values():
            if relationship.user_id != user_id:
                continue
            adjacency.setdefault(relationship.source_key, []).append((relationship.target_key, relationship))
            adjacency.setdefault(relationship.target_key, []).append((relationship.source_key, relationship))
        queue: deque[tuple[str, int, list[str]]] = deque([(start.key, 0, [start.key])])
        seen = {start.key}
        results: list[SearchHit] = []
        while queue and len(results) < limit:
            current, hops, path = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbor_key, relationship in sorted(adjacency.get(current, []), key=lambda item: item[0]):
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)
                neighbor = self.entities[(user_id, neighbor_key)]
                neighbor_path = [*path, neighbor_key]
                results.append(
                    SearchHit(
                        id=neighbor.key,
                        text=f"{start.name} -[{relationship.relation}]- {neighbor.name}",
                        score=1.0 / (hops + 2),
                        backend=self.backend_name,
                        metadata={
                            **_hit_metadata(entity=neighbor, hops=hops + 1, path=neighbor_path),
                            "relation": relationship.relation,
                            "relationship_sources": [source.as_dict() for source in relationship.sources],
                        },
                    )
                )
                queue.append((neighbor_key, hops + 1, neighbor_path))
                if len(results) >= limit:
                    break
        return results

    def search(self, query: str, *, user_id: str, limit: int = 8, max_hops: int = 2) -> list[SearchHit]:
        if limit <= 0:
            return []
        direct = self.find_entities(query, user_id=user_id, limit=limit)
        hits: list[SearchHit] = list(direct)
        seen = {hit.id for hit in hits}
        for entity_hit in direct:
            for hit in self.neighborhood(entity_hit.id, user_id=user_id, max_hops=max_hops, limit=limit):
                if hit.id not in seen:
                    hits.append(hit)
                    seen.add(hit.id)
                if len(hits) >= limit:
                    return hits
        return hits


class Neo4jGraphStore:
    """Neo4j adapter for explainable entities, relationships and multi-hop retrieval."""

    backend_name = "neo4j"

    def __init__(self, settings: Settings, *, driver: Any | None = None) -> None:
        self.settings = settings
        self._driver = driver

    @property
    def driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            if not self.settings.neo4j_password:
                raise ValueError("PA_NEO4J_PASSWORD is not configured")
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def upsert(
        self,
        entities: Iterable[GraphEntity],
        relationships: Iterable[GraphRelationship],
    ) -> GraphUpsertResult:
        entity_list = list(entities)
        relationship_list = list(relationships)
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for entity in entity_list:
                session.run(
                    """
                    MERGE (n:Entity {user_id: $user_id, key: $key})
                    SET n.name = $name,
                        n.entity_type = $entity_type,
                        n.properties_json = $properties_json,
                        n.source_refs_json = $source_refs_json
                    """,
                    user_id=entity.user_id,
                    key=entity.key,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    properties_json=json.dumps(entity.properties, ensure_ascii=False, sort_keys=True),
                    source_refs_json=_sources_json(entity.sources),
                ).consume()
            for relationship in relationship_list:
                session.run(
                    """
                    MATCH (source:Entity {user_id: $user_id, key: $source_key})
                    MATCH (target:Entity {user_id: $user_id, key: $target_key})
                    MERGE (source)-[r:RELATED_TO {relation: $relation}]->(target)
                    SET r.properties_json = $properties_json,
                        r.source_refs_json = $source_refs_json
                    """,
                    user_id=relationship.user_id,
                    source_key=relationship.source_key,
                    target_key=relationship.target_key,
                    relation=relationship.relation,
                    properties_json=json.dumps(relationship.properties, ensure_ascii=False, sort_keys=True),
                    source_refs_json=_sources_json(relationship.sources),
                ).consume()
        return GraphUpsertResult(entities=len(entity_list), relationships=len(relationship_list))

    def find_entities(
        self,
        name: str,
        *,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 8,
    ) -> list[SearchHit]:
        if limit <= 0:
            return []
        query = """
        MATCH (n:Entity {user_id: $user_id})
        WHERE (toLower(n.name) CONTAINS toLower($needle) OR toLower(n.key) CONTAINS toLower($needle))
          AND ($entity_type IS NULL OR n.entity_type = $entity_type)
        RETURN n.key AS key, n.name AS name, n.entity_type AS entity_type,
               n.properties_json AS properties_json, n.source_refs_json AS source_refs_json
        ORDER BY n.name, n.key
        LIMIT $limit
        """
        with self.driver.session(database=self.settings.neo4j_database) as session:
            rows = session.run(query, user_id=user_id, needle=name.strip(), entity_type=entity_type, limit=limit)
            return [self._entity_hit(row) for row in rows]

    def neighborhood(
        self,
        entity_key: str,
        *,
        user_id: str,
        max_hops: int = 2,
        limit: int = 8,
    ) -> list[SearchHit]:
        if limit <= 0:
            return []
        max_hops = _validate_hops(max_hops)
        query = f"""
        MATCH (start:Entity {{user_id: $user_id, key: $entity_key}})
        MATCH p=(start)-[:RELATED_TO*1..{max_hops}]-(neighbor:Entity {{user_id: $user_id}})
        WITH start, neighbor, min(length(p)) AS hops
        RETURN start.name AS start_name, neighbor.key AS key, neighbor.name AS name,
               neighbor.entity_type AS entity_type, neighbor.properties_json AS properties_json,
               neighbor.source_refs_json AS source_refs_json, hops
        ORDER BY hops, neighbor.name, neighbor.key
        LIMIT $limit
        """
        with self.driver.session(database=self.settings.neo4j_database) as session:
            rows = session.run(query, user_id=user_id, entity_key=entity_key, limit=limit)
            hits: list[SearchHit] = []
            for row in rows:
                entity = self._entity_from_row(row)
                hops = int(row["hops"])
                hits.append(
                    SearchHit(
                        id=entity.key,
                        text=f"{row['start_name']} -> {entity.name}",
                        score=1.0 / (hops + 1),
                        backend=self.backend_name,
                        metadata=_hit_metadata(entity=entity, hops=hops),
                    )
                )
            return hits

    def search(self, query: str, *, user_id: str, limit: int = 8, max_hops: int = 2) -> list[SearchHit]:
        direct = self.find_entities(query, user_id=user_id, limit=limit)
        hits = list(direct)
        seen = {hit.id for hit in hits}
        for entity_hit in direct:
            for hit in self.neighborhood(entity_hit.id, user_id=user_id, max_hops=max_hops, limit=limit):
                if hit.id not in seen:
                    hits.append(hit)
                    seen.add(hit.id)
                if len(hits) >= limit:
                    return hits
        return hits

    @staticmethod
    def _entity_from_row(row: Any) -> GraphEntity:
        return GraphEntity(
            key=str(row["key"]),
            name=str(row["name"]),
            entity_type=str(row["entity_type"]),
            user_id=str(row["user_id"]) if row["user_id"] is not None else "",
            properties=_parse_json(row["properties_json"]),
            sources=tuple(
                GraphSource(
                    source_type=str(item.get("source_type", "unknown")),
                    source_id=str(item.get("source_id", "")),
                    source_uri=item.get("source_uri"),
                )
                for item in _parse_sources(row["source_refs_json"])
            ),
        )

    @classmethod
    def _entity_hit(cls, row: Any) -> SearchHit:
        entity = cls._entity_from_row(row)
        return SearchHit(
            id=entity.key,
            text=f"{entity.entity_type}: {entity.name}",
            score=1.0,
            backend=cls.backend_name,
            metadata=_hit_metadata(entity=entity),
        )


def _validate_hops(max_hops: int) -> int:
    if not 1 <= max_hops <= 5:
        raise ValueError("max_hops must be between 1 and 5")
    return max_hops
