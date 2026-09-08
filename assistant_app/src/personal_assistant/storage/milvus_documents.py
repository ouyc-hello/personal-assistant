from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from personal_assistant.rag.embeddings import EmbeddingsLike
from personal_assistant.rag.models import IndexedChunk
from personal_assistant.schemas import SearchHit
from personal_assistant.settings import Settings


class MilvusDocumentStore:
    """MilvusClient adapter for document chunks; PostgreSQL remains metadata truth."""

    def __init__(self, settings: Settings, embeddings: EmbeddingsLike) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.collection = settings.milvus_collection
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            kwargs: dict[str, Any] = {"uri": settings_uri(self.settings.milvus_uri)}
            if self.settings.milvus_token:
                kwargs["token"] = self.settings.milvus_token
            self._client = MilvusClient(**kwargs)
        return self._client

    def ensure_collection(self) -> None:
        if self.client.has_collection(collection_name=self.collection):
            return
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings.dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="source_uri", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(field_name="version", datatype=DataType.INT64)
        schema.add_field(field_name="metadata_json", datatype=DataType.JSON)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: Sequence[IndexedChunk]) -> None:
        if not chunks:
            return
        self.ensure_collection()
        rows = [
            {
                "id": chunk.id,
                "vector": chunk.vector,
                "text": chunk.text,
                "document_id": chunk.document_id,
                "user_id": chunk.user_id,
                "source_uri": chunk.source_uri,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "version": chunk.version,
                "metadata_json": {**chunk.metadata, "source_uri": chunk.source_uri, "page": chunk.page},
            }
            for chunk in chunks
        ]
        self.client.upsert(collection_name=self.collection, data=rows)

    def search(self, query: str, *, user_id: str, limit: int = 8) -> list[SearchHit]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            data=[self.embeddings.embed_query(query)],
            limit=limit,
            filter=f"user_id == {json.dumps(user_id)}",
            output_fields=["text", "document_id", "user_id", "source_uri", "page", "chunk_index", "version", "metadata_json"],
        )
        hits: list[SearchHit] = []
        for group in results:
            for hit in group:
                entity = hit.get("entity", {})
                metadata = _parse_metadata(entity.get("metadata_json", "{}"))
                metadata.update(
                    {
                        "document_id": entity.get("document_id"),
                        "user_id": entity.get("user_id"),
                        "source_uri": entity.get("source_uri"),
                        "page": entity.get("page", -1),
                        "chunk_index": entity.get("chunk_index"),
                        "version": entity.get("version"),
                    }
                )
                hits.append(
                    SearchHit(
                        id=str(hit.get("id")),
                        text=str(entity.get("text", "")),
                        score=float(hit.get("distance")) if hit.get("distance") is not None else None,
                        backend="milvus",
                        metadata=metadata,
                    )
                )
        return hits

    def delete_document(self, *, document_id: str, user_id: str) -> Any:
        self.ensure_collection()
        expression = f"document_id == {json.dumps(document_id)} and user_id == {json.dumps(user_id)}"
        return self.client.delete(collection_name=self.collection, filter=expression)


def settings_uri(uri: str) -> str:
    """Keep local file URI support explicit for Milvus Lite and server URI support."""
    return uri


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
