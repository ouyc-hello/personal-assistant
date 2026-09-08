from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from personal_assistant.rag.embeddings import HashEmbeddings
from personal_assistant.rag.ingest import build_chunks, ingest_file
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import KnowledgeChunk, KnowledgeDocument
from personal_assistant.storage.repositories import KnowledgeRepository


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


class FakeIndex:
    def __init__(self) -> None:
        self.chunks = []

    def upsert(self, chunks) -> None:
        self.chunks.extend(chunks)


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = HashEmbeddings(8)
    first = embeddings.embed_query("same text")
    second = embeddings.embed_query("same text")
    assert first == second
    assert len(first) == 8
    assert round(sum(value * value for value in first), 6) == 1.0


def test_build_chunks_has_stable_ids_and_source_metadata() -> None:
    embeddings = HashEmbeddings(8)
    chunks = build_chunks(
        documents=[FakeDocument("hello", {"page": 2}), FakeDocument("world", {})],
        source_uri="/tmp/example.md",
        user_id="user-1",
        version=3,
        embeddings=embeddings,
        document_id="document-1",
    )
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page for chunk in chunks] == [2, -1]
    assert chunks[0].id != chunks[1].id
    assert all(chunk.document_id == "document-1" for chunk in chunks)


def test_ingest_file_calls_index_and_returns_content_hash(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("one\ntwo", encoding="utf-8")
    index = FakeIndex()

    result = ingest_file(
        source,
        user_id="user-1",
        embeddings=HashEmbeddings(8),
        index=index,
        loader=lambda _: [FakeDocument("one", {}), FakeDocument("two", {})],
        splitter=lambda docs: docs,
    )

    assert result.chunk_count == 2
    assert len(index.chunks) == 2
    assert result.content_hash


def test_ingest_registers_postgres_metadata_projection(tmp_path) -> None:
    source = tmp_path / "note.md"
    source.write_text("one", encoding="utf-8")
    index = FakeIndex()
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema_for_dev()
    try:
        with database.session() as session:
            result = ingest_file(
                source,
                user_id="user-1",
                embeddings=HashEmbeddings(8),
                index=index,
                loader=lambda _: [FakeDocument("one", {"page": 0})],
                splitter=lambda docs: docs,
                metadata_store=KnowledgeRepository(session),
            )
            document = session.get(KnowledgeDocument, result.document_id)
            chunks = session.scalars(select(KnowledgeChunk)).all()
            assert document is not None
            assert document.status == "ACTIVE"
            assert len(chunks) == 1
            assert chunks[0].milvus_id == index.chunks[0].id
    finally:
        database.dispose()
