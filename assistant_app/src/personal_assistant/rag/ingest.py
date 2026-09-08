from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import NAMESPACE_URL, uuid5

from personal_assistant.rag.embeddings import EmbeddingsLike
from personal_assistant.rag.loaders import load_documents
from personal_assistant.rag.models import IndexedChunk, IngestionResult
from personal_assistant.rag.chunking import split_documents


class KnowledgeMetadataStore(Protocol):
    def register_document(self, *, document_id: str, user_id: str, source_uri: str,
                          content_hash: str, version: int) -> None: ...

    def register_chunks(self, chunks: list[IndexedChunk]) -> None: ...

    def set_document_status(self, document_id: str, status: str) -> None: ...


def content_hash(documents: list[Any]) -> str:
    value = "\n".join(str(getattr(document, "page_content", "")) for document in documents)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(document: Any) -> dict[str, Any]:
    raw = getattr(document, "metadata", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def build_chunks(*, documents: list[Any], source_uri: str, user_id: str, version: int,
                 embeddings: EmbeddingsLike, document_id: str) -> list[IndexedChunk]:
    texts = [str(getattr(document, "page_content", "")) for document in documents]
    vectors = embeddings.embed_documents(texts)
    chunks: list[IndexedChunk] = []
    for index, (document, text, vector) in enumerate(zip(documents, texts, vectors, strict=True)):
        metadata = _metadata(document)
        page = metadata.get("page", -1)
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = -1
        chunk_id = str(uuid5(NAMESPACE_URL, f"{document_id}:{version}:{index}"))
        chunks.append(
            IndexedChunk(
                id=chunk_id,
                document_id=document_id,
                user_id=user_id,
                text=text,
                vector=vector,
                chunk_index=index,
                version=version,
                source_uri=source_uri,
                page=page,
                metadata=metadata,
            )
        )
    return chunks


def ingest_file(path: str | Path, *, user_id: str, embeddings: EmbeddingsLike,
                index: Any, version: int = 1,
                loader: Callable[[str | Path], list[Any]] = load_documents,
                splitter: Callable[[list[Any]], list[Any]] = split_documents,
                metadata_store: KnowledgeMetadataStore | None = None) -> IngestionResult:
    source = Path(path).expanduser().resolve()
    loaded = loader(source)
    document_hash = content_hash(loaded)
    document_id = str(uuid5(NAMESPACE_URL, f"{user_id}:{source}:{document_hash}"))
    chunks = build_chunks(
        documents=splitter(loaded),
        source_uri=str(source),
        user_id=user_id,
        version=version,
        embeddings=embeddings,
        document_id=document_id,
    )
    if metadata_store is not None:
        metadata_store.register_document(
            document_id=document_id,
            user_id=user_id,
            source_uri=str(source),
            content_hash=document_hash,
            version=version,
        )
    try:
        index.upsert(chunks)
    except Exception:
        if metadata_store is not None:
            metadata_store.set_document_status(document_id, "FAILED")
        raise
    if metadata_store is not None:
        metadata_store.register_chunks(chunks)
        metadata_store.set_document_status(document_id, "ACTIVE")
    return IngestionResult(
        document_id=document_id,
        source_uri=str(source),
        version=version,
        chunk_count=len(chunks),
        content_hash=document_hash,
    )
