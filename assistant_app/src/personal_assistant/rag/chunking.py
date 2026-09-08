from __future__ import annotations

from typing import Any


def split_documents(documents: list[Any], *, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Any]:
    """Use LangChain's recursive splitter while keeping source metadata intact."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
