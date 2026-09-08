from __future__ import annotations

from pathlib import Path
from typing import Any


class UnsupportedDocumentType(ValueError):
    pass


def load_documents(path: str | Path) -> list[Any]:
    """Load Markdown/TXT/PDF through LangChain community loaders."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(source), encoding="utf-8", autodetect_encoding=True).load()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(source)).load()
    raise UnsupportedDocumentType(f"unsupported document type: {suffix or '<none>'}")
