from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

from personal_assistant.settings import Settings


class EmbeddingsLike(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddings:
    """Deterministic offline embeddings for tests and local wiring only."""

    def __init__(self, dimension: int = 32) -> None:
        if dimension < 2:
            raise ValueError("embedding dimension must be at least 2")
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                number = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((number / 2**31) - 1.0)
                if len(values) == self.dimension:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LangChainEmbeddingsAdapter:
    """Expose a LangChain-compatible client plus its configured dimension."""

    def __init__(self, client: object, *, dimension: int) -> None:
        if dimension < 2:
            raise ValueError("embedding dimension must be at least 2")
        self._client = client
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)  # type: ignore[union-attr,no-any-return]

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)  # type: ignore[union-attr,no-any-return]


class LocalBGEEmbeddings:
    """CPU/GPU local BGE embeddings loaded from ``model.safetensors``.

    The checked-in model directory is a Hugging Face Transformer directory,
    but it does not contain the optional ``1_Pooling`` Sentence-Transformers
    module files.  Loading ``AutoModel`` directly is therefore more robust
    than asking SentenceTransformer to reconstruct those modules.  BGE uses
    CLS pooling followed by L2 normalization for dense retrieval.
    """

    def __init__(self, model_path: str, *, device: str, dimension: int) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "Local BGE embeddings require torch and transformers. "
                "Install them with: pip install -e '.[local]'"
            ) from error

        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"local embedding model directory not found: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"local embedding model is missing config.json: {path}")
        if not any((path / filename).is_file() for filename in ("model.safetensors", "pytorch_model.bin")):
            raise FileNotFoundError(
                f"local embedding model is missing model.safetensors/pytorch_model.bin: {path}"
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = AutoModel.from_pretrained(str(path), local_files_only=True)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self.dimension = int(self._model.config.hidden_size)
        if self.dimension != dimension:
            raise ValueError(
                f"embedding dimension mismatch: model={self.dimension}, configured={dimension}"
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            vectors = output.last_hidden_state[:, 0]
            vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def _build_local_embeddings(settings: Settings) -> LocalBGEEmbeddings:
    return LocalBGEEmbeddings(
        settings.embedding_model,
        device=settings.embedding_device,
        dimension=settings.embedding_dimension,
    )


def build_embeddings(settings: Settings) -> EmbeddingsLike:
    if settings.embedding_provider == "fake":
        return HashEmbeddings(settings.embedding_dimension)
    if settings.embedding_provider == "local":
        return _build_local_embeddings(settings)
    if settings.embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as error:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "OpenAI-compatible embeddings require the optional dependencies. "
                "Install them with: pip install -e '.[openai]'"
            ) from error

        kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.embedding_api_key or settings.openai_api_key,
        }
        if settings.embedding_base_url:
            kwargs["base_url"] = settings.embedding_base_url.rstrip("/")
        return LangChainEmbeddingsAdapter(
            OpenAIEmbeddings(**kwargs),
            dimension=settings.embedding_dimension,
        )
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
