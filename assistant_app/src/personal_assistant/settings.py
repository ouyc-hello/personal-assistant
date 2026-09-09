from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables only."""

    app_env: str = "development"
    llm_provider: str = "fake"
    llm_model: str | None = None
    embedding_provider: str = "fake"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_device: str = "cpu"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_timeout_seconds: float = 60.0
    database_url: str = "postgresql+psycopg://assistant:assistant@localhost:5432/assistant"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "knowledge_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    default_user_id: str = "local-user"
    max_graph_steps: int = 5
    max_tool_attempts: int = 4
    enabled_tools: tuple[str, ...] = ("current_time",)
    checkpoint_enabled: bool = False

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        """Load `.env` opportunistically, without ever requiring one in CI."""
        if env_file is not None:
            try:
                from dotenv import load_dotenv

                load_dotenv(Path(env_file), override=False)
            except ImportError:
                pass
        return cls(
            app_env=os.getenv("PA_APP_ENV", cls.app_env),
            llm_provider=os.getenv("PA_LLM_PROVIDER", cls.llm_provider),
            llm_model=os.getenv("PA_LLM_MODEL") or None,
            embedding_provider=os.getenv("PA_EMBEDDING_PROVIDER", cls.embedding_provider),
            embedding_base_url=os.getenv("PA_EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("PA_EMBEDDING_API_KEY") or None,
            embedding_device=os.getenv("PA_EMBEDDING_DEVICE", cls.embedding_device),
            embedding_model=os.getenv("PA_EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimension=int(os.getenv("PA_EMBEDDING_DIMENSION", str(cls.embedding_dimension))),
            openai_api_key=os.getenv("PA_OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("PA_OPENAI_BASE_URL") or None,
            llm_timeout_seconds=float(os.getenv("PA_LLM_TIMEOUT_SECONDS", str(cls.llm_timeout_seconds))),
            database_url=os.getenv("PA_DATABASE_URL", cls.database_url),
            milvus_uri=os.getenv("PA_MILVUS_URI", cls.milvus_uri),
            milvus_token=os.getenv("PA_MILVUS_TOKEN") or None,
            milvus_collection=os.getenv("PA_MILVUS_COLLECTION", cls.milvus_collection),
            neo4j_uri=os.getenv("PA_NEO4J_URI", cls.neo4j_uri),
            neo4j_user=os.getenv("PA_NEO4J_USER", cls.neo4j_user),
            neo4j_password=os.getenv("PA_NEO4J_PASSWORD") or None,
            neo4j_database=os.getenv("PA_NEO4J_DATABASE", cls.neo4j_database),
            default_user_id=os.getenv("PA_DEFAULT_USER_ID", cls.default_user_id),
            max_graph_steps=int(os.getenv("PA_MAX_GRAPH_STEPS", str(cls.max_graph_steps))),
            max_tool_attempts=int(os.getenv("PA_MAX_TOOL_ATTEMPTS", str(cls.max_tool_attempts))),
            enabled_tools=tuple(
                item.strip()
                for item in os.getenv("PA_ENABLED_TOOLS", ",".join(cls.enabled_tools)).split(",")
                if item.strip()
            ),
            checkpoint_enabled=os.getenv("PA_CHECKPOINT_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        )
