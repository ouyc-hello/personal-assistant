from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from personal_assistant.agent.graph import run_fake
from personal_assistant.settings import Settings
from personal_assistant.rag.embeddings import build_embeddings
from personal_assistant.rag.ingest import ingest_file
from personal_assistant.rag.router import classify_route
from personal_assistant.storage.database import Database
from personal_assistant.storage.milvus import health as milvus_health
from personal_assistant.storage.neo4j import health as neo4j_health
from personal_assistant.storage.postgres import health as postgres_health

app = typer.Typer(help="CLI-first personal assistant")
console = Console()


def _settings() -> Settings:
    return Settings.from_env(Path.cwd() / ".env")


def _print_fake_answer(message: str, settings: Settings) -> None:
    result = run_fake(message, user_id=settings.default_user_id)
    console.print(result.answer)
    console.print(f"[dim]thread_id={result.thread_id}[/dim]")
    console.print(f"[dim]trace={' -> '.join(result.trace)}[/dim]")


def _print_real_answer(message: str, settings: Settings) -> None:
    """Call an OpenAI-compatible chat endpoint configured through the environment."""
    if settings.llm_provider not in {"openai", "openai-compatible"}:
        raise typer.BadParameter(
            f"Unsupported PA_LLM_PROVIDER={settings.llm_provider!r}; use fake or openai."
        )
    if not settings.openai_api_key:
        raise typer.BadParameter("PA_OPENAI_API_KEY is required when PA_LLM_PROVIDER=openai.")
    if not settings.llm_model:
        raise typer.BadParameter("PA_LLM_MODEL is required when PA_LLM_PROVIDER=openai.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise typer.BadParameter(
            "langchain-openai is required for real chat; install with: pip install -e '.[openai]'"
        ) from exc

    kwargs = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 1,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    try:
        response = ChatOpenAI(**kwargs).invoke(message)
    except Exception as exc:
        raise typer.BadParameter(
            f"LLM 请求失败（{type(exc).__name__}）：{exc}"
        ) from exc

    content = response.content
    if isinstance(content, str):
        answer = content
    elif isinstance(content, list):
        answer = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        answer = str(content)
    if not answer.strip():
        raise typer.BadParameter("LLM 返回了空内容。")
    console.print(answer)


def _answer(message: str, settings: Settings, *, fake: bool = False) -> None:
    if fake or settings.llm_provider == "fake":
        _print_fake_answer(message, settings)
    else:
        _print_real_answer(message, settings)


@app.command()
def chat(message: str, fake: bool = typer.Option(False, help="Use deterministic Fake mode.")) -> None:
    """Run one assistant turn using .env's LLM provider."""
    _answer(message, _settings(), fake=fake)


@app.command()
def route(message: str) -> None:
    """Show the deterministic first-pass retrieval route."""
    console.print(classify_route(message).value)


@app.command()
def architecture() -> None:
    """Print the storage responsibility split."""
    console.print("PostgreSQL = business truth and transactions")
    console.print("pgvector    = trusted-memory semantic index")
    console.print("Milvus      = document chunk vector retrieval")
    console.print("Neo4j       = entity relationships and Graph RAG")
    console.print("LangChain   = capability components")
    console.print("LangGraph   = execution flow and recovery")


@app.command()
def index(
    path: Path,
    user_id: str | None = typer.Option(None, help="Owner scope for indexed chunks."),
) -> None:
    """Load, split, embed and upsert one Markdown/TXT/PDF file into Milvus."""
    settings = _settings()
    from personal_assistant.storage.milvus_documents import MilvusDocumentStore
    from personal_assistant.storage.repositories import KnowledgeRepository

    embeddings = build_embeddings(settings)
    store = MilvusDocumentStore(settings, embeddings)
    database = Database(settings=settings)
    try:
        with database.session() as session:
            result = ingest_file(
                path,
                user_id=user_id or settings.default_user_id,
                embeddings=embeddings,
                index=store,
                metadata_store=KnowledgeRepository(session),
            )
    finally:
        database.dispose()
    console.print(
        f"indexed {result.chunk_count} chunks from {result.source_uri} "
        f"(document_id={result.document_id}, version={result.version})"
    )


@app.command("db-init")
def db_init() -> None:
    """Create the portable development schema from SQLAlchemy models."""
    database = Database(settings=_settings())
    try:
        database.create_schema_for_dev()
    finally:
        database.dispose()
    console.print("[green]development schema ready[/green]")


@app.command()
def health() -> None:
    """Check configured PostgreSQL/pgvector, Milvus and Neo4j endpoints."""
    settings = _settings()
    statuses = [postgres_health(settings), milvus_health(settings), neo4j_health(settings)]
    table = Table("component", "status", "detail")
    for item in statuses:
        color = "green" if item.status == "ok" else "yellow"
        table.add_row(item.name, f"[{color}]{item.status}[/{color}]", item.detail)
    console.print(table)


@app.command()
def repl(fake: bool = typer.Option(False, help="Use deterministic Fake mode.")) -> None:
    """Start the terminal REPL using .env's LLM provider."""
    settings = _settings()
    mode = "fake mode" if fake or settings.llm_provider == "fake" else f"{settings.llm_provider} mode"
    console.print(f"Personal Assistant ({mode}). 输入 /help 或 /quit。")
    while True:
        try:
            message = typer.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if message.strip() in {"/quit", "/exit"}:
            break
        if message.strip() == "/help":
            console.print("/route <内容>  查看路由\n/quit          退出")
            continue
        if message.startswith("/route "):
            console.print(classify_route(message[7:]).value)
            continue
        try:
            _answer(message, settings, fake=fake)
        except typer.BadParameter as exc:
            console.print(f"[red]{exc}[/red]")


if __name__ == "__main__":
    app()
