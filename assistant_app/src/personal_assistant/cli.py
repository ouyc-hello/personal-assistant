from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from personal_assistant.agent.graph import build_graph, run_fake
from personal_assistant.settings import Settings
from personal_assistant.rag.router import classify_route
from personal_assistant.storage.milvus import health as milvus_health
from personal_assistant.storage.neo4j import health as neo4j_health
from personal_assistant.storage.postgres import health as postgres_health

app = typer.Typer(help="CLI-first personal assistant")
console = Console()


def _settings() -> Settings:
    return Settings.from_env(Path.cwd() / ".env")


@app.command()
def chat(message: str, fake: bool = typer.Option(False, help="Use deterministic Fake mode.")) -> None:
    """Run one assistant turn."""
    settings = _settings()
    if fake or settings.llm_provider == "fake":
        result = run_fake(message, user_id=settings.default_user_id)
        console.print(result.answer)
        console.print(f"[dim]thread_id={result.thread_id}[/dim]")
        console.print(f"[dim]trace={' -> '.join(result.trace)}[/dim]")
        return
    raise typer.BadParameter("Only PA_LLM_PROVIDER=fake is wired in the initial skeleton; use --fake.")


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
def repl() -> None:
    """Start a dependency-light terminal REPL in Fake mode."""
    console.print("Personal Assistant (fake mode). 输入 /help 或 /quit。")
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
        result = run_fake(message)
        console.print(result.answer)


if __name__ == "__main__":
    app()
