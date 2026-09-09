from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from personal_assistant.agent.graph import run_fake, run_real
from personal_assistant.agent.persistence import (
    load_conversation_history,
    persist_conversation_turn,
    persist_task_request,
)
from personal_assistant.agent.runtime import (
    build_chat_model,
    build_runtime_retriever,
    close_runtime_retriever,
)
from personal_assistant.rag.embeddings import build_embeddings
from personal_assistant.rag.ingest import ingest_file
from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import AssistantState, Route, SearchHit
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.milvus import health as milvus_health
from personal_assistant.storage.neo4j import health as neo4j_health
from personal_assistant.storage.postgres import health as postgres_health

app = typer.Typer(help="CLI-first personal assistant")
console = Console()
_last_result: AssistantState | None = None


def _settings() -> Settings:
    return Settings.from_env(Path.cwd() / ".env")


def _print_fake_answer(message: str, settings: Settings) -> AssistantState:
    global _last_result
    result = run_fake(message, user_id=settings.default_user_id)
    _last_result = result
    console.print(result.answer)
    console.print(f"[dim]thread_id={result.thread_id}[/dim]")
    console.print(f"[dim]trace={' -> '.join(result.trace)}[/dim]")
    return result


def _print_real_answer(
    message: str,
    settings: Settings,
    *,
    thread_id: str | None = None,
) -> str | None:
    """Run the real LangGraph RAG turn and persist it only after success."""
    global _last_result
    try:
        persisted_task = persist_task_request(settings, message, thread_id=thread_id)
    except Exception as exc:
        raise typer.BadParameter(
            f"任务写入 PostgreSQL 失败（{type(exc).__name__}）。请检查数据库连接和表结构。"
        ) from exc
    if persisted_task is not None:
        _last_result = AssistantState(
            user_id=settings.default_user_id,
            thread_id=persisted_task.thread_id,
            user_message=message,
            route=Route.CHAT,
            answer=persisted_task.reply,
            trace=["run:start", "task:persisted", "answer:task"],
            loop_step=1,
        )
        console.print(persisted_task.reply)
        console.print(f"[dim]thread_id={persisted_task.thread_id}[/dim]")
        return persisted_task.thread_id

    route = classify_route(message)
    retriever = None
    try:
        chat_model = build_chat_model(settings)
        if route.value != "chat":
            retriever = build_runtime_retriever(settings)
        history = [
            {"role": role, "content": content}
            for role, content in load_conversation_history(settings, thread_id)
        ]
        result = run_real(
            message,
            user_id=settings.default_user_id,
            thread_id=thread_id,
            chat_model=chat_model,
            retriever=retriever,
            conversation_history=history,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        raise typer.BadParameter(
            f"Agent/RAG 执行失败（{type(exc).__name__}）：{exc}"
        ) from exc
    finally:
        close_runtime_retriever(retriever)

    try:
        durable_thread_id = persist_conversation_turn(
            settings, message, result.answer, thread_id=result.thread_id
        )
    except Exception as exc:
        raise typer.BadParameter(
            f"对话写入 PostgreSQL 失败（{type(exc).__name__}）。未输出未持久化的回答。"
        ) from exc
    _last_result = result
    console.print(result.answer)
    console.print(f"[dim]thread_id={durable_thread_id}[/dim]")
    _print_sources(result.retrieved_hits)
    return durable_thread_id


def _print_sources(hits: list[SearchHit]) -> None:
    if not hits:
        return
    console.print("[dim]来源：[/dim]")
    for index, hit in enumerate(hits, start=1):
        metadata = hit.metadata
        source = metadata.get("source_uri") or metadata.get("source_id") or hit.id
        page = metadata.get("page")
        page_text = f", page={page}" if page is not None and page != -1 else ""
        console.print(
            f"[dim][S{index}] {hit.backend} | {source}{page_text} | id={hit.id}[/dim]"
        )


def _print_debug(result: AssistantState) -> None:
    console.print(f"thread_id={result.thread_id}")
    console.print(f"route={result.route.value}")
    console.print(f"loop_step={result.loop_step}")
    console.print(f"retrieved_hits={len(result.retrieved_hits)}")
    console.print(f"trace={' -> '.join(result.trace) or '(empty)'}")


def _show_last_debug(thread_id: str | None = None) -> None:
    if _last_result is None:
        console.print("当前进程没有可查看的执行记录。请先运行一次 chat/repl。")
        return
    if thread_id and thread_id != _last_result.thread_id:
        console.print(f"当前只保留本进程最近一次 trace；没有 thread_id={thread_id} 的记录。")
        return
    _print_debug(_last_result)


def _show_last_sources(thread_id: str | None = None) -> None:
    if _last_result is None:
        console.print("当前进程没有可查看的来源。请先运行一次 chat/repl。")
        return
    if thread_id and thread_id != _last_result.thread_id:
        console.print(f"当前只保留本进程最近一次来源；没有 thread_id={thread_id} 的记录。")
        return
    _print_sources(_last_result.retrieved_hits)


def _answer(
    message: str,
    settings: Settings,
    *,
    fake: bool = False,
    thread_id: str | None = None,
) -> str | None:
    if fake or settings.llm_provider == "fake":
        result = _print_fake_answer(message, settings)
        return result.thread_id
    return _print_real_answer(message, settings, thread_id=thread_id)


@app.command()
def chat(message: str, fake: bool = typer.Option(False, help="Use deterministic Fake mode.")) -> None:
    """Run one assistant turn using .env's LLM provider."""
    _answer(message, _settings(), fake=fake)


@app.command()
def route(message: str) -> None:
    """Show the deterministic first-pass retrieval route."""
    console.print(classify_route(message).value)


@app.command()
def debug(thread_id: str | None = typer.Argument(None, help="Optional thread_id; only the current process trace is available.")) -> None:
    """Show the most recent in-process route/retrieval/graph trace."""
    _show_last_debug(thread_id)

@app.command()
def sources(thread_id: str | None = typer.Argument(None, help="Optional thread_id; only the current process sources are available.")) -> None:
    """Show sources from the most recent in-process real RAG turn."""
    _show_last_sources(thread_id)

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
    thread_id: str | None = None
    while True:
        try:
            message = typer.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if message.strip() in {"/quit", "/exit"}:
            break
        if message.strip() == "/help":
            console.print("/route <内容>  查看路由\n/debug         查看本轮执行信息\n/sources       查看本轮来源\n/quit          退出")
            continue
        if message.startswith("/route "):
            console.print(classify_route(message[7:]).value)
            continue
        if message.strip() == "/debug":
            _show_last_debug()
            continue
        if message.strip() == "/sources":
            _show_last_sources()
            continue
        try:
            thread_id = _answer(message, settings, fake=fake, thread_id=thread_id)
        except typer.BadParameter as exc:
            console.print(f"[red]{exc}[/red]")


if __name__ == "__main__":
    app()
