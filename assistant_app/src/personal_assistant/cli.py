from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from personal_assistant.agent.checkpoint import postgres_checkpointer
from personal_assistant.agent.execution import ApprovalService
from personal_assistant.agent.graph import (
    run_approval_resume,
    run_approval_start,
    run_fake,
    run_real,
)
from personal_assistant.agent.persistence import (
    load_chat_run,
    load_conversation_history,
    persist_conversation_turn,
    persist_task_request,
)
from personal_assistant.agent.runtime import (
    build_chat_model,
    build_runtime_retriever,
    close_runtime_retriever,
)
from personal_assistant.agent.tools import build_runtime_tools
from personal_assistant.memory.service import MemoryService
from personal_assistant.rag.embeddings import build_embeddings
from personal_assistant.rag.ingest import ingest_file
from personal_assistant.rag.router import classify_route
from personal_assistant.schemas import AssistantState, Route, SearchHit
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.migrations.runner import run_migrations
from personal_assistant.storage.milvus import health as milvus_health
from personal_assistant.storage.models import MemoryCandidate, MemoryRecord
from personal_assistant.storage.neo4j import health as neo4j_health
from personal_assistant.storage.postgres import health as postgres_health
from personal_assistant.storage.repositories import MemoryRepository

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
    run_thread_id = thread_id or str(uuid4())
    try:
        chat_model = build_chat_model(settings)
        if route.value != "chat":
            retriever = build_runtime_retriever(settings)
        tools = build_runtime_tools(
            settings, user_id=settings.default_user_id, thread_id=run_thread_id
        )
        history = [
            {"role": role, "content": content}
            for role, content in load_conversation_history(settings, run_thread_id)
        ]
        with postgres_checkpointer(settings) as checkpointer:
            result = run_real(
                message,
                user_id=settings.default_user_id,
                thread_id=run_thread_id,
                chat_model=chat_model,
                retriever=retriever,
                tools=tools,
                max_graph_steps=settings.max_graph_steps,
                checkpointer=checkpointer,
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
            settings, message, result.answer, thread_id=result.thread_id, run=result
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
    console.print(f"tool_calls={len(result.tool_calls)}")
    for index, call in enumerate(result.tool_calls, start=1):
        # Deliberately omit tool arguments: debug output may be copied to a
        # terminal log and arguments can contain personal or secret values.
        name = call.get("name", "unknown")
        call_id = call.get("id") or call.get("tool_call_id") or "(no-id)"
        console.print(f"tool[{index}] name={name} call_id={call_id}")
    console.print(f"trace={' -> '.join(result.trace) or '(empty)'}")


def _load_debug_result(thread_id: str | None = None) -> AssistantState | None:
    if _last_result is not None and (thread_id is None or thread_id == _last_result.thread_id):
        return _last_result
    try:
        result = load_chat_run(_settings(), thread_id)
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]读取 PostgreSQL 执行记录失败（{type(exc).__name__}）：{exc}[/red]")
        return None
    if result is None:
        console.print(f"没有找到 thread_id={thread_id} 的执行记录。")
    return result


def _show_last_debug(thread_id: str | None = None) -> None:
    result = _load_debug_result(thread_id)
    if result is not None:
        _print_debug(result)


def _show_last_sources(thread_id: str | None = None) -> None:
    result = _load_debug_result(thread_id)
    if result is not None:
        _print_sources(result.retrieved_hits)


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
def debug(thread_id: str | None = typer.Argument(None, help="Thread ID; omit only when viewing the current process result.")) -> None:
    """Show a route/retrieval/graph trace, loading persisted history by thread ID."""
    _show_last_debug(thread_id)

@app.command()
def sources(thread_id: str | None = typer.Argument(None, help="Thread ID; omit only when viewing the current process result.")) -> None:
    """Show persisted sources by thread ID, or the current process result."""
    _show_last_sources(thread_id)

def _memory_rows(
    settings: Settings, *, include_candidates: bool, limit: int, query: str | None = None
) -> tuple[list[MemoryRecord], list[MemoryCandidate]]:
    database = Database(settings=settings)
    try:
        with database.session() as session:
            repository = MemoryRepository(session)
            embeddings = None
            try:
                if query:
                    embeddings = build_embeddings(settings)
                    hits = repository.search_published(
                        user_id=settings.default_user_id,
                        query_vector=embeddings.embed_query(query),
                        limit=limit,
                    )
                    record_rows = []
                    for hit in hits:
                        record = session.get(MemoryRecord, hit.metadata.get("memory_id"))
                        if record is not None:
                            record_rows.append(record)
                    records = record_rows
                else:
                    records = repository.list_records(user_id=settings.default_user_id, limit=limit)
            finally:
                close = getattr(embeddings, "close", None)
                if callable(close):
                    close()
            candidates = (
                repository.list_candidates(user_id=settings.default_user_id, limit=limit)
                if include_candidates
                else []
            )
            return records, candidates
    finally:
        database.dispose()


def _print_memory_rows(records: list[MemoryRecord], candidates: list[MemoryCandidate]) -> None:
    table = Table("type", "id", "kind", "status", "confidence", "content")
    for record in records:
        table.add_row("record", record.id, record.kind, record.status, f"{record.confidence:.2f}", record.content)
    for candidate in candidates:
        table.add_row("candidate", candidate.id, candidate.kind, candidate.status, f"{candidate.confidence:.2f}", candidate.content)
    if records or candidates:
        console.print(table)
    else:
        console.print("暂无记忆记录。")


def _remember(content: str, settings: Settings, *, kind: str = "context", sensitive: bool = False) -> None:
    embeddings = build_embeddings(settings)
    database = Database(settings=settings)
    try:
        with database.session() as session:
            candidate = MemoryService(session, embeddings).propose(
                user_id=settings.default_user_id,
                kind=kind,
                content=content,
                source="MANUAL",
                confidence=1.0,
                evidence={"source": "manual_cli"},
                sensitive=sensitive,
            )
            console.print(
                f"记忆已处理：status={candidate.status}, id={candidate.id}, kind={candidate.kind}"
            )
    finally:
        close = getattr(embeddings, "close", None)
        if callable(close):
            close()
        database.dispose()


def _forget(memory_id: str, settings: Settings) -> None:
    database = Database(settings=settings)
    try:
        with database.session() as session:
            # Withdrawal changes only PostgreSQL business state; no embedding is needed.
            record = MemoryService(session).withdraw(memory_id)
            console.print(f"已撤回记忆：{record.id}")
    finally:
        database.dispose()



@app.command()
def memories(
    query: str | None = typer.Option(None, "--query", "-q", help="Semantic search text."),
    include_candidates: bool = typer.Option(False, "--include-candidates", help="Include unconfirmed candidates."),
    limit: int = typer.Option(50, min=1, max=500),
) -> None:
    """List trusted memory records and optionally pending candidates."""
    records, candidates = _memory_rows(
        _settings(), include_candidates=include_candidates, limit=limit, query=query
    )
    _print_memory_rows(records, candidates)


@app.command()
def remember(
    content: str,
    kind: str = typer.Option("context", help="Memory kind, for example preference or location."),
    sensitive: bool = typer.Option(False, help="Keep as a candidate instead of auto-publishing."),
) -> None:
    """Create a manually confirmed memory candidate."""
    _remember(content, _settings(), kind=kind, sensitive=sensitive)


@app.command()
def forget(memory_id: str) -> None:
    """Withdraw a published memory within its withdrawal window."""
    try:
        _forget(memory_id, _settings())
    except (LookupError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("approve-start")
def approve_start(
    snapshot_json: str,
    thread_id: str = typer.Option(..., help="Stable LangGraph thread ID."),
) -> None:
    """Create an approval request, pause it, and print the interrupt payload."""
    settings = _settings()
    try:
        snapshot = json.loads(snapshot_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("snapshot_json 必须是合法 JSON") from exc
    if not isinstance(snapshot, dict):
        raise typer.BadParameter("snapshot_json 顶层必须是 JSON object")

    database = Database(settings=settings)
    try:
        service = ApprovalService(database.session_factory)
        request = service.request(
            user_id=settings.default_user_id,
            snapshot=snapshot,
            workflow_version="approval-v1",
            thread_id=thread_id,
        )
        with postgres_checkpointer(settings) as checkpointer:
            if checkpointer is None:
                raise typer.BadParameter("PA_CHECKPOINT_ENABLED=true is required for approval workflows")
            result = run_approval_start(
                thread_id=thread_id,
                user_id=settings.default_user_id,
                approval_request_id=request.id,
                snapshot=request.snapshot,
                checkpointer=checkpointer,
            )
        console.print(json.dumps({"request_id": request.id, "result": result}, ensure_ascii=False, default=str))
    except (LookupError, PermissionError, ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        database.dispose()


@app.command("approve-resume")
def approve_resume(
    thread_id: str = typer.Argument(...),
    request_id: str = typer.Argument(...),
    approved: bool = typer.Option(..., "--approved/--rejected"),
) -> None:
    """Resume a paused approval and then persist the matching decision."""
    settings = _settings()
    database = Database(settings=settings)
    service = ApprovalService(database.session_factory)
    try:
        request = service.get(
            request_id=request_id,
            user_id=settings.default_user_id,
            thread_id=thread_id,
        )
        expected_terminal = "APPROVED" if approved else "REJECTED"
        if request.status == expected_terminal:
            console.print(
                json.dumps(
                    {
                        "request_id": request.id,
                        "approval_status": request.status,
                        "replayed": True,
                    },
                    ensure_ascii=False,
                )
            )
            return
        # Move an approval into SUBMITTING before resuming the checkpoint. If
        # resume fails, UNKNOWN records the split between graph and business
        # state instead of pretending that the action was approved.
        service.begin_resolution(
            request_id=request_id,
            user_id=settings.default_user_id,
            approved=approved,
            thread_id=thread_id,
        )
        try:
            with postgres_checkpointer(settings) as checkpointer:
                if checkpointer is None:
                    raise typer.BadParameter("PA_CHECKPOINT_ENABLED=true is required for approval workflows")
                result = run_approval_resume(
                    thread_id=thread_id,
                    approved=approved,
                    checkpointer=checkpointer,
                )
        except Exception:
            if approved:
                try:
                    service.mark_unknown(
                        request_id=request_id,
                        user_id=settings.default_user_id,
                        thread_id=thread_id,
                    )
                except (LookupError, PermissionError, ValueError, OSError, RuntimeError):
                    pass
            raise
        service.resolve(
            request_id=request_id,
            user_id=settings.default_user_id,
            approved=approved,
            thread_id=thread_id,
        )
        console.print(json.dumps(result, ensure_ascii=False, default=str))
    except (LookupError, PermissionError, ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        database.dispose()


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


@app.command("db-migrate")
def db_migrate() -> None:
    """Apply safe, forward-only PostgreSQL migrations."""
    try:
        applied = run_migrations(_settings())
    except (OSError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if applied:
        console.print("已应用迁移：" + ", ".join(applied))
    else:
        console.print("数据库已是最新迁移版本。")


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
            console.print("/route <内容>  查看路由\n/debug         查看本轮执行信息\n/sources       查看本轮来源\n/memories      查看记忆\n/remember ...  手动记忆\n/forget <id>   撤回记忆\n/quit          退出")
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
        if message.strip() == "/memories":
            try:
                records, candidates = _memory_rows(settings, include_candidates=True, limit=50)
                _print_memory_rows(records, candidates)
            except (OSError, ValueError, RuntimeError) as exc:
                console.print(f"[red]读取记忆失败（{type(exc).__name__}）：{exc}[/red]")
            continue
        if message.startswith("/remember "):
            try:
                _remember(message[len("/remember "):].strip(), settings)
            except (OSError, ValueError, RuntimeError) as exc:
                console.print(f"[red]写入记忆失败（{type(exc).__name__}）：{exc}[/red]")
            continue
        if message.startswith("/forget "):
            try:
                _forget(message[len("/forget "):].strip(), settings)
            except (LookupError, OSError, ValueError, RuntimeError) as exc:
                console.print(f"[red]撤回记忆失败（{type(exc).__name__}）：{exc}[/red]")
            continue
        try:
            thread_id = _answer(message, settings, fake=fake, thread_id=thread_id)
        except typer.BadParameter as exc:
            console.print(f"[red]{exc}[/red]")


if __name__ == "__main__":
    app()
