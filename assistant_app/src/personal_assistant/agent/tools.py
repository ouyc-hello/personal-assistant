from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from personal_assistant.agent.execution import IdempotentTool, ToolExecutor, ToolOutcome
from personal_assistant.settings import Settings
from personal_assistant.storage.database import Database


class CurrentTimeExternalTool(IdempotentTool):
    """Pure local tool used as the first safe tool-calling capability."""

    def execute(self, arguments: dict[str, Any], *, idempotency_key: str) -> ToolOutcome:
        timezone_name = str(arguments.get("timezone") or "Asia/Shanghai")
        try:
            zone = ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
        now = datetime.now(zone)
        result = {
            "timezone": timezone_name,
            "iso": now.isoformat(),
            "display": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return ToolOutcome(
            external_id=f"local-time:{timezone_name}:{now.isoformat()}",
            result=result,
        )

    def reconcile(self, *, idempotency_key: str) -> ToolOutcome | None:
        # The local clock has no remote operation to reconcile. A crashed run is
        # retried with its original idempotency key and gets a fresh reading.
        return None


def _tool_result(run: Any) -> str:
    for event in reversed(run.trace):
        if event.get("event") == "external_success":
            return json.dumps(event.get("result", {}), ensure_ascii=False)
    raise RuntimeError(f"tool did not complete successfully: {run.status}")


def build_runtime_tools(
    settings: Settings,
    *,
    user_id: str,
    thread_id: str | None = None,
) -> dict[str, BaseTool]:
    """Build safe LangChain tools backed by durable ``ToolRun`` records.

    Database engines are created inside each invocation and disposed in a
    ``finally`` block. This keeps a CLI process and a LangGraph checkpoint from
    retaining a PostgreSQL pool just because a model never called a tool.
    ``InjectedToolCallId`` makes the provider's call ID the idempotency scope,
    so two legitimate time queries are not accidentally cached forever.
    """

    if "current_time" not in settings.enabled_tools:
        return {}

    @tool("current_time")
    def current_time(
        tool_call_id: Annotated[str, InjectedToolCallId],
        timezone: str = "Asia/Shanghai",
    ) -> str:
        """Get the current local time. This tool has no external side effects."""
        call_id = tool_call_id or str(uuid4())
        idempotency_key = f"{thread_id or 'adhoc'}:current_time:{call_id}"
        database = Database(settings=settings)
        try:
            executor = ToolExecutor(
                database.session_factory,
                {"current_time": CurrentTimeExternalTool()},
                max_attempts=settings.max_tool_attempts,
            )
            run = executor.execute(
                user_id=user_id,
                tool_name="current_time",
                arguments={"timezone": timezone},
                idempotency_key=idempotency_key,
                thread_id=thread_id,
                invocation_id=call_id,
            )
            return _tool_result(run)
        finally:
            database.dispose()

    return {"current_time": current_time}
