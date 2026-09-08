from __future__ import annotations

from collections.abc import Callable

from personal_assistant.schemas import HealthStatus


def check_optional(name: str, check: Callable[[], str]) -> HealthStatus:
    try:
        return HealthStatus(name=name, status="ok", detail=check())
    except ImportError as exc:
        return HealthStatus(name=name, status="unavailable", detail=f"driver not installed: {exc.name}")
    except Exception as exc:  # health commands must not crash the CLI
        return HealthStatus(name=name, status="unavailable", detail=f"{type(exc).__name__}: {exc}")
