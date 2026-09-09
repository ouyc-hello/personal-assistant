from __future__ import annotations

from personal_assistant import cli
from personal_assistant.settings import Settings


def test_repl_handles_missing_openai_key_without_traceback(monkeypatch) -> None:
    messages = iter(["你好", "/quit"])
    output: list[str] = []

    monkeypatch.setattr(
        cli,
        "_settings",
        lambda: Settings(llm_provider="openai", llm_model="test-model"),
    )
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(messages))
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda message="", *args, **kwargs: output.append(str(message)),
    )

    cli.repl(fake=False)

    rendered = "\n".join(output)
    assert "PA_OPENAI_API_KEY is required" in rendered
    assert "Traceback" not in rendered
    assert "AttributeError" not in rendered
