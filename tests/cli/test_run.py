"""Locks `voussoir run` end-to-end via the click CliRunner (Phase 6 B3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from voussoir.cli.main import main


def _stub_llm_response(content: str) -> MagicMock:
    """Mock LLM that emits one assistant turn with the given content + end_turn."""
    from ctxforge.protocols.llm import ILLMProvider, LLMResponse

    llm = MagicMock(spec=ILLMProvider)
    llm.name = "anthropic"
    llm.chat = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="stub",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
            raw_response=None,
        )
    )

    # Simple agents use llm.stream for token-by-token streaming
    async def _stream(*args: Any, **kwargs: Any):
        yield content

    llm.stream = _stream
    return llm


def _write_yaml(path: Path, name: str = "ok") -> None:
    path.write_text(
        f"agents:\n  {name}:\n    model: claude-opus-4-7\n    system_prompt: 'be brief'\n",
        encoding="utf-8",
    )


def _patch_container_with_stub_llm(monkeypatch: pytest.MonkeyPatch, stub_llm: MagicMock) -> None:
    """Monkeypatch default_container so the CLI's container has a stub LLM bound.

    v1.0.4 E3: default_container() now freezes ILLMProvider; tests must not
    rebind on a frozen container. Build a fresh Container() directly so the
    stub can be injected without triggering the freeze guard.
    """
    from ctxforge.protocols.llm import ILLMProvider

    from voussoir.a2a.keys import EnvKeyProvider, KeyProvider
    from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer
    from voussoir.auth.protocol import Authorizer
    from voussoir.container import Container
    from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
    from voussoir.observability.sink import ITelemetrySink, NullTelemetrySink
    from voussoir.protocols import IMemoryStore, ISessionStore

    def _patched_default_container() -> Any:
        c = Container()
        c.bind(ILLMProvider, stub_llm)  # type: ignore[type-abstract]
        c.bind(IMemoryStore, InMemoryStore())
        c.bind(ISessionStore, InMemorySessionStore())
        c.bind(ITelemetrySink, NullTelemetrySink())  # type: ignore[type-abstract]
        c.bind(KeyProvider, EnvKeyProvider(allow_ephemeral=True))  # type: ignore[type-abstract]
        c.bind(Authorizer, AllowAllAuthorizer())  # type: ignore[type-abstract]
        return c

    monkeypatch.setattr("voussoir.cli.cmd_run.default_container", _patched_default_container)


def test_run_executes_agent_and_prints_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`voussoir run my_agent --input 'hi'` runs the agent and prints its reply."""
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path / "voussoir.yaml", name="my_agent")
    _patch_container_with_stub_llm(monkeypatch, _stub_llm_response("hello world"))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "my_agent", "--input", "hi"])
    assert result.exit_code == 0, f"output: {result.output}"
    assert "hello world" in result.output


def test_run_unknown_agent_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running a nonexistent agent name exits non-zero with a useful message."""
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path / "voussoir.yaml", name="known")
    _patch_container_with_stub_llm(monkeypatch, _stub_llm_response("ok"))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "nonexistent", "--input", "hi"])
    assert result.exit_code != 0
    # Some message about the unknown agent
    assert (
        "nonexistent" in result.output.lower()
        or "not found" in result.output.lower()
        or "no such" in result.output.lower()
    )


def test_run_no_yaml_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no voussoir.yaml is found (and no --config given), exit 1."""
    monkeypatch.chdir(tmp_path)  # empty dir, no yaml
    # Ensure VOUSSOIR_CONFIG env var isn't pointing somewhere else
    monkeypatch.delenv("VOUSSOIR_CONFIG", raising=False)
    # Patch container to avoid LLM key errors, but no yaml means no registry entries
    _patch_container_with_stub_llm(monkeypatch, _stub_llm_response("ok"))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "anything", "--input", "hi"])
    assert result.exit_code != 0


def test_run_reads_from_stdin_when_input_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`voussoir run agent --input -` reads from stdin."""
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path / "voussoir.yaml", name="agent")
    _patch_container_with_stub_llm(monkeypatch, _stub_llm_response("got it"))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "agent", "--input", "-"], input="hello via stdin\n")
    assert result.exit_code == 0, f"output: {result.output}"
    assert "got it" in result.output
