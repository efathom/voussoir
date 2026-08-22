"""voussoir run — load a named agent from voussoir.yaml and execute it (Phase 6 B3).

Use this CLI command to invoke a yaml-registered agent and stream its events
to stdout. Token text goes to stdout (inline); tool/delegation/cascade markers
go to stderr so consumers can pipe stdout cleanly. Exit 0 on completed runs;
exit 1 on failures.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from voussoir.agent.agent import Agent
from voussoir.agent.bootstrap import bind_agent_registry
from voussoir.agent.registry import AgentRegistry
from voussoir.container.defaults import default_container


@click.command(name="run")
@click.argument("agent_name")
@click.option(
    "--config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to voussoir.yaml (default: discover via bind_agent_registry order).",
)
@click.option(
    "--input",
    "input_arg",
    default="-",
    show_default=True,
    help="Input text, or '-' to read from stdin.",
)
def run(agent_name: str, config: Path | None, input_arg: str) -> None:
    """Run a registered agent and stream events to stdout."""
    text = sys.stdin.read() if input_arg == "-" else input_arg
    exit_code = asyncio.run(_run(agent_name, config, text))
    if exit_code != 0:
        sys.exit(exit_code)


async def _run(agent_name: str, config: Path | None, text: str) -> int:
    container = default_container()
    bind_agent_registry(container, config_path=config)
    registry = container.resolve(AgentRegistry)

    # Resolve the agent; surface a clear error on miss
    try:
        delegate = registry.get(agent_name)
    except KeyError as exc:
        raise click.ClickException(f"agent {agent_name!r} not found: {exc}") from exc

    # stream() is only available on local Agent instances; IDelegate.delegate()
    # is the generic sub-agent path. For the CLI we require a concrete Agent.
    if not isinstance(delegate, Agent):
        raise click.ClickException(
            f"agent {agent_name!r} is a remote/plugin delegate "
            f"({type(delegate).__name__}); streaming is only supported for "
            "local Agent instances registered in voussoir.yaml."
        )
    agent: Agent = delegate

    finish_reason: str | None = None
    async for ev in agent.stream(text):
        kind = ev.kind
        payload = ev.payload
        if kind == "token":
            # token text -> stdout, no trailing newline so consumers can pipe
            click.echo(payload.get("text", ""), nl=False)
        elif kind == "tool_started":
            click.echo(f"[tool.call: {payload.get('name', '?')}]", err=True)
        elif kind == "delegation_started":
            click.echo(f"[delegate: {payload.get('delegate_name', '?')}]", err=True)
        elif kind == "cascade_passed":
            click.echo("[cascade: PASS]", err=True)
        elif kind == "cascade_failed":
            click.echo("[cascade: FAIL]", err=True)
        elif kind == "done":
            # done_event carries finish_reason as of the audit pass, so a
            # blocked or budget-cut stream now exits non-zero as documented.
            finish_reason = payload.get("finish_reason")
            click.echo("")  # trailing newline
            if finish_reason and finish_reason != "completed":
                click.echo(f"[finish_reason: {finish_reason}]", err=True)

    if finish_reason and finish_reason != "completed":
        return 1
    return 0
