# Contributing to voussoir

Thanks for your interest in contributing! This guide covers how to set up a
development environment, run the test suite, and submit a change.

## Development environment

- **Python 3.12+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A sibling checkout of [ctxforge](https://github.com/efathom/ctxforge) (voussoir
  depends on it via an editable path source)

```bash
git clone https://github.com/efathom/voussoir.git
git clone https://github.com/efathom/ctxforge.git ../ctxforge
cd voussoir
uv sync --extra dev --extra a2a --extra mcp --extra mem-sqlite
```

## Build & test

```bash
uv run pytest -q            # unit + contract tests (mock providers, no network)
uv run ruff check src tests # lint
uv run black --check src tests
uv run mypy src/voussoir    # type check
make ci                     # lint + typecheck + test in one
```

The suite is safe to run without any API key: non-live tests use a dummy key
via an autouse fixture, and live tests skip when no real `ANTHROPIC_API_KEY` is
present.

## Code style

- Follow the repo's `[tool.ruff]` config (line length 100); `black` handles
  formatting.
- Use `structlog` keyword-arg logging (`get_logger`) — never `print`.
- Keep extension points (`Tool`, `Guardrail`, `Authorizer`, `CredentialBroker`,
  `IToolExecutor`, `ITelemetrySink`, `KeyProvider`) as `runtime_checkable`
  `Protocol`s.
- Return errors up the call stack rather than swallowing them silently; security
  denials (`PolicyViolationError`) must propagate, never become `TOOL_ERROR`
  strings.

## Conventions

- The `Container` is the composition root: bind `Protocol → impl`, freeze
  security-critical keys, use `child()` for delegation scoping.
- `voussoir.protocols` re-exports ctxforge contracts; don't redefine them.
- Agent orchestration lives in `voussoir.agent`; security enforcement in
  `voussoir.executors.standard`; soft policies in `voussoir.guardrails`.
- New built-in guardrails go in `voussoir.guardrails.builtin` and are wired via
  `bind_default_guardrails`.

## Pull request process

1. Open an issue (or comment on an existing one) to discuss larger changes
   before implementing.
2. Fork the repo and create a feature branch.
3. Make your change, adding tests where practical.
4. Run `uv run pytest -q`, `ruff check src tests`, `black --check src tests`,
   and `mypy src/voussoir`.
5. Open a PR using the pull request template.

All contributions are made under the [Apache 2.0](LICENSE) license.

## Getting help

- Ask questions in [Discussions](https://github.com/efathom/voussoir/discussions).
- Report bugs via [Issues](https://github.com/efathom/voussoir/issues).
