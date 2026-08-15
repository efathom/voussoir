# {{project_name}} -- Research

Multi-agent research workflow: a lead agent delegates to a `researcher` (with a
`web_search` tool stub) and a `writer`.

## Run it

    pip install voussoir
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py

## What's here

- `voussoir.yaml` -- agent topology (lead + 2 delegates).
- `tools.py` -- `@tool web_search` (stub; replace with a real backend).
- `main.py` -- Python entry point loading agents from yaml.

## How tools are wired

The `@tool` decorator does not auto-register tools globally; you must
explicitly attach them to agents. `main.py` shows the pattern — import
your tool, then pass it via `Agent(tools=[...])`.

`voussoir.yaml` documents the intended topology but is not loaded at
runtime; the yaml path does not support tool registration, so the three
agents are constructed directly in Python.

## Swap in a real web_search

Replace `web_search` in `tools.py` with a call to your search provider (e.g.
SerpAPI, Brave Search). Match the `@tool(capability=READ_PUBLIC, name="web_search")`
signature so the existing Python wiring continues to find it.
