# {{project_name}}

Minimal voussoir starter -- one agent, no tools.

## Run it

    pip install voussoir
    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py

## What's here

- `voussoir.yaml` -- yaml-driven topology (one agent named `{{project_name}}`).
- `main.py` -- Python entry point that runs the agent.

## Next steps

Add a tool with `@tool`, wire it via `Agent(tools=[...])`, and explore
`voussoir new {{project_name}} --template research` for a multi-agent example.
