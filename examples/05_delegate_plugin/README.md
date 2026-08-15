# 05 — Delegate plugin (Phase 4d)

Demonstrates the `voussoir.delegates` entry-point group: an installed
Python package advertises an `IDelegate` factory that voussoir picks up
when the host application opts in.

## Author side

The plugin declares its factory in `pyproject.toml`:

```toml
[project.entry-points."voussoir.delegates"]
example_plugin_agent = "example_plugin:make_plugin_agent"
```

`make_plugin_agent(c: Container) -> Agent` (in `src/example_plugin/__init__.py`)
returns an Agent. Any `IDelegate` works — `AgentRef` (Phase 4b remote
delegate), `NamedDelegate`, or a custom class.

## Consumer side

```bash
pip install -e .  # from examples/05_delegate_plugin/
```

Then in the host app:

```python
from voussoir import Agent
from voussoir.agent import bind_agent_registry
from voussoir.container.defaults import default_container

c = default_container()
bind_agent_registry(c, load_plugins=True)  # opt-in

lead = Agent(name="lead", delegates=["example_plugin_agent"], container=c)
result = await lead.run("Say hi to the plugin.")
print(result.output)
```

## Failure modes (Phase 4d guarantee)

A broken plugin (raise during import, raise during factory call, factory
returns a non-IDelegate, or name collision with a code/yaml registration)
logs a warning through the `voussoir.agent.plugins` logger and is skipped.
The host app continues to boot regardless.
