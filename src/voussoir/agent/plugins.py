"""Phase 4d — discover delegate factories from installed-package entry points.

A plugin author declares an entry point in their `pyproject.toml`:

    [project.entry-points."voussoir.delegates"]
    my_helper = "my_package.delegates:make_helper"

The factory must be a `Callable[[Container], IDelegate]`. `Agent` (Phase 4a)
and `AgentRef` (Phase 4b) both satisfy `IDelegate` natively; so does any
third-party implementation with `name`, `description`, async `delegate`.

The host application opts in via `bind_agent_registry(c, load_plugins=True)`;
loader failures all log a warning and skip rather than crashing startup.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from voussoir.agent.delegate import IDelegate
from voussoir.agent.registry import AgentRegistry
from voussoir.container import Container
from voussoir.observability.logging_setup import get_logger

ENTRY_POINT_GROUP = "voussoir.delegates"

DelegateFactory = Callable[[Container], IDelegate]


def load_delegate_plugins(
    c: Container,
    *,
    registry: AgentRegistry,
    allowed_names: set[str] | None = None,
) -> list[str]:
    """Iterate the `voussoir.delegates` entry-point group; call each factory;
    add the returned IDelegate to `registry`.

    Returns the list of delegate names successfully registered. Never raises:
    each loader failure (broken import, factory raise, non-IDelegate return,
    name collision) logs a warning via the voussoir.agent.plugins logger and
    is skipped so the host application continues to boot.

    Phase 4.5a P0 #4:
      - `allowed_names=None` (default): every successfully-loaded plugin
        is added AND emits an INFO log so operators can audit what an
        opaque venv pulled in.
      - `allowed_names={"x", "y"}`: only plugins whose `delegate.name` is
        in the set are added; others log a warning and are skipped.
      - `allowed_names=set()`: strict deny-all (test/dev pattern).

    Called from `bind_agent_registry(load_plugins=True)` AFTER yaml + code
    registration, so plugins always lose collisions against the application's
    own registrations.
    """
    log = get_logger(__name__)
    loaded: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = ep.load()
        except Exception as exc:
            log.warning(
                "plugin_load_failed",
                entry_point=ep.name,
                error=str(exc),
            )
            continue
        try:
            delegate = factory(c)
        except Exception as exc:
            log.warning(
                "plugin_factory_raised",
                entry_point=ep.name,
                error=str(exc),
            )
            continue
        if not isinstance(delegate, IDelegate):
            log.warning(
                "plugin_returned_non_idelegate",
                entry_point=ep.name,
                returned_type=type(delegate).__name__,
            )
            continue
        if allowed_names is not None and delegate.name not in allowed_names:
            log.warning(
                "plugin_not_in_allowed_names",
                entry_point=ep.name,
                delegate_name=delegate.name,
                allowed_names=sorted(allowed_names),
            )
            continue
        if registry.has(delegate.name):
            log.warning(
                "plugin_name_collides",
                entry_point=ep.name,
                delegate_name=delegate.name,
            )
            continue
        registry.add(delegate)
        loaded.append(delegate.name)
        log.info(
            "plugin_loaded",
            entry_point=ep.name,
            delegate_name=delegate.name,
            dist_name=getattr(getattr(ep, "dist", None), "name", None),
        )
    return loaded
