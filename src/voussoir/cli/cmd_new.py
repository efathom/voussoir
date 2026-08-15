"""voussoir new -- scaffold a starter project (Phase 6 Task B2).

Use this CLI command to bootstrap a new voussoir project. Two templates:
`minimal` (single agent, no tools) and `research` (lead + 2 delegates +
web_search stub). Templates ship inside the wheel under
voussoir/templates/<name>/ and are copied via importlib.resources.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import click

_TEMPLATE_PACKAGE = "voussoir.templates"
_SUPPORTED_TEMPLATES = ("minimal", "research")


def _copy_template_tree(src: Traversable, dst: Path, *, project_name: str) -> None:
    """Recursively copy a template Traversable tree to `dst`, substituting
    {{project_name}} placeholders. Skips __pycache__ + __init__.py."""
    for entry in src.iterdir():
        name = entry.name
        if name in {"__pycache__", "__init__.py"}:
            continue
        target = dst / name
        if entry.is_dir():
            target.mkdir(exist_ok=True)
            _copy_template_tree(entry, target, project_name=project_name)
        else:
            text = entry.read_text(encoding="utf-8")
            text = text.replace("{{project_name}}", project_name)
            target.write_text(text, encoding="utf-8")


@click.command(name="new")
@click.argument("name")
@click.option(
    "--template",
    type=click.Choice(_SUPPORTED_TEMPLATES),
    default="minimal",
    show_default=True,
    help="Starter template to scaffold.",
)
def new(name: str, template: str) -> None:
    """Scaffold a new voussoir project under ./<name>/."""
    target = Path.cwd() / name
    if target.exists() and target.is_file():
        raise click.ClickException(
            f"target {target} is a file, not a directory; refusing to overwrite"
        )
    if target.exists() and any(target.iterdir()):
        raise click.ClickException(
            f"target {target} exists and is non-empty; refusing to overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)

    src = resources.files(_TEMPLATE_PACKAGE) / template
    _copy_template_tree(src, target, project_name=name)

    click.echo(f"created {target} from template={template}")
    click.echo(f"next: cd {name} && python main.py")
