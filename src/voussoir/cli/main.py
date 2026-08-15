"""voussoir CLI entry point (Phase 6 Tranche B).

Use this module as the entry point for the `voussoir` console script.
Sub-commands (new/run/validate/doctor) live in sibling cmd_*.py modules.
"""

from __future__ import annotations

import click

from voussoir import __version__
from voussoir.cli.cmd_doctor import doctor
from voussoir.cli.cmd_new import new
from voussoir.cli.cmd_run import run
from voussoir.cli.cmd_validate import validate


@click.group()
@click.version_option(__version__, prog_name="voussoir")
def main() -> None:
    """voussoir — default-runnable LLM agent framework CLI."""


main.add_command(new)
main.add_command(run)
main.add_command(validate)
main.add_command(doctor)


if __name__ == "__main__":
    main()
