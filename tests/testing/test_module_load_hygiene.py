"""voussoir.testing must not eagerly import heavy optional deps (v1.1.0 F10)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_importing_voussoir_testing_does_not_pull_heavy_deps() -> None:
    """A subprocess imports voussoir.testing and reports its sys.modules.

    Heavy optional deps (keyring, fastapi, uvicorn) should be lazy-imported
    only inside helper bodies if at all — never eagerly at module load.
    """
    script = textwrap.dedent(
        """
        import sys
        import voussoir.testing  # noqa: F401
        forbidden = {"keyring", "fastapi", "uvicorn"}
        found = forbidden & set(sys.modules)
        print(",".join(sorted(found)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    found = result.stdout.strip()
    assert found == "", f"voussoir.testing eagerly imports: {found}"
