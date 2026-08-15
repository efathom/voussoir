"""v1.0.4 E8 — warn-once flags reset between tests via autouse fixture.

Without the reset, the second test in this module would see the warn-once
flag stuck at True from the first test, and structlog.testing.capture_logs()
would record 0 events for the second call.
"""

from __future__ import annotations

import structlog


async def test_allow_all_warning_fires_first_invocation() -> None:
    from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer

    authorizer = AllowAllAuthorizer()
    with structlog.testing.capture_logs() as captured:
        await authorizer.authorize(
            principal=object(),
            tool=object(),
            args=object(),
            ctx=object(),
        )

    warn_events = [r for r in captured if r.get("event") == "authz_unenforced"]
    assert len(warn_events) == 1


async def test_allow_all_warning_fires_again_in_next_test() -> None:
    """If the autouse reset fixture is working, this second test sees the warning fire too."""
    from voussoir.auth.authorizers.allow_all import AllowAllAuthorizer

    authorizer = AllowAllAuthorizer()
    with structlog.testing.capture_logs() as captured:
        await authorizer.authorize(
            principal=object(),
            tool=object(),
            args=object(),
            ctx=object(),
        )

    warn_events = [r for r in captured if r.get("event") == "authz_unenforced"]
    assert len(warn_events) == 1, (
        "Second test in the module sees no warn -- the autouse fixture isn't "
        "resetting voussoir.auth.authorizers.allow_all._warned between tests."
    )
