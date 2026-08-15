"""End-to-end A2A round-trip: publisher + AgentRef in the same process via
ASGI, plus one real-uvicorn smoke test."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import threading
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_e2e_asgi_publisher_caller_in_same_process(
    make_container, stub_llm, fresh_env
) -> None:
    """Full round-trip via ASGITransport — no real network."""
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext

    server_c = make_container(stub_llm(content="research result"))
    server_agent = Agent("researcher", description="r", container=server_c)
    app = FastAPI()
    app.include_router(make_a2a_router(server_agent, endpoint="https://test/a2a"))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        ref = await AgentRef.discover("https://test", http_client=client)

        caller_c = make_container()
        caller_c.bind(KeyProvider, server_c.resolve(KeyProvider))  # type: ignore[type-abstract]

        async with await AgentContext.open(
            container=caller_c, run_id="r", session_id="s", user_id="u"
        ) as parent_ctx:
            parent_ctx.agent_name = "caller"
            result = await ref.delegate("research X", parent_ctx=parent_ctx)
        assert "research result" in result.output


@pytest.mark.skipif(
    os.environ.get("VOUSSOIR_SKIP_REAL_UVICORN") == "1",
    reason="VOUSSOIR_SKIP_REAL_UVICORN set",
)
@pytest.mark.asyncio
async def test_e2e_real_uvicorn_smoke(make_container, stub_llm, fresh_env) -> None:
    """Spin up serve_a2a in a background thread on an ephemeral port and
    verify a real-network round-trip works."""
    fresh_env.setenv("VOUSSOIR_A2A_ALLOWED_ISSUERS", "caller")
    import uvicorn

    from voussoir.a2a.agent_ref import AgentRef
    from voussoir.a2a.keys import KeyProvider
    from voussoir.a2a.publisher import make_a2a_router
    from voussoir.agent.agent import Agent
    from voussoir.agent.context import AgentContext

    server_c = make_container(stub_llm(content="real-uvicorn ran"))
    server_agent = Agent("researcher", description="r", container=server_c)

    # Pick an ephemeral port.
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    app = FastAPI()
    app.include_router(make_a2a_router(server_agent, endpoint=f"http://127.0.0.1:{port}/a2a"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Wait for the server to bind (poll).
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                break
        except OSError:
            await asyncio.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.skip("uvicorn failed to bind within 10s")

    try:
        # Phase 4.5a P0 #8: discover without http_client returns a ref with
        # _http=None; wrap in async-with for the .delegate() call.
        ref = await AgentRef.discover(f"http://127.0.0.1:{port}")
        async with ref:
            caller_c = make_container()
            caller_c.bind(KeyProvider, server_c.resolve(KeyProvider))  # type: ignore[type-abstract]

            async with await AgentContext.open(
                container=caller_c, run_id="r", session_id="s", user_id="u"
            ) as parent_ctx:
                parent_ctx.agent_name = "caller"
                result = await ref.delegate("X", parent_ctx=parent_ctx)
            assert "real-uvicorn ran" in result.output
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            pytest.fail("uvicorn server thread didn't exit within 5s")
