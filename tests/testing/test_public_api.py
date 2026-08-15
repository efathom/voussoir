"""Locks voussoir.testing public API (v1.1.0 F10)."""

from __future__ import annotations

import asyncio


def test_make_container_returns_fresh_container() -> None:
    from voussoir.container import Container
    from voussoir.protocols import IMemoryStore
    from voussoir.testing import make_container

    container = make_container()
    assert isinstance(container, Container)

    # NOT a default_container() result — no freeze should fire.
    class Stub:
        pass

    container.bind(IMemoryStore, Stub())  # Should NOT raise.


def test_stub_llm_default_end_turn() -> None:
    from ctxforge.protocols.llm import LLMResponse

    from voussoir.testing import stub_llm

    llm = stub_llm()
    response = asyncio.run(llm.chat(messages=[], tools=[]))
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "end_turn"
    assert response.content == "ok"


def test_stub_llm_with_tool_calls() -> None:
    from voussoir.testing import stub_llm

    llm = stub_llm(
        content="",
        finish_reason="tool_use",
        tool_calls=[{"id": "tc1", "name": "search", "arguments": {"q": "x"}}],
    )
    response = asyncio.run(llm.chat(messages=[], tools=[]))
    assert response.finish_reason == "tool_use"


def test_multi_turn_llm_consumes_responses_in_order() -> None:
    from ctxforge.protocols.llm import LLMResponse

    from voussoir.testing import multi_turn_llm

    turns = [
        LLMResponse(
            content="first",
            model="x",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
            raw_response={},
        ),
        LLMResponse(
            content="second",
            model="x",
            input_tokens=1,
            output_tokens=1,
            finish_reason="end_turn",
            raw_response={},
        ),
    ]
    llm = multi_turn_llm(turns)
    r1 = asyncio.run(llm.chat(messages=[], tools=[]))
    r2 = asyncio.run(llm.chat(messages=[], tools=[]))
    assert r1.content == "first"
    assert r2.content == "second"


def test_stub_llm_side_effect_raises(monkeypatch: object) -> None:
    """stub_llm(side_effect=ExcCls) wires AsyncMock.side_effect so .chat raises."""
    from voussoir.testing import stub_llm

    class BoomError(Exception):
        pass

    llm = stub_llm(side_effect=BoomError("boom"))
    try:
        asyncio.run(llm.chat(messages=[], tools=[]))
    except BoomError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected BoomError to propagate from .chat")


def test_stub_llm_custom_name() -> None:
    """stub_llm(name='openai') wires the LLM .name attribute (used by adapter_for)."""
    from voussoir.testing import stub_llm

    llm = stub_llm(name="openai")
    assert llm.name == "openai"


def test_make_key_provider_returns_keyprovider() -> None:
    from voussoir.a2a.keys import KeyProvider
    from voussoir.testing import make_key_provider

    provider = make_key_provider()
    assert isinstance(provider, KeyProvider)


def test_make_key_provider_with_explicit_secret() -> None:
    from voussoir.testing import make_key_provider

    provider = make_key_provider(jwt_secret=b"deterministic-secret")
    assert provider.jwt_secret() == b"deterministic-secret"
