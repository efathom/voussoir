"""Simple-agent streaming path estimates token totals (v1.0.3 housekeeping).

`ILLMProvider.stream()` yields raw `str` chunks without usage totals, so prior
to v1.0.3 the simple-agent stream path always reported `tokens_in=0` and
`tokens_out=0` on the post-stream `pseudo_result`. v1.0.3 estimates both
client-side via `llm.count_tokens(...)` after the stream completes. These
are approximations -- the provider's actual tokenizer may give slightly
different numbers -- but they're far better than 0.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ctxforge.protocols.llm import ILLMProvider

from voussoir import Container
from voussoir.agent import Agent
from voussoir.memory.adapter import InMemorySessionStore, InMemoryStore
from voussoir.protocols import ILLMProvider as ILLMProviderProto
from voussoir.protocols import IMemoryStore, ISessionStore


async def _streaming_chunks(*texts: str):  # type: ignore[no-untyped-def]
    for t in texts:
        yield t


def _container_with_counting_llm(chunks: tuple[str, ...]) -> tuple[Container, MagicMock]:
    """Build a Container + a counting-aware ILLMProvider mock.

    `count_tokens(text)` returns `len(text)` -- a deterministic 1-char-per-token
    proxy adequate for asserting "> 0 and proportional to text length".
    """
    p = MagicMock(spec=ILLMProvider)
    p.stream = MagicMock(return_value=_streaming_chunks(*chunks))
    p.count_tokens = MagicMock(side_effect=lambda text, model=None: len(text))
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    return c, p


async def test_simple_path_stream_estimates_tokens_in_out() -> None:
    """Use this when verifying that simple-path streaming surfaces non-zero
    token estimates via llm.count_tokens(). Reads the post-stream agent state
    via the `done` event's pseudo_result-equivalent fields aren't exposed on
    the event itself, so this test checks the count_tokens calls were made
    with the expected inputs (which is what drives tokens_in / tokens_out)."""
    container, llm = _container_with_counting_llm(("hello", " ", "world"))
    agent = Agent(name="t", container=container)

    events = [e async for e in agent.stream("hi user")]
    kinds = [e.kind for e in events]
    assert kinds[-1] == "done"

    # After the stream, count_tokens should have been called for:
    #   - each message in the input prompt (>= 1: the user message "hi user")
    #   - once on the full assembled output "hello world"
    call_texts = [call.args[0] for call in llm.count_tokens.call_args_list]
    # At least one input-message call and one output call.
    assert any(text == "hello world" for text in call_texts), (
        f"expected count_tokens called on full output 'hello world'; "
        f"actual calls: {call_texts!r}"
    )
    # And at least one call for an input message (the user's "hi user" is
    # built into the messages list).
    assert any("hi user" in (text or "") for text in call_texts), (
        f"expected count_tokens called on a message containing 'hi user'; "
        f"actual calls: {call_texts!r}"
    )


async def test_simple_path_stream_pseudo_result_has_nonzero_tokens() -> None:
    """End-to-end: the post-`done` AgentResult passed to after_run middleware
    must carry tokens_in > 0 and tokens_out > 0 once count_tokens succeeds.
    Prior to v1.0.3 both stayed at 0 because llm.stream() yields raw chunks
    without usage info."""
    from voussoir.agent.result import AgentResult

    captured: list[AgentResult[str]] = []

    class _Capture:
        """Structural Middleware implementation (duck-typed against
        voussoir.middleware.protocol.Middleware)."""

        async def before_run(self, ctx, input):  # type: ignore[no-untyped-def]
            return None

        async def after_step(self, ctx, step):  # type: ignore[no-untyped-def]
            return None

        async def after_run(self, ctx, result):  # type: ignore[no-untyped-def]
            captured.append(result)
            return None

        async def on_error(self, ctx, exc):  # type: ignore[no-untyped-def]
            return None

    container, _llm = _container_with_counting_llm(("hello", " ", "world"))
    agent = Agent(name="t", container=container)
    agent.middleware.append(_Capture())

    async for _ in agent.stream("ping"):
        pass

    assert len(captured) == 1, "after_run should fire exactly once"
    result = captured[0]
    # `count_tokens` returns len(text); the assembled output "hello world" is
    # 11 chars so tokens_out == 11 in our proxy. tokens_in is at least the
    # length of "ping" (4) and likely more (system prompt, etc.).
    assert result.tokens_out == 11, f"expected tokens_out=11, got {result.tokens_out}"
    assert result.tokens_in >= 4, f"expected tokens_in >= len('ping')==4, got {result.tokens_in}"


async def test_simple_path_stream_tolerates_count_tokens_failure() -> None:
    """If llm.count_tokens raises (un-configured mock, provider that doesn't
    implement it, etc.), the simple-path stream must not crash; it falls
    back to tokens_in=0, tokens_out=0 silently."""
    p = MagicMock(spec=ILLMProvider)
    p.stream = MagicMock(return_value=_streaming_chunks("a", "b"))
    p.count_tokens = MagicMock(side_effect=NotImplementedError("not supported"))
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    agent = Agent(name="t", container=c)

    events = [e async for e in agent.stream("hi")]
    kinds = [e.kind for e in events]
    # Stream completes with `done` -- no exception leaked from the count_tokens
    # NotImplementedError.
    assert kinds[-1] == "done"
    assert events[-1].payload["output"] == "ab"


async def test_simple_path_stream_tolerates_non_int_count_tokens() -> None:
    """A MagicMock spec'd off ILLMProvider returns a MagicMock from
    `count_tokens()` by default (not an int). The simple-path code must
    detect that and leave tokens_in/tokens_out at their default 0,
    rather than propagating a MagicMock into AgentResult fields."""
    p = MagicMock(spec=ILLMProvider)
    p.stream = MagicMock(return_value=_streaming_chunks("x"))
    # Don't configure count_tokens.side_effect/return_value; spec'd MagicMock
    # will return another MagicMock when called.
    c = Container()
    c.bind(ILLMProviderProto, p)
    c.bind(IMemoryStore, InMemoryStore())
    c.bind(ISessionStore, InMemorySessionStore())
    agent = Agent(name="t", container=c)

    events = [e async for e in agent.stream("hi")]
    assert events[-1].kind == "done"
    assert events[-1].payload["output"] == "x"
