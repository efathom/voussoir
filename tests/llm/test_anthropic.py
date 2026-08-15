import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from ctxforge.protocols.llm import ChatMessage, LLMResponse

from voussoir.llm.anthropic import AnthropicLLMProvider


@pytest.fixture
def mock_client():
    """A mock anthropic.AsyncAnthropic with a Messages API."""
    client = MagicMock()
    client.messages.create = AsyncMock()
    return client


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(*, id: str, name: str, input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


async def test_chat_returns_llmresponse(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("hi from claude")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=12, output_tokens=4),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client, default_model="claude-opus-4-7")
    resp = await p.chat([ChatMessage(role="user", content="hello")])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "hi from claude"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 4
    assert resp.model == "claude-opus-4-7"
    assert resp.finish_reason == "end_turn"


async def test_chat_passes_system_message(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("sure")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=5, output_tokens=2),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    await p.chat(
        [
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"
    # System message is NOT in the messages array (Anthropic split):
    assert all(m["role"] != "system" for m in kwargs["messages"])


async def test_chat_translates_functions_to_tools(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=5, output_tokens=2),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    functions = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    await p.chat([ChatMessage(role="user", content="hi")], functions=functions)
    kwargs = mock_client.messages.create.call_args.kwargs
    assert "tools" in kwargs
    assert kwargs["tools"][0]["name"] == "web_search"
    assert kwargs["tools"][0]["input_schema"] == functions[0]["parameters"]


async def test_chat_surfaces_tool_use_in_raw_response(mock_client):
    tool_use = _tool_use_block(id="tool_call_1", name="web_search", input={"q": "voussoir"})
    mock_client.messages.create.return_value = MagicMock(
        content=[tool_use],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=8, output_tokens=12),
        stop_reason="tool_use",
    )
    p = AnthropicLLMProvider(client=mock_client)
    resp = await p.chat(
        [ChatMessage(role="user", content="search")],
        functions=[{"name": "web_search", "description": "", "parameters": {}}],
    )
    assert resp.finish_reason == "tool_use"
    assert resp.raw_response is not None
    assert resp.raw_response["tool_calls"][0]["name"] == "web_search"
    assert resp.raw_response["tool_calls"][0]["arguments"] == {"q": "voussoir"}


def test_provider_name_and_default_model(mock_client):
    p = AnthropicLLMProvider(client=mock_client, default_model="claude-haiku-4-5-20251001")
    assert p.name == "anthropic"
    assert p.default_model == "claude-haiku-4-5-20251001"


def test_from_env_raises_when_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMProvider.from_env()


async def test_chat_passes_temperature_only_when_set(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    await p.chat([ChatMessage(role="user", content="hi")])
    assert "temperature" not in mock_client.messages.create.call_args.kwargs

    mock_client.messages.create.reset_mock()
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    await p.chat([ChatMessage(role="user", content="hi")], temperature=0.2)
    assert mock_client.messages.create.call_args.kwargs["temperature"] == 0.2


async def test_chat_translates_assistant_tool_calls_to_tool_use_blocks(mock_client):
    """Assistant message with function_call={'tool_calls': [...]} translates
    to Anthropic's tool_use content blocks."""
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    await p.chat(
        [
            ChatMessage(role="user", content="search please"),
            ChatMessage(
                role="assistant",
                content="",
                function_call={
                    "tool_calls": [{"id": "tc_1", "name": "echo", "arguments": {"text": "hi"}}]
                },
            ),
            ChatMessage(
                role="function",
                content="echoed: hi",
                name="echo",
                function_call={"tool_use_id": "tc_1"},
            ),
        ]
    )
    sent = mock_client.messages.create.call_args.kwargs["messages"]
    # Three messages: user, assistant(tool_use), user(tool_result):
    assert len(sent) == 3
    assert sent[0] == {"role": "user", "content": "search please"}
    assert sent[1]["role"] == "assistant"
    assert isinstance(sent[1]["content"], list)
    use_block = sent[1]["content"][0]
    assert use_block["type"] == "tool_use"
    assert use_block["id"] == "tc_1"
    assert use_block["name"] == "echo"
    assert use_block["input"] == {"text": "hi"}
    # Tool result lands in a user message:
    assert sent[2]["role"] == "user"
    assert isinstance(sent[2]["content"], list)
    res_block = sent[2]["content"][0]
    assert res_block["type"] == "tool_result"
    assert res_block["tool_use_id"] == "tc_1"
    assert res_block["content"] == "echoed: hi"


async def test_chat_merges_consecutive_tool_results_into_one_user_message(mock_client):
    """Multiple tool_result replies in a row merge into one user message."""
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    await p.chat(
        [
            ChatMessage(role="user", content="parallel search"),
            ChatMessage(
                role="assistant",
                content="",
                function_call={
                    "tool_calls": [
                        {"id": "tc_1", "name": "a", "arguments": {}},
                        {"id": "tc_2", "name": "b", "arguments": {}},
                    ]
                },
            ),
            ChatMessage(
                role="function",
                content="a-result",
                name="a",
                function_call={"tool_use_id": "tc_1"},
            ),
            ChatMessage(
                role="function",
                content="b-result",
                name="b",
                function_call={"tool_use_id": "tc_2"},
            ),
        ]
    )
    sent = mock_client.messages.create.call_args.kwargs["messages"]
    # Three messages: user, assistant(2 tool_use), user(2 tool_result):
    assert len(sent) == 3
    assert sent[2]["role"] == "user"
    blocks = sent[2]["content"]
    assert len(blocks) == 2
    assert blocks[0]["tool_use_id"] == "tc_1"
    assert blocks[1]["tool_use_id"] == "tc_2"


async def test_chat_passes_stop_sequences(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("ok")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    await p.chat([ChatMessage(role="user", content="hi")], stop=["END", "DONE"])
    assert mock_client.messages.create.call_args.kwargs["stop_sequences"] == ["END", "DONE"]


async def test_generate_delegates_to_chat(mock_client):
    mock_client.messages.create.return_value = MagicMock(
        content=[_text_block("from generate")],
        model="claude-opus-4-7",
        usage=MagicMock(input_tokens=3, output_tokens=2),
        stop_reason="end_turn",
    )
    p = AnthropicLLMProvider(client=mock_client)
    resp = await p.generate("hello world")
    assert resp.content == "from generate"
    # Wraps the prompt in a single user ChatMessage:
    sent = mock_client.messages.create.call_args.kwargs["messages"]
    assert sent == [{"role": "user", "content": "hello world"}]


def test_token_counts_positive_and_summed(mock_client):
    """count_tokens is positive for non-empty text, 0 for empty, and
    count_message_tokens sums per-message counts (tokenizer is pluggable:
    tiktoken when installed, a char-based fallback otherwise)."""
    p = AnthropicLLMProvider(client=mock_client)
    assert p.count_tokens("") == 0
    assert p.count_tokens("one two three") > 0
    a = p.count_tokens("alpha beta")
    b = p.count_tokens("gamma")
    assert (
        p.count_message_tokens(
            [
                ChatMessage(role="user", content="alpha beta"),
                ChatMessage(role="assistant", content="gamma"),
            ]
        )
        == a + b
    )


async def test_stream_yields_chunks(mock_client):
    """stream() wraps Anthropic's async-context streaming API into an async generator."""

    class _FakeTextStream:
        def __init__(self, chunks: list[str]) -> None:
            self._chunks = chunks

        def __aiter__(self):
            async def gen():
                for c in self._chunks:
                    yield c

            return gen()

    class _FakeStreamCM:
        def __init__(self, chunks: list[str]) -> None:
            self.text_stream = _FakeTextStream(chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    mock_client.messages.stream = MagicMock(return_value=_FakeStreamCM(["hello", " ", "world"]))

    p = AnthropicLLMProvider(client=mock_client)
    chunks = []
    async for c in p.stream(
        [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")],
        temperature=0.3,
    ):
        chunks.append(c)
    assert chunks == ["hello", " ", "world"]
    # System message split out; temperature forwarded.
    kwargs = mock_client.messages.stream.call_args.kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["temperature"] == 0.3


@pytest.mark.skipif(
    "ANTHROPIC_API_KEY" not in os.environ
    or os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-test"),
    reason="live integration test — requires a real ANTHROPIC_API_KEY",
)
async def test_chat_live_anthropic():
    """One sanity test against the real API. Skip if no key."""
    p = AnthropicLLMProvider.from_env(default_model="claude-haiku-4-5-20251001")
    resp = await p.chat([ChatMessage(role="user", content="say 'pong'")])
    assert "pong" in resp.content.lower()
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
