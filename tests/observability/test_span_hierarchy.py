"""Locks span hierarchy emitted by Agent.run (Phase 5 Task C3).

Phase 5 §6.1 commits to a specific span tree. This test exercises a simple
Agent.run with a stub LLM + recording chain, then inspects the in-memory
span exporter for expected span names and gen_ai.* attributes.

OTel provider strategy
-----------------------
OTel's ``set_tracer_provider`` is guarded by a ``Once`` — the first real SDK
provider wins and cannot be replaced (only a warning is emitted). We therefore
install a **single** shared ``TracerProvider`` once per session (session-scoped
``_span_provider`` fixture defined in conftest.py), attach an
``InMemorySpanExporter`` to it, and expose a function-scoped ``span_exporter``
fixture that clears the exporter before each test. All span tests across the
observability suite share the same provider so the module-level tracers in
``standard.py`` and ``turn.py`` forward to the right SDK backend regardless
of import order.
"""

from __future__ import annotations


async def test_agent_run_emits_root_span(span_exporter, make_container, stub_llm):  # type: ignore[no-untyped-def]
    from voussoir import Agent

    a = Agent(name="x", container=make_container(stub_llm(content="hi")))
    await a.run("hello")
    span_names = [s.name for s in span_exporter.get_finished_spans()]
    assert "agent.run" in span_names


async def test_agent_run_span_has_agent_name_attr(span_exporter, make_container, stub_llm):  # type: ignore[no-untyped-def]
    from voussoir import Agent

    a = Agent(name="my-agent", container=make_container(stub_llm(content="hi")))
    await a.run("hello")
    agent_spans = [s for s in span_exporter.get_finished_spans() if s.name == "agent.run"]
    assert agent_spans, "expected agent.run span"
    attrs = dict(agent_spans[0].attributes or {})
    assert attrs.get("agent_name") == "my-agent"
    assert "finish_reason" in attrs


async def test_llm_complete_span_has_gen_ai_attrs(span_exporter, make_container, stub_llm):  # type: ignore[no-untyped-def]
    from voussoir import Agent

    a = Agent(name="x", container=make_container(stub_llm(content="hi")))
    await a.run("hello")
    llm_spans = [s for s in span_exporter.get_finished_spans() if s.name == "llm.complete"]
    assert llm_spans, "expected at least one llm.complete span"
    attrs = dict(llm_spans[0].attributes or {})
    assert "gen_ai.request.model" in attrs
    assert "gen_ai.usage.input_tokens" in attrs
    assert "gen_ai.usage.output_tokens" in attrs


async def test_tool_call_span_has_capability_attribute(span_exporter, make_container, stub_llm):  # type: ignore[no-untyped-def]
    """tool.call spans now carry the capability attribute (C3 widening)."""
    from voussoir.executors.standard import StandardExecutor
    from voussoir.tools import Capability, ToolContext, tool

    @tool(capability=Capability.READ_PUBLIC, name="ping")
    async def ping() -> str:
        return "pong"

    ex = StandardExecutor()
    ctx = ToolContext(
        run_id="r",
        span_id="s",
        allowed_capabilities=Capability.READ_PUBLIC,
    )
    await ex.invoke(ping, ping.input_schema(), ctx)
    tool_call_spans = [s for s in span_exporter.get_finished_spans() if s.name == "tool.call"]
    assert tool_call_spans, "expected at least one tool.call span"
    attrs = dict(tool_call_spans[0].attributes or {})
    assert "capability" in attrs
    assert "tool_name" in attrs
