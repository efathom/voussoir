"""Defense-in-depth Pydantic schema re-validation for tool_call args.

Use this on tool_call stage as a sanity layer past the LLM's tool-call args
generation — re-validates `payload.tool_args` against the target tool's
Pydantic input schema. Catches a class of "garbage args reaching the tool"
bugs where the LLM might have hallucinated a parameter shape.

Looks up the schema via `ctx.tool_input_schema(tool_name) -> type[BaseModel] | None`
when present on ctx; falls back to ALLOW if the schema isn't accessible (e.g.,
test fixtures that don't carry a registry). The B5 task wires `AgentContext`
with a real `tool_input_schema` accessor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict


class ArgsSchemaCheck:
    """Re-validate tool_call args against the tool's Pydantic input schema.

    Use this as a defense-in-depth layer on tool_call stage. If the ctx
    exposes a `tool_input_schema` accessor and a schema is found for the tool,
    validation errors become BLOCK verdicts. Falls through to ALLOW when no
    schema accessor or schema is available.
    """

    name = "args_schema_check"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "tool_call"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        # ctx may be None, a real AgentContext, or a test stub. Look up the
        # schema accessor via getattr to stay loose-coupled.
        accessor = getattr(ctx, "tool_input_schema", None)
        if accessor is None or payload.tool_args is None or payload.tool_name is None:
            return GuardrailVerdict(verdict="ALLOW")
        schema = accessor(payload.tool_name)
        if schema is None:
            return GuardrailVerdict(verdict="ALLOW")
        try:
            schema.model_validate(payload.tool_args)
        except ValidationError as e:
            return GuardrailVerdict(
                verdict="BLOCK",
                reason=f"args schema validation: {e.errors()[0]['msg']}",
            )
        return GuardrailVerdict(verdict="ALLOW")
