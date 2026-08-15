def test_voussoir_specific_protocols_importable():
    from voussoir.auth.protocol import Authorizer, CredentialBroker
    from voussoir.executors.protocol import ToolExecutor
    from voussoir.guardrails.protocol import Guardrail, GuardrailVerdict
    from voussoir.middleware.protocol import Middleware
    from voussoir.tools.protocol import Tool, ToolContext

    # GuardrailVerdict is a Pydantic model; the rest are runtime_checkable Protocols.
    for proto in [Tool, ToolExecutor, Guardrail, Middleware, CredentialBroker, Authorizer]:
        assert isinstance(proto, type)

    # ToolContext + GuardrailVerdict are concrete Pydantic models:
    assert isinstance(ToolContext, type)
    assert isinstance(GuardrailVerdict, type)
