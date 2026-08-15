"""JSON-RPC envelope types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voussoir.a2a.errors import A2AErrorCode, CardVerificationError
from voussoir.a2a.transport import (
    DelegateParams,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
)


def test_jsonrpc_request_minimal() -> None:
    req = JsonRpcRequest(method="voussoir.delegate")
    assert req.jsonrpc == "2.0"
    assert req.method == "voussoir.delegate"
    assert req.params == {}
    assert req.id is None


def test_jsonrpc_request_full() -> None:
    req = JsonRpcRequest(
        method="voussoir.delegate",
        params={"task": "do it"},
        id="req-42",
    )
    assert req.params == {"task": "do it"}
    assert req.id == "req-42"


def test_jsonrpc_request_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        JsonRpcRequest(method="x", extra="nope")  # type: ignore[call-arg]


def test_jsonrpc_response_with_result() -> None:
    resp = JsonRpcResponse(id="r1", result={"output": "ok"})
    assert resp.jsonrpc == "2.0"
    assert resp.result == {"output": "ok"}
    assert resp.error is None


def test_jsonrpc_response_with_error() -> None:
    err = JsonRpcError(code=-32602, message="invalid params")
    resp = JsonRpcResponse(id="r1", error=err)
    assert resp.error is not None
    assert resp.error.code == -32602
    assert resp.result is None


def test_delegate_params_requires_task() -> None:
    with pytest.raises(ValidationError):
        DelegateParams()  # type: ignore[call-arg]


def test_delegate_params_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        DelegateParams(task="ok", extra="nope")  # type: ignore[call-arg]


def test_a2a_error_codes_present() -> None:
    """Lock the error-code allocation from spec §5.3."""
    assert A2AErrorCode.PARSE_ERROR == -32700
    assert A2AErrorCode.INVALID_REQUEST == -32600
    assert A2AErrorCode.METHOD_NOT_FOUND == -32601
    assert A2AErrorCode.INVALID_PARAMS == -32602
    assert A2AErrorCode.INTERNAL_ERROR == -32603
    assert A2AErrorCode.AUTHENTICATION_FAILED == -32001
    assert A2AErrorCode.AUTHORIZATION_FAILED == -32002
    assert A2AErrorCode.POLICY_VIOLATION == -32003
    assert A2AErrorCode.DELEGATION_REFUSED == -32004


def test_card_verification_error_is_exception() -> None:
    """CardVerificationError is a regular Exception subclass."""
    assert issubclass(CardVerificationError, Exception)
    err = CardVerificationError("tampered sig")
    assert str(err) == "tampered sig"
