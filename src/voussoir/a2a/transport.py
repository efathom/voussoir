"""JSON-RPC 2.0 envelope types + internal DelegateParams.

JsonRpcRequest / JsonRpcResponse / JsonRpcError are the wire-level envelope.
DelegateParams is the validated `params` shape for the voussoir.delegate
method.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request envelope."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | int | None = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object embedded in a response."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response envelope. Exactly one of result / error is set."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


class DelegateParams(BaseModel):
    """Validated `params` shape for the voussoir.delegate method.

    Phase 4b ships exactly one method; future methods (`voussoir.cancel`,
    `voussoir.list_capabilities`) will have their own Params models.
    """

    model_config = ConfigDict(extra="forbid")

    task: str
