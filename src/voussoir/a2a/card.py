"""AgentCard Pydantic model + AuthMethod + card_from_agent helper.

The AgentCard is the public, JWS-signed contract published at
/.well-known/agent-card.json. extra='forbid' keeps the wire shape pinned.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

if TYPE_CHECKING:
    from voussoir.agent.agent import Agent


class AuthMethod(BaseModel):
    """One auth method the publisher accepts.

    Phase 4b ships only Bearer JWT. mTLS / OAuth2 land in v1.5.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["bearer"]
    scheme: Literal["JWT"] = "JWT"


class AgentCard(BaseModel):
    """Schema published at /.well-known/agent-card.json.

    Construction: built by Agent.expose_a2a wiring via card_from_agent.
    Validation: any third party fetching the card parses through this
    model. extra='forbid' rejects unknown fields, pinning the wire shape.

    Signature: card is JWS-signed by the publisher's key; consumers MUST
    verify against the JWKS advertised by `jwks_uri` (HTTPS-only, loopback
    permitted for local dev).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    version: str = "1.0.0"
    endpoint: HttpUrl
    capabilities: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    auth: list[AuthMethod] = Field(default_factory=list)
    supports: list[Literal["json-rpc/2.0", "sse-streaming"]] = Field(
        default_factory=lambda: ["json-rpc/2.0"]  # type: ignore[arg-type]
    )
    jwks_uri: HttpUrl | None = None
    # v1.0.4 E2 / C3: optional Unix-timestamp (seconds) expiry. Cards with
    # `exp` in the past are rejected by discover_card; absent `exp` preserves
    # back-compat (cards signed before E2 had no expiry concept). Operators
    # opt in by passing `ttl_s=` to `card_from_agent` (or by binding a custom
    # publisher in Phase 6).
    exp: int | None = None


def card_from_agent(
    agent: Agent,
    *,
    endpoint: HttpUrl,
    jwks_uri: HttpUrl | None = None,
    ttl_s: int | None = None,
) -> AgentCard:
    """Build an AgentCard from an Agent's public surface.

    capabilities is empty in Phase 4b (Phase 5 walks the agent's tool/skill
    graph to populate). input_schema is the hardcoded {"task": str} shape
    matching the JSON-RPC `voussoir.delegate` method.

    `ttl_s` (v1.0.4 E2 / C3): when set, stamps `exp = int(time.time()) + ttl_s`
    so the card naturally expires `ttl_s` seconds after issuance. The
    discover_card consumer enforces this. Default `None` means no expiry
    (back-compat for pre-E2 callers).
    """
    exp: int | None = int(time.time()) + ttl_s if ttl_s is not None else None
    return AgentCard(
        name=agent.name,
        description=agent.description,
        endpoint=endpoint,
        capabilities=[],
        input_schema={
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
        output_schema={"type": "string"},
        auth=[AuthMethod(type="bearer")],
        supports=["json-rpc/2.0"],
        jwks_uri=jwks_uri,
        exp=exp,
    )
