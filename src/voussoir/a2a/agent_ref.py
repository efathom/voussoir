"""AgentRef — IDelegate impl for remote A2A peers.

Use this when you want to delegate to an agent published at a remote
endpoint discovered via voussoir.a2a.discover_card. Cross-references
the AgentCard for endpoint + auth metadata; mints a JWT per call using
the configured KeyProvider; raises typed DelegationError subclasses
(RemoteUnreachable / RemoteAuthFailed / RemoteProtocolError /
RemoteMalformed) on failure.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import httpx
import jwt
import pydantic

from voussoir.a2a.card import AgentCard
from voussoir.a2a.discovery import discover_card
from voussoir.a2a.errors import (
    RemoteAuthFailed,
    RemoteMalformed,
    RemoteProtocolError,
    RemoteUnreachable,
)
from voussoir.a2a.keys import KeyProvider
from voussoir.a2a.transport import JsonRpcRequest, JsonRpcResponse
from voussoir.agent.result import AgentResult
from voussoir.auth.a2a import PrincipalForwarder

if TYPE_CHECKING:
    from voussoir.agent.context import AgentContext


class AgentRef:
    """Remote agent reference. Satisfies IDelegate (Phase 4a Protocol) so
    it composes uniformly with local Agent and NamedDelegate in
    Agent(delegates=[...]).

    Phase 4.5a P0 #8 lifecycle:
      1. Recommended — `async with AgentRef(card) as ref:` lazily constructs
         an httpx.AsyncClient on __aenter__ and closes it on __aexit__.
      2. Caller-owned — `AgentRef(card, http_client=external)` uses the
         provided client and does NOT close it (caller owns lifecycle).
      3. Bare `AgentRef(card)` followed by `.delegate()` raises with a
         clear pointer at the two supported patterns. Pre-4.5a, the sync
         constructor opened a pool that nobody closed (long-running Agents
         with delegates=[ref] leaked connection pools).
    """

    def __init__(
        self,
        card: AgentCard,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.card = card
        self.name = card.name
        self.description = card.description
        self._http: httpx.AsyncClient | None = http_client
        # _owns_client is True only when __aenter__ constructed a client; an
        # externally-supplied client is never closed by AgentRef.
        self._owns_client = False

    @classmethod
    async def discover(
        cls,
        url: str,
        *,
        verify_signature: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> AgentRef:
        """Sugar over discover_card(url) + AgentRef(card).

        Phase 4.5a lifecycle: if `http_client` is provided, the returned
        AgentRef holds a reference to it (external-client mode; `.delegate()`
        works without async-with). If not, a temporary client is created for
        the card fetch and closed immediately; the returned AgentRef has
        `_http=None` and the caller MUST wrap it in `async with` before
        calling `.delegate()`.
        """
        if http_client is not None:
            card = await discover_card(
                url,
                verify_signature=verify_signature,
                http_client=http_client,
            )
            return cls(card, http_client=http_client)
        async with httpx.AsyncClient(timeout=30.0) as temp_client:
            card = await discover_card(
                url,
                verify_signature=verify_signature,
                http_client=temp_client,
            )
        return cls(card, http_client=None)

    async def aclose(self) -> None:
        """Close the owned httpx client (if any). No-op for caller-supplied clients."""
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None
            self._owns_client = False

    async def __aenter__(self) -> AgentRef:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
            self._owns_client = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def delegate(self, task: str, *, parent_ctx: AgentContext) -> AgentResult[str]:
        """IDelegate.delegate — make a JSON-RPC call to the remote agent.

        Mints a fresh JWT via the caller's KeyProvider, POSTs the
        voussoir.delegate request, maps transport / auth / JSON-RPC errors
        to PolicyViolationError (which the synthetic-tool layer wraps as
        DELEGATION_REFUSED). On success: parses the result as
        AgentResult.model_validate.

        Phase 4.5a P0 #8: requires that _http be set, either via
        `async with AgentRef(card) as ref:` or via the http_client kwarg.
        """
        if self._http is None:
            raise RuntimeError(
                f"AgentRef({self.card.name!r}) requires either "
                "`async with AgentRef(card) as ref:` or "
                "`AgentRef(card, http_client=external_client)` "
                "before .delegate() can be called. The bare AgentRef(card) "
                "constructor no longer opens its own httpx client "
                "(Phase 4.5a P0 #8)."
            )
        # Phase 6 A9: if a PrincipalForwarder is bound on the container, mint a
        # downstream JWT from parent_ctx.principal. Otherwise fall back to the
        # existing symmetric-secret JWT via _issue_jwt.
        endpoint_str = str(self.card.endpoint)
        try:
            forwarder: PrincipalForwarder = parent_ctx.container.resolve(
                PrincipalForwarder,  # type: ignore[type-abstract]
            )
            # Pass audience=self.card.name so the forwarder can bind the `aud`
            # claim to the receiving agent's name — which is what make_a2a_router
            # verifies (audience=agent.name in jwt.decode).
            token = await forwarder.forward(
                parent_ctx.principal, endpoint_str, audience=self.card.name
            )
        except LookupError:
            token = self._issue_jwt(parent_ctx)
        request = JsonRpcRequest(
            method="voussoir.delegate",
            params={"task": task},
            id=uuid.uuid4().hex,
        )
        # Phase 4.5a P1 #25: typed errors per failure mode. The synthetic
        # delegate-tool invoker (dispatch.py) wraps them all as
        # DELEGATION_REFUSED for the LLM.
        try:
            resp = await self._http.post(
                str(self.card.endpoint),
                json=request.model_dump(mode="json", exclude_none=True),
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise RemoteUnreachable(f"remote {self.card.endpoint} unreachable: {exc}") from exc

        if resp.status_code in (401, 403):
            raise RemoteAuthFailed(
                f"remote {self.card.endpoint} refused authentication "
                f"(status {resp.status_code})"
            )
        if resp.status_code >= 500:
            raise RemoteUnreachable(f"remote {self.card.endpoint} returned {resp.status_code}")

        try:
            rpc_resp = JsonRpcResponse.model_validate(resp.json())
        except (ValueError, TypeError) as exc:
            raise RemoteMalformed(f"remote returned malformed JSON-RPC response: {exc}") from exc

        if rpc_resp.error is not None:
            raise RemoteProtocolError(
                f"remote error {rpc_resp.error.code}: {rpc_resp.error.message}"
            )

        if rpc_resp.result is None:
            raise RemoteMalformed("remote returned neither result nor error")

        # Phase 4.5a P0 #3: with wire_profile="public" (the default) the server
        # returns the narrow WireAgentResult. With wire_profile="trusted" it
        # returns a full AgentResult dump — and WireAgentResult sets
        # extra="forbid", so parsing that unconditionally raised a raw
        # pydantic ValidationError from OUTSIDE the RemoteMalformed guard,
        # which _dispatch_one then swallowed as TOOL_ERROR. The trusted profile
        # had therefore never worked against voussoir's own client (audit M4).
        # Try the narrow shape first, fall back to the full one, and surface a
        # genuine parse failure as RemoteMalformed like every other wire error.
        from voussoir.a2a.wire import WireAgentResult

        try:
            wire = WireAgentResult.model_validate(rpc_resp.result)
        except pydantic.ValidationError:
            try:
                trusted = AgentResult[str].model_validate(rpc_resp.result)
            except pydantic.ValidationError as exc:
                raise RemoteMalformed(
                    f"remote returned a result matching neither the public "
                    f"(WireAgentResult) nor the trusted (AgentResult) wire "
                    f"shape: {exc}"
                ) from exc
            # Trusted peers are entitled to have their lineage believed, but
            # the local caller still owns its own chain — prepend self.name.
            return trusted.model_copy(
                update={"delegation_chain": [self.name, *trusted.delegation_chain]}
            )

        return AgentResult[str](
            output=wire.output,
            trace_id=f"remote:{self.card.name}",
            steps=[],
            tokens_in=wire.tokens_in,
            tokens_out=wire.tokens_out,
            cost_usd=wire.cost_usd,
            duration_ms=wire.duration_ms,
            delegation_chain=[self.name],
            cascade_history=[],
            guardrail_decisions=[],
            # audit H3: carry the remote run's taint so accumulate_outcomes
            # merges it into the caller's ctx. Without it a remote sub-agent
            # laundered UNTRUSTED taint and unblocked the parent's EXFIL tools.
            taint=set(wire.taint),
            finish_reason=wire.finish_reason,
        )

    def _issue_jwt(self, parent_ctx: AgentContext) -> str:
        provider = parent_ctx.container.resolve(KeyProvider)  # type: ignore[type-abstract]
        now = int(time.time())
        claims = {
            "iss": parent_ctx.agent_name,
            "aud": self.card.name,
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            "sub": parent_ctx.user_id,
            "trace_id": parent_ctx.trace_id,
        }
        # v1.1.0 F9: use jwt_signing_key(alg) so RS256 surfaces a clean
        # JWTKeyNotConfiguredError instead of a cryptic PyJWT InvalidKeyError.
        alg = provider.jwt_algorithm()
        return jwt.encode(
            claims,
            provider.jwt_signing_key(alg),
            algorithm=alg,
        )
