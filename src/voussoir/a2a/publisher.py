"""A2A publisher — make_a2a_router (card + JWKS + JSON-RPC) + serve_a2a.

Phase 4b / Task 3 adds the card + JWKS routes. Task 4 adds the JSON-RPC
endpoint. Task 5 adds the serve_a2a uvicorn helper.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit

import jwt
import uvicorn
from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from voussoir.a2a.card import card_from_agent
from voussoir.a2a.errors import A2AErrorCode
from voussoir.a2a.keys import (
    InboundJWTVerificationNotConfiguredError,
    KeyProvider,
    NoCardSigningKeyError,
)
from voussoir.a2a.transport import (
    DelegateParams,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
)
from voussoir.agent.policy import PolicyViolationError
from voussoir.auth.a2a import DefaultJWTPrincipalMapper, JWTPrincipalMapper

# Phase 4.5a P1 #15: JWT algorithm allow-list hardcoded here so an attacker
# with env-var write can't downgrade VOUSSOIR_A2A_JWT_ALGORITHM to "none" or
# a deprecated variant.
_ALLOWED_JWT_ALGORITHMS = frozenset({"HS256", "HS384", "HS512", "RS256"})

if TYPE_CHECKING:
    from pydantic import HttpUrl

    from voussoir.agent.agent import Agent


class _SlidingWindowLimiter:
    """In-process sliding-window rate limiter keyed by client identity.

    A per-process limiter is a first line of defense only; front it with a
    gateway (nginx / Envoy) for real per-host rate limiting in production.
    """

    def __init__(self, max_requests: int, window_s: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_s
        self._hits: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            hits = self._hits.setdefault(key, [])
            while hits and now - hits[0] > self._window:
                hits.pop(0)
            if len(hits) >= self._max:
                return False
            hits.append(now)
            # Bounded cleanup so a long-lived process with many unique clients
            # doesn't grow without limit.
            if len(self._hits) > 10_000:
                self._hits = {
                    k: v for k, v in self._hits.items() if v and now - v[-1] <= self._window
                }
            return True


def make_a2a_router(
    agent: Agent,
    *,
    endpoint: HttpUrl | str,
    wire_profile: Literal["public", "trusted"] = "public",
    principal_mapper: JWTPrincipalMapper | None = None,
    rate_limit_per_minute: int | None = 60,
    max_request_bytes: int = 1_000_000,
) -> APIRouter:
    """Build a FastAPI router exposing `agent` over A2A.

    Routes mounted:
      GET /.well-known/agent-card.json   signed AgentCard (JWS Compact or JSON)
      GET /.well-known/jwks.json         publisher's public JWKS
      POST /a2a                          JSON-RPC endpoint (Task 4)

    `endpoint` is the URL the card advertises as its invocation endpoint.

    `principal_mapper` (Phase 6 A9): converts the verified JWT claims into a
    Principal that is threaded through to `agent.run(principal=...)`. Defaults
    to DefaultJWTPrincipalMapper (sub -> user_id, email -> email, groups -> roles).
    """
    mapper: JWTPrincipalMapper = principal_mapper or DefaultJWTPrincipalMapper()
    router = APIRouter()
    limiter = _SlidingWindowLimiter(rate_limit_per_minute) if rate_limit_per_minute else None
    endpoint_url = str(endpoint)

    @router.get("/.well-known/health")
    async def get_health() -> Response:
        """Liveness/readiness probe — no auth, no side effects."""
        return JSONResponse(content={"status": "ok"})

    # v1.0.4 E2 / C5: derive the canonical jwks_uri from the endpoint's
    # origin so the card published here points at the JWKS route registered
    # immediately below. Pre-E2 the card's jwks_uri was always None, and
    # consumers fell back to "{base}/.well-known/jwks.json" derived from the
    # card's endpoint — which silently breaks when the router is mounted
    # under a path prefix (e.g. /api/v1/a2a). Now we publish the URL
    # explicitly. NOTE: we attach the JWKS route under the FastAPI router's
    # root, so the well-known path is always relative to the router mount
    # origin — i.e. scheme + netloc + "/.well-known/jwks.json".
    _parsed = urlsplit(endpoint_url)
    jwks_uri_url = urlunsplit((_parsed.scheme, _parsed.netloc, "/.well-known/jwks.json", "", ""))

    @router.get("/.well-known/agent-card.json")
    async def get_card(request: Request, accept: str | None = Header(default=None)) -> Response:
        del request  # not used; consumers may stuff trace headers in future.
        provider = agent.container.resolve(KeyProvider)  # type: ignore[type-abstract]
        card = card_from_agent(
            agent,
            endpoint=endpoint_url,  # type: ignore[arg-type]
            jwks_uri=jwks_uri_url,  # type: ignore[arg-type]
        )
        # v1.0.4 E2 / C3: drop `exp` from the signed payload when it is
        # None, otherwise PyJWT's verify_exp validator (in discover_card)
        # chokes on `int(None)`. Including `exp: null` would be semantically
        # equivalent to "no expiry" but is incompatible with the standard
        # JWT claim semantics — better to omit the claim entirely.
        payload = card.model_dump(mode="json", exclude_none=True)
        try:
            private_jwk = provider.card_signing_jwk()
        except NoCardSigningKeyError:
            # Dev mode: return unsigned card as JSON.
            return JSONResponse(
                content=payload,
                headers={"Cache-Control": "max-age=60, dev-mode"},
            )

        # RSAAlgorithm.from_jwk returns RSAPrivateKey | RSAPublicKey; with a
        # private JWK (contains `d`) it's always RSAPrivateKey.
        signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(private_jwk)
        compact = jwt.encode(
            payload,
            signing_key,  # type: ignore[arg-type]
            algorithm="RS256",
            headers={"typ": "agent-card+jwt", "kid": private_jwk["kid"]},
        )

        if accept and "application/json" in accept and "application/jose" not in accept:
            # JSON serialization form for human inspection.
            header_b64, payload_b64, sig_b64 = compact.split(".")
            return JSONResponse(
                content={
                    "protected": header_b64,
                    "payload": payload_b64,
                    "signature": sig_b64,
                },
                headers={"Cache-Control": "max-age=60"},
            )

        return Response(
            content=compact,
            media_type="application/jose",
            headers={"Cache-Control": "max-age=60"},
        )

    @router.get("/.well-known/jwks.json")
    async def get_jwks() -> Response:
        provider = agent.container.resolve(KeyProvider)  # type: ignore[type-abstract]
        try:
            jwks = provider.card_verification_jwks()
        except NoCardSigningKeyError:
            jwks = {"keys": []}
        return JSONResponse(
            content=jwks,
            headers={"Cache-Control": "max-age=300"},
        )

    @router.post("/a2a")
    async def post_a2a(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        # 0. Rate limit (before auth/body work so unauthenticated floods can't
        #    consume JWT verification or body parsing). Keyed by peer address.
        if limiter is not None:
            client_key = request.client.host if request.client else "unknown"
            if not await limiter.allow(client_key):
                return Response(
                    status_code=429,
                    content=json.dumps({"error": "rate limit exceeded"}),
                    headers={"Retry-After": "60", "Content-Type": "application/json"},
                )

        # 1. Authn — Bearer token required.
        if not authorization or not authorization.startswith("Bearer "):
            return Response(
                status_code=401,
                content=json.dumps({"error": "missing bearer token"}),
                headers={
                    "WWW-Authenticate": 'Bearer realm="voussoir-a2a"',
                    "Content-Type": "application/json",
                },
            )
        token = authorization.removeprefix("Bearer ").strip()

        provider = agent.container.resolve(KeyProvider)  # type: ignore[type-abstract]
        alg = provider.jwt_algorithm()
        if alg not in _ALLOWED_JWT_ALGORITHMS:
            # Phase 4.5a P1 #15: refuse to serve if configured algorithm isn't
            # on the allow-list. Surfaces as a generic 401 (no detail leak).
            return Response(
                status_code=401,
                content=json.dumps({"error": "server misconfigured: jwt algorithm not allowed"}),
                headers={
                    "WWW-Authenticate": 'Bearer realm="voussoir-a2a"',
                    "Content-Type": "application/json",
                },
            )
        expected = provider.expected_issuers()
        if not expected:
            # Phase 4.5a P0 #2: empty issuer allow-list is default-deny. The
            # prod operator MUST set VOUSSOIR_A2A_ALLOWED_ISSUERS=peerA,peerB.
            return Response(
                status_code=401,
                content=json.dumps({"error": "no allowed issuers configured"}),
                headers={
                    "WWW-Authenticate": 'Bearer realm="voussoir-a2a"',
                    "Content-Type": "application/json",
                },
            )
        # v1.0.4 E2 / C4: dispatch on `alg` via the KeyProvider Protocol.
        # Pre-E2 we passed `provider.jwt_secret()` regardless of algorithm,
        # which silently broke RS256 (InvalidKeyError → caught by the generic
        # InvalidTokenError clause → 401 "invalid bearer token" with no
        # operator diagnostic). Now an unsupported alg/provider combination
        # raises InboundJWTVerificationNotConfiguredError, which we surface as
        # 500 (server misconfiguration), not 401 (client auth failure).
        try:
            verify_key = provider.jwt_verification_key(alg)
        except InboundJWTVerificationNotConfiguredError as exc:
            return Response(
                status_code=500,
                content=json.dumps({"error": f"server misconfigured: {exc}"}),
                headers={"Content-Type": "application/json"},
            )
        try:
            claims = jwt.decode(
                token,
                verify_key,
                algorithms=[alg],
                audience=agent.name,
                issuer=sorted(expected),
                options={"require": ["iss", "exp", "aud", "nbf", "iat", "sub"]},
            )
        except jwt.InvalidTokenError:
            return Response(
                status_code=401,
                content=json.dumps({"error": "invalid bearer token"}),
                headers={
                    "WWW-Authenticate": 'Bearer realm="voussoir-a2a"',
                    "Content-Type": "application/json",
                },
            )

        # 2. Parse JSON-RPC envelope (with a request-size cap).
        content_length = request.headers.get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > max_request_bytes
        ):
            return _rpc_error(None, A2AErrorCode.INVALID_REQUEST, "request too large")
        try:
            body_bytes = await request.body()
            if len(body_bytes) > max_request_bytes:
                return _rpc_error(None, A2AErrorCode.INVALID_REQUEST, "request too large")
            body_obj = json.loads(body_bytes)
            rpc = JsonRpcRequest.model_validate(body_obj)
        except (json.JSONDecodeError, ValueError, TypeError):
            return _rpc_error(None, A2AErrorCode.PARSE_ERROR, "parse error")

        if rpc.method != "voussoir.delegate":
            return _rpc_error(
                rpc.id,
                A2AErrorCode.METHOD_NOT_FOUND,
                f"method {rpc.method!r} not found",
            )

        # 3. Validate params shape.
        try:
            params = DelegateParams.model_validate(rpc.params)
        except ValueError:
            return _rpc_error(
                rpc.id,
                A2AErrorCode.INVALID_PARAMS,
                "params must be {'task': str}",
            )

        # 4. Dispatch — agent.run, not agent.delegate (spec §4.4 step 6).
        # The remote-invoked agent is a fresh top-level run; the caller's
        # delegation lineage stays with the caller.
        session_id = str(claims.get("trace_id") or "a2a-default")
        user_id = str(claims.get("sub") or "a2a-anonymous")
        # Phase 6 A9: map JWT claims to a Principal and pass it into agent.run.
        principal = await mapper.map(claims)
        try:
            result = await agent.run(
                params.task,
                session_id=session_id,
                user_id=user_id,
                principal=principal,
            )
        except PolicyViolationError as exc:
            return _rpc_error(rpc.id, A2AErrorCode.POLICY_VIOLATION, str(exc))
        except Exception:  # noqa: BLE001 — wire boundary, do not leak trace
            return _rpc_error(rpc.id, A2AErrorCode.INTERNAL_ERROR, "internal error")

        # Phase 4.5a P0 #3 + Phase 5 C6: wire response defaults to the
        # redacted WireAgentResult (profile="public"). Pass wire_profile="trusted"
        # to make_a2a_router for intra-org peering where the caller is trusted
        # to see lineage/cost/steps.
        payload = result.to_wire(profile=wire_profile)
        return JSONResponse(
            content=JsonRpcResponse(
                id=rpc.id,
                result=payload.model_dump(mode="json"),
            ).model_dump(mode="json", exclude_none=True)
        )

    return router


def serve_a2a(
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    endpoint_override: str | None = None,
) -> None:
    """Build a FastAPI app, mount the agent's A2A router, run uvicorn.

    Blocking — runs until uvicorn exits. For lifecycle-controlled tests
    (e.g. with `port=0`), use `make_a2a_router` + your own uvicorn server
    invocation.

    `endpoint_override` controls what URL the AgentCard advertises as its
    invocation endpoint. Resolution order:
      1. endpoint_override (highest)
      2. VOUSSOIR_A2A_PUBLIC_URL env var
      3. f"http://{host}:{port}/a2a"
    """
    endpoint = (
        endpoint_override
        or os.environ.get("VOUSSOIR_A2A_PUBLIC_URL")
        or f"http://{host}:{port}/a2a"
    )
    app = FastAPI(title=f"voussoir.a2a:{agent.name}")
    app.include_router(make_a2a_router(agent, endpoint=endpoint))
    uvicorn.run(app, host=host, port=port)


def _rpc_error(request_id: str | int | None, code: int, message: str) -> JSONResponse:
    """Build a JSON-RPC error response (HTTP 200, error in body)."""
    return JSONResponse(
        content=JsonRpcResponse(
            id=request_id,
            error=JsonRpcError(code=code, message=message),
        ).model_dump(mode="json", exclude_none=True)
    )
