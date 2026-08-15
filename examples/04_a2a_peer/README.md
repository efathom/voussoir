# 04 — A2A peer demo

Phase 4b: two voussoir processes find each other via the AgentCard well-known endpoint and delegate over JSON-RPC.

## Run

Both processes share a JWT secret. Generate one and export it in both terminals.

```bash
# Terminal 1 + Terminal 2: same secret
export VOUSSOIR_A2A_JWT_SECRET=$(python -c "import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())")
export ANTHROPIC_API_KEY=sk-ant-...
```

Terminal 1 (publisher): the publisher must explicitly allow the caller's
issuer (Phase 4.5a P0 #2: default-deny). The caller agent is named
`lead`, so it mints JWTs with `iss="lead"`.

```bash
export VOUSSOIR_A2A_ALLOWED_ISSUERS=lead
python examples/04_a2a_peer/publisher.py
```

Terminal 2 (caller):

```bash
python examples/04_a2a_peer/caller.py
```

The caller will:
1. Fetch `http://127.0.0.1:8765/.well-known/agent-card.json`
2. Verify the JWS signature against `http://127.0.0.1:8765/.well-known/jwks.json`
3. Construct an `AgentRef` and use it as a delegate from a local lead agent
4. Print the lead's final output and the `delegation_chain` (`['lead', 'researcher']`)

## Limitations (Phase 4b)

- The publisher auto-generates an ephemeral RSA-2048 keypair for AgentCard signing. Restart = new identity. Production deployments configure `VOUSSOIR_A2A_CARD_SIGNING_KEY_PATH` to a persistent PEM file.
- Only HS256 JWTs (shared secret). Cross-org deployments need RS256 (v1.5).
- No retry / backoff / rate limiting (Phase 5).
- Full `AgentResult` is sent over the wire including `delegation_chain` and `steps` (Phase 5 redacts).
