# Security Policy

## Reporting a Vulnerability

We take security issues seriously. If you believe you have found a security
vulnerability in voussoir, please report it responsibly rather than opening a
public issue.

**Do not** open a GitHub issue or PR with the details. Instead, report it
privately via one of:

- **GitHub**: use the repository's *Private vulnerability reporting* feature
  (Security → Report a vulnerability), or
- **Email**: `security@efathom.com`

Please include:

- A clear description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept if available
- The affected version(s) / commit(s)
- Any suggested remediation

## What to expect

- You will receive an acknowledgment within a few business days.
- We will validate the report and keep you informed of our assessment and fix
  timeline.
- Once a fix is available we will publish a security advisory and credit the
  reporter (unless you prefer to remain anonymous).

## Supported versions

| Version | Supported |
|---------|-----------|
| latest `main` | :white_check_mark: |
| latest tagged release | :white_check_mark: |

## Security model at a glance

voussoir's security posture is structural, not opt-in:

- Every tool carries a `Capability` mask and produces a `Trust` taint; the
  `StandardExecutor` enforces both inline — there is no skip flag.
- The `Authorizer` / `CredentialBroker` Protocols implement authz/authn; the
  default `Authorizer` is the fail-closed `DenyByDefaultAuthorizer` — bind a
  concrete `Authorizer` (Role / Domain / Chained) to grant tool access.
- The `Guardrail` chain screens input / tool_call / tool_output / output
  stages. `default_container()` now binds the `standard` profile; use
  `bind_default_guardrails(profile="strict", url_allowlist=[...])` for
  PII + URL allow-listing.
- Security-critical container bindings (`Authorizer`, `KeyProvider`,
  `ITelemetrySink`, `ILLMProvider`, `IMemoryStore`, `ISessionStore`,
  `IToolExecutor`) are frozen after `default_container()` returns so plugins
  cannot rebind them.

## Security best practices for deployments

- **Never commit real credentials** — use environment variables or a secret
  manager for `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, the A2A JWT secret, and
  database passwords.
- **Set `VOUSSOIR_A2A_JWT_SECRET` and `VOUSSOIR_A2A_ALLOWED_ISSUERS`** for any
  A2A peer deployment. Empty issuer allow-list is default-deny.
- **Bind a concrete `Authorizer`** (Role / Domain / Chained) before exposing
  tool-using agents.
- **Enable the `strict` guardrail profile** and configure a URL allow-list if
  user data or external URLs flow through your agents.
- **Use JSON logs** (`VOUSSOIR_LOG_FORMAT=json`) and export OpenTelemetry
  traces + metrics to your observability backend.
- **Pin and review dependencies** — Dependabot updates are enabled, but review
  them before merging.
