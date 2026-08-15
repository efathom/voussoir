"""voussoir.auth — Two-axis authentication and authorization for agent tool calls.

Axis 1 (AuthN) — CredentialBroker resolves credentials for a tool's declared
AuthRequirement before the tool is invoked. Credentials are injected into
ToolContext and never exposed to the LLM.

Axis 2 (AuthZ) — Authorizer decides ALLOW / DENY / MASK for (principal, tool, args)
before capability and taint checks. MASK records dotted field paths that are
redacted from the tool output before the result reaches the model.

Public API:
  Principal, default_principal   — identity behind the agent run
  AuthRequirement, AuthType      — credential requirement declared on @tool
  Credentials                    — resolved credential bundle (headers, cert, expiry)
  CredentialBroker               — Protocol for resolving credentials
  Authorizer                     — Protocol for authorization decisions
  AuthzDecision                  — result of Authorizer.authorize()
  AuthError, AuthenticationFailedError, MissingCredentialError — exception hierarchy
  JWTPrincipalMapper             — Protocol for mapping A2A inbound JWTs to Principal
  PrincipalForwarder             — Protocol for forwarding Principal on A2A outbound calls
  DefaultJWTPrincipalMapper      — standard-claims implementation (sub, email, groups)
  SameOrgPassthroughForwarder    — forwards Principal JWT unchanged (same-org deployments)

  Concrete Authorizer implementations:
  AllowAllAuthorizer             — permits every (principal, tool, args) triple
  DenyByDefaultAuthorizer        — fail-closed; denies every triple (zero-trust)
  ChainedAuthorizer              — composes multiple Authorizers (first DENY/MASK wins)
  DomainAuthorizer               — restricts access by principal domain / org
  RoleAuthorizer                 — restricts access by principal role membership

  Concrete CredentialBroker implementations:
  ChainedCredentialBroker        — tries brokers in order, returns first match
  EnvCredentialBroker            — resolves credentials from environment variables
  FileCredentialBroker           — resolves credentials from a secrets file
  KeychainCredentialBroker       — resolves credentials from the OS keychain
  MTLSCredentialBroker           — resolves mTLS client-cert credentials
  OAuth2CredentialBroker         — resolves OAuth 2 bearer-token credentials
"""

from voussoir.auth.a2a import (
    DefaultJWTPrincipalMapper,
    JWTPrincipalMapper,
    PrincipalForwarder,
    SameOrgPassthroughForwarder,
)
from voussoir.auth.authorizers import (
    AllowAllAuthorizer,
    ChainedAuthorizer,
    DenyByDefaultAuthorizer,
    DomainAuthorizer,
    RoleAuthorizer,
)
from voussoir.auth.brokers import (
    ChainedCredentialBroker,
    EnvCredentialBroker,
    FileCredentialBroker,
    KeychainCredentialBroker,
    MTLSCredentialBroker,
    OAuth2CredentialBroker,
)
from voussoir.auth.credentials import AuthRequirement, AuthType, Credentials
from voussoir.auth.decision import AuthzDecision
from voussoir.auth.errors import AuthenticationFailedError, AuthError, MissingCredentialError
from voussoir.auth.principal import Principal, default_principal
from voussoir.auth.protocol import Authorizer, CredentialBroker

__all__ = [
    "AllowAllAuthorizer",
    "AuthError",
    "AuthRequirement",
    "AuthType",
    "AuthenticationFailedError",
    "Authorizer",
    "AuthzDecision",
    "ChainedAuthorizer",
    "ChainedCredentialBroker",
    "CredentialBroker",
    "Credentials",
    "DefaultJWTPrincipalMapper",
    "DenyByDefaultAuthorizer",
    "DomainAuthorizer",
    "EnvCredentialBroker",
    "FileCredentialBroker",
    "JWTPrincipalMapper",
    "KeychainCredentialBroker",
    "MTLSCredentialBroker",
    "MissingCredentialError",
    "OAuth2CredentialBroker",
    "Principal",
    "PrincipalForwarder",
    "RoleAuthorizer",
    "SameOrgPassthroughForwarder",
    "default_principal",
]
