"""v1.0.4 E7 — concrete authorizers + brokers re-exported from voussoir.auth.

Round-3 Architecture review (I2/I7): the framework's _bind_default_authorizer
warning tells users to "Bind a concrete Authorizer (RoleAuthorizer,
DomainAuthorizer, ChainedAuthorizer)" but those names weren't reachable from
voussoir.auth — must use the sub-package paths. E7 surfaces them at the top
of the voussoir.auth public namespace.
"""

from __future__ import annotations


def test_concrete_authorizers_importable_from_voussoir_auth():
    """All four concrete Authorizer implementations are importable from voussoir.auth."""
    from voussoir.auth import (
        AllowAllAuthorizer,
        ChainedAuthorizer,
        DomainAuthorizer,
        RoleAuthorizer,
    )

    # Sanity: each has the .name attribute the Protocol requires
    assert AllowAllAuthorizer.name == "allow_all"
    assert RoleAuthorizer.name == "role"
    assert DomainAuthorizer.name == "domain"
    assert ChainedAuthorizer.name == "chained"


def test_concrete_brokers_importable_from_voussoir_auth():
    """All six concrete CredentialBroker implementations are importable from voussoir.auth."""
    from voussoir.auth import (
        ChainedCredentialBroker,
        EnvCredentialBroker,
        FileCredentialBroker,
        KeychainCredentialBroker,
        MTLSCredentialBroker,
        OAuth2CredentialBroker,
    )

    assert EnvCredentialBroker.name == "env"
    assert FileCredentialBroker.name == "file"
    assert MTLSCredentialBroker.name == "mtls"
    assert KeychainCredentialBroker.name == "keychain"
    assert OAuth2CredentialBroker.name == "oauth2"
    assert ChainedCredentialBroker.name == "chained"


def test_authorizers_and_brokers_in_voussoir_auth_all():
    """The concrete impls are in voussoir.auth.__all__ — discoverable via dir() / Pyright."""
    import voussoir.auth as auth_mod

    expected_authorizers = {
        "AllowAllAuthorizer",
        "RoleAuthorizer",
        "DomainAuthorizer",
        "ChainedAuthorizer",
    }
    expected_brokers = {
        "EnvCredentialBroker",
        "FileCredentialBroker",
        "MTLSCredentialBroker",
        "KeychainCredentialBroker",
        "OAuth2CredentialBroker",
        "ChainedCredentialBroker",
    }
    declared_all = set(auth_mod.__all__)
    missing_authz = expected_authorizers - declared_all
    missing_broker = expected_brokers - declared_all
    assert not missing_authz, f"Authorizers missing from voussoir.auth.__all__: {missing_authz}"
    assert not missing_broker, f"Brokers missing from voussoir.auth.__all__: {missing_broker}"
