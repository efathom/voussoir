"""Built-in CredentialBroker implementations (Phase 6 A7).

Use these as a starting point. Compose via ChainedCredentialBroker for
typical production setups: env → mtls → file → keychain.
"""

from voussoir.auth.brokers.chained import ChainedCredentialBroker
from voussoir.auth.brokers.env import EnvCredentialBroker
from voussoir.auth.brokers.file import FileCredentialBroker
from voussoir.auth.brokers.keychain import KeychainCredentialBroker
from voussoir.auth.brokers.mtls import MTLSCredentialBroker
from voussoir.auth.brokers.oauth2 import OAuth2CredentialBroker

__all__ = [
    "ChainedCredentialBroker",
    "EnvCredentialBroker",
    "FileCredentialBroker",
    "KeychainCredentialBroker",
    "MTLSCredentialBroker",
    "OAuth2CredentialBroker",
]
