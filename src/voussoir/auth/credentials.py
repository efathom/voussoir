from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthType(StrEnum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    MTLS = "mtls"
    OAUTH2 = "oauth2"
    SERVICE_TOKEN = "service_token"


class AuthRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_type: AuthType
    service: str | None = None
    scopes: list[str] = Field(default_factory=list)
    refresh: bool = True
    cache_ttl_s: int = 0


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_type: AuthType
    headers: dict[str, str] = Field(default_factory=dict)
    cert: tuple[Path, Path] | None = None
    ca_bundle: Path | None = None
    expires_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, buffer_s: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > (self.expires_at - buffer_s)
