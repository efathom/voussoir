from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    email: str | None = None
    auth_method: Literal["sso", "service", "api_key", "anonymous"] = "service"
    roles: list[str] = Field(default_factory=list)
    teams: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    classification: Literal["public", "internal", "confidential", "restricted"] = "internal"
    attributes: dict[str, Any] = Field(default_factory=dict)
    issued_at: datetime
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at < datetime.now(UTC)


def default_principal() -> Principal:
    """Use this as the `Principal | None` fallback when none is provided.

    Returns `Principal(user_id="system", issued_at=now_utc)` — a non-authenticated
    sentinel for runs where the caller didn't supply a principal.
    """
    return Principal(user_id="system", issued_at=datetime.now(UTC))
