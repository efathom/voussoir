"""URL-allowlist + exfil-pattern guardrails.

Use `URLAllowlist` on input/tool_output to block any URL not in a known-good
host list. Use `ExfilPatternScan` on output to catch suspicious patterns that
look like data-exfiltration vectors (image tags with URLs, etc.).
"""

from __future__ import annotations

import re
from typing import Literal

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict

# Match an http(s) URL and capture only the host. Two forms:
#   - Bracketed IPv6 literal:  `https://[::1]/page`, `https://[2001:db8::1]:8080/`
#     → captured group is the bracketed portion including the brackets, e.g. `[::1]`.
#   - Plain host (DNS / IPv4):  `https://evil.com:8080/page` → captured `evil.com`.
# Optional `:<port>` is consumed but not captured. The host terminates at `:`, `/`,
# `?`, `#`, or whitespace.
#
# IPv6 (the bracketed form) was previously silently allowed because `[\w.-]+` does
# not match `[` or `:`. v1.0.1 restores enforcement.
_URL_RX = re.compile(r"https?://(\[[0-9a-fA-F:]+\]|[\w.-]+)(?::\d+)?(?:[/?#\s]|$)")
_IMAGE_WITH_URL = re.compile(r"!\[.*?\]\(https?://[^)]+\)")


class URLAllowlist:
    """Block URLs whose host is not in the configured allowlist.

    Use this when you want to restrict outbound URL references to a known-good
    set of hosts, blocking potential SSRF or data-exfil via URL parameters.

    `stage` is configurable so the same allowlist can guard both `input`
    (user-supplied URLs) and `tool_output` (URLs returned by external tools).
    `bind_default_guardrails(profile="strict", url_allowlist=...)` registers
    two instances, one per stage.
    """

    name = "url_allowlist"

    def __init__(
        self,
        *,
        allowlist: list[str],
        stage: Literal["input", "tool_call", "tool_output", "output"] = "input",
    ) -> None:
        if not allowlist:
            raise ValueError("URLAllowlist requires a non-empty allowlist")
        self._allowed = set(allowlist)
        self.stage: Literal["input", "tool_call", "tool_output", "output"] = stage

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        for raw_host in _URL_RX.findall(payload.content):
            # Strip IPv6 brackets so operators can write `::1` in the allowlist
            # rather than having to bracket-quote it.
            host = (
                raw_host[1:-1] if raw_host.startswith("[") and raw_host.endswith("]") else raw_host
            )
            if host not in self._allowed:
                return GuardrailVerdict(
                    verdict="BLOCK",
                    reason=f"URL host {host!r} not in allowlist",
                )
        return GuardrailVerdict(verdict="ALLOW")


class ExfilPatternScan:
    """Block final-output content that contains an image-with-URL exfil vector.

    Use this on output-stage screening to catch Markdown image syntax that
    embeds a URL — a common data-exfiltration vector where an attacker encodes
    data in query parameters of an external image request.
    """

    name = "exfil_pattern_scan"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        if _IMAGE_WITH_URL.search(payload.content):
            return GuardrailVerdict(
                verdict="BLOCK",
                reason="output contains image-with-URL (data-exfil vector)",
            )
        return GuardrailVerdict(verdict="ALLOW")
