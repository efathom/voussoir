"""URL-allowlist + exfil-pattern guardrails.

Use `URLAllowlist` on input/tool_output to block any URL not in a known-good
host list. Use `ExfilPatternScan` on output to catch suspicious patterns that
look like data-exfiltration vectors (image tags with URLs, etc.).
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from voussoir.guardrails.protocol import GuardrailPayload, GuardrailVerdict

# Find URL-looking spans; the HOST is then extracted by urlsplit, not by this
# regex. Hand-parsing the host inside the pattern produced three bypasses, all
# of which ended in ALLOW because a non-match yields no finding at all
# (audit H6):
#   - `https://good.com@evil.com/`  — userinfo wasn't modelled
#   - `HTTPS://EVIL.COM/`           — the pattern was case-sensitive
#   - `https://evil.com\steal`      — browsers treat `\` as `/`, the regex didn't
# It also rejected legitimate `https://GOOD.com/` because the captured host was
# compared case-sensitively. So: match loosely, parse strictly, fail closed.
#
# The span ends at whitespace, `<`, `>`, `"`, `'`, or a backtick — everything
# else (including `\`) belongs to the URL as a browser would read it.
_URL_SPAN_RX = re.compile(r"(?i:https?)://[^\s<>\"\'`]+")
_IMAGE_WITH_URL = re.compile(r"!\[.*?\]\(https?://[^)]+\)")
# `<img src=...>` is the same exfil vector in HTML dress; the markdown-only
# pattern above let it through untouched (audit, minor).
_HTML_IMG_WITH_URL = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"\']?https?://", re.I)


def _hosts_in(text: str) -> list[str | None]:
    """Extract the hostname of every URL-looking span in *text*.

    A span that fails to parse, or parses without a hostname, yields ``None``
    so the caller can fail closed on it rather than silently allowing it.
    Hostnames come back lowercased — `urlsplit().hostname` already case-folds
    and strips userinfo and port.
    """
    out: list[str | None] = []
    for span in _URL_SPAN_RX.findall(text):
        # A URL at the end of a sentence usually swallows the punctuation.
        candidate = span.rstrip(".,;:!?)]}")
        try:
            host = urlsplit(candidate).hostname
        except ValueError:
            out.append(None)
            continue
        out.append(host)
    return out


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
        # Case-folded on both sides: hostnames are case-insensitive, and
        # comparing them case-sensitively rejected `https://GOOD.com/` while a
        # bypass let `HTTPS://EVIL.COM/` through. Brackets are stripped so an
        # operator can write `::1` rather than `[::1]`.
        self._allowed = {h.strip().strip("[]").lower() for h in allowlist}
        self.stage: Literal["input", "tool_call", "tool_output", "output"] = stage

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        for host in _hosts_in(payload.content):
            if host is None:
                # Looked like a URL, didn't parse. Fail closed — matching the
                # rest of the framework's posture, and closing the hole where an
                # unparseable URL produced no finding and was therefore allowed.
                return GuardrailVerdict(
                    verdict="BLOCK",
                    reason="content contains an unparseable URL",
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
    data in query parameters of an external image request. The HTML form,
    `<img src="https://...">`, is caught too.
    """

    name = "exfil_pattern_scan"
    stage: Literal["input", "tool_call", "tool_output", "output"] = "output"

    async def screen(self, payload: GuardrailPayload, ctx: object) -> GuardrailVerdict:
        del ctx
        if _IMAGE_WITH_URL.search(payload.content) or _HTML_IMG_WITH_URL.search(payload.content):
            return GuardrailVerdict(
                verdict="BLOCK",
                reason="output contains image-with-URL (data-exfil vector)",
            )
        return GuardrailVerdict(verdict="ALLOW")
