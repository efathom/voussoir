"""voussoir.tools — Tool Protocol, @tool decorator, Capability scopes, and ToolRegistry.

Public API:
  Capability             — IntFlag enum declaring what a tool is allowed to do
  Tool                   — Protocol every tool must satisfy
  ToolContext            — context injected into each tool invocation (principal, credentials, taint)
  ToolRegistry           — name-based tool resolution used by the executor
  tool                   — @tool decorator that wraps a Python function as a Tool
  parse_capability_list  — helper to convert yaml/env capability name lists to a Capability flag
"""

from voussoir.tools.decorator import tool
from voussoir.tools.protocol import Capability, Tool, ToolContext, parse_capability_list
from voussoir.tools.registry import ToolRegistry

__all__ = [
    "Capability",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "parse_capability_list",
    "tool",
]
