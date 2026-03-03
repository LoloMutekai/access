"""
A.C.C.E.S.S. — Tool Registry

Central store for all registered tools.

Design:
- Flat dict: name → tool instance
- No lazy loading — all tools are registered at startup
- Raises on duplicate registration (fail-fast, no silent overwrite)
- Thread-safe for reads (concurrent tool lookups are fine)
- get_prompt_section() generates the tool catalog for system prompt injection

Usage:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(GetDateTimeTool())

    tool = registry.get("echo")          # raises if not found
    tool = registry.get_or_none("echo")  # returns None if not found
    all_tools = registry.list()
    prompt_section = registry.get_prompt_section()
"""

from __future__ import annotations

import logging
from typing import Optional

from .base_tool import BaseTool, ToolError, ToolProtocol

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry mapping tool names to tool instances.

    All registered tools must satisfy ToolProtocol.
    Duplicate names raise immediately — no silent overwrite.
    """

    def __init__(self):
        self._tools: dict[str, object] = {}   # name → tool (duck-typed)

    def register(self, tool: object, overwrite: bool = False) -> None:
        """
        Register a tool.

        Args:
            tool:      Any object satisfying ToolProtocol.
            overwrite: If True, silently replaces existing tool with same name.
                       If False (default), raises on name collision.

        Raises:
            ValueError: If the tool does not satisfy ToolProtocol.
            ValueError: If a tool with the same name is already registered (unless overwrite=True).
        """
        if not isinstance(tool, ToolProtocol):
            raise ValueError(
                f"Object {tool!r} does not satisfy ToolProtocol. "
                f"Must have: name, description, schema, execute()."
            )

        name = tool.name
        if not name or not isinstance(name, str):
            raise ValueError(f"Tool name must be a non-empty string, got: {name!r}")

        if name in self._tools and not overwrite:
            raise ValueError(
                f"Tool '{name}' is already registered. "
                f"Use overwrite=True to replace."
            )

        self._tools[name] = tool
        logger.info(f"ToolRegistry: registered '{name}'")

    def get(self, name: str) -> object:
        """
        Retrieve a tool by name.

        Raises:
            ToolError: If no tool with this name is registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = sorted(self._tools.keys())
            raise ToolError(
                f"Unknown tool: '{name}'. "
                f"Registered tools: {known}"
            )
        return tool

    def get_or_none(self, name: str) -> Optional[object]:
        """Return the tool or None if not found."""
        return self._tools.get(name)

    def list(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def all_tools(self) -> list[object]:
        """Return all registered tool objects (sorted by name)."""
        return [self._tools[k] for k in self.list()]

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        removed = self._tools.pop(name, None)
        if removed is not None:
            logger.info(f"ToolRegistry: unregistered '{name}'")

    def get_prompt_section(self) -> str:
        """
        Generate a tool catalog section for injection into the system prompt.

        Format:
            AVAILABLE TOOLS:
            You may call one tool per response using this exact JSON format:
            {"tool_call": {"name": "<tool_name>", "args": {<args>}}}

            Tool: echo
              Description: ...
              Args: ...

            Tool: get_datetime
              ...
        """
        if not self._tools:
            return ""

        header = (
            "AVAILABLE TOOLS:\n"
            "To use a tool, respond with ONLY this JSON (no other text):\n"
            '{"tool_call": {"name": "<tool_name>", "args": {<args>}}}\n'
            "\nRegistered tools:\n"
        )

        tool_blocks = []
        for name in self.list():
            tool = self._tools[name]
            if hasattr(tool, "to_prompt_description"):
                tool_blocks.append(tool.to_prompt_description())
            else:
                desc = getattr(tool, "description", "No description.")
                tool_blocks.append(f"Tool: {name}\n  Description: {desc}")

        return header + "\n\n".join(tool_blocks)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list()})"