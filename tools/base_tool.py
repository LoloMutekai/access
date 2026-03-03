"""
A.C.C.E.S.S. — Tool Base

Defines the contracts all tools must satisfy.

Design decisions:
- ToolProtocol (structural typing) — any object with name/description/schema/execute
  satisfies it. No forced inheritance.
- BaseTool (abstract class) — convenience base for rapid tool creation.
  Implements boilerplate, enforces execute() signature.
- ToolResult — immutable. Carries output + metadata for logging and injection.
- ToolError — raised when execution fails. Always carries original exception.

Security constraints:
- No subprocess, no os.system, no eval, no exec
- No direct file writes outside designated sandbox paths
- All tools must be explictly registered — no auto-discovery
- Tools receive only the args the LLM passes — no implicit context injection

Tool call format (what the LLM must produce to trigger tool use):
    {"tool_call": {"name": "<tool_name>", "args": {"key": "value", ...}}}

Tool result injection (added to messages for second LLM pass):
    {"role": "tool", "name": "<tool_name>", "content": "<result_json>"}
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT & ERROR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolResult:
    """
    Immutable output of a tool execution.

    success:     True if execution completed without errors.
    output:      Serializable result dict — injected into LLM as JSON string.
    tool_name:   Name of the tool that produced this result.
    error_msg:   If success=False, human-readable error description.
    latency_ms:  Execution time in milliseconds.
    """
    tool_name: str
    success: bool
    output: dict
    error_msg: Optional[str] = None
    latency_ms: float = 0.0

    def to_message_content(self) -> str:
        """
        Serialized form injected into the LLM as the tool response.
        Always valid JSON string.
        """
        if self.success:
            return json.dumps(self.output, ensure_ascii=False)
        return json.dumps({
            "error": self.error_msg or "Unknown tool error.",
            "tool": self.tool_name,
        }, ensure_ascii=False)

    def to_log_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "output_keys": list(self.output.keys()) if self.success else [],
            "error": self.error_msg,
        }

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"ToolResult({status} {self.tool_name}, "
            f"latency={self.latency_ms:.0f}ms)"
        )


class ToolError(Exception):
    """
    Raised when a tool fails during execution.

    Wraps the original exception so the ToolDispatcher can catch a single
    exception type and produce a ToolResult(success=False).
    """

    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.original = original


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL — structural contract
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class ToolProtocol(Protocol):
    """
    Structural typing contract for any tool.

    Any class with these attributes/methods satisfies this protocol
    without needing to inherit from BaseTool.
    """

    @property
    def name(self) -> str:
        """Unique identifier. Used by LLM to call the tool."""
        ...

    @property
    def description(self) -> str:
        """Natural language description. Injected into the system prompt."""
        ...

    @property
    def schema(self) -> dict:
        """
        JSON Schema (draft-07) for the args dict.

        Example:
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        """
        ...

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the tool with the provided arguments.

        Args:
            **kwargs: Arguments matching the schema.

        Returns:
            dict — serializable result. Will be JSON-encoded and injected as LLM message.

        Raises:
            ToolError: on validation or execution failure.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# BASE TOOL — convenience class, not required
# ─────────────────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    Abstract base class for tools. Provides boilerplate.

    Subclass this for convenience:

        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def description(self) -> str:
                return "Does something useful."

            @property
            def schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                }

            def execute(self, input: str, **kwargs) -> dict:
                return {"result": input.upper()}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def schema(self) -> dict:
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict:
        ...

    def validate_args(self, kwargs: dict) -> None:
        """
        Optional basic validation against required fields.
        Does NOT do full JSON Schema validation (no jsonschema dep in core).
        Override for custom validation.
        """
        required = self.schema.get("required", [])
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ToolError(
                f"Tool '{self.name}' missing required args: {missing}"
            )

    def safe_execute(self, **kwargs: Any) -> dict:
        """
        Validate + execute with error wrapping.
        Used internally by ToolDispatcher.
        """
        self.validate_args(kwargs)
        return self.execute(**kwargs)

    def to_prompt_description(self) -> str:
        """
        Human-readable tool description for injection into system prompt.
        """
        props = self.schema.get("properties", {})
        args_desc = ", ".join(
            f"{k}: {v.get('type', 'any')} — {v.get('description', '')}"
            for k, v in props.items()
        )
        return (
            f"Tool: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Args: {args_desc or 'none'}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"