"""
A.C.C.E.S.S. — Tools Module

Provides the tool-use infrastructure for AgentCore.

Quick start:
    from tools import ToolRegistry, EchoTool, GetDateTimeTool, CalculatorTool

    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(GetDateTimeTool())
    registry.register(CalculatorTool())

    # Pass to AgentCore via AgentConfig or directly
    # (see AgentCore docs for integration)

Tool call format (LLM must produce exactly this JSON):
    {"tool_call": {"name": "<tool_name>", "args": {<args>}}}

Custom tool example:
    from tools import BaseTool, ToolError

    class MyTool(BaseTool):
        @property
        def name(self) -> str: return "my_tool"

        @property
        def description(self) -> str: return "Does something."

        @property
        def schema(self) -> dict:
            return {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }

        def execute(self, input: str, **kwargs) -> dict:
            return {"result": input.upper()}

    registry.register(MyTool())
"""

from .base_tool import BaseTool, ToolProtocol, ToolResult, ToolError
from .tool_registry import ToolRegistry
from .tool_dispatcher import ToolDispatcher, detect_tool_call, ParsedToolCall, DispatchResult
from .builtins import EchoTool, GetDateTimeTool, CalculatorTool

__all__ = [
    "BaseTool",
    "ToolProtocol",
    "ToolResult",
    "ToolError",
    "ToolRegistry",
    "ToolDispatcher",
    "detect_tool_call",
    "ParsedToolCall",
    "DispatchResult",
    "EchoTool",
    "GetDateTimeTool",
    "CalculatorTool",
]