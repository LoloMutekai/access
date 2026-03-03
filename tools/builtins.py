"""
A.C.C.E.S.S. — Built-in Tools

Production-safe tools included with the framework.

EchoTool:       Returns its input unchanged. Ideal for testing and debugging.
GetDateTimeTool: Returns the current UTC datetime. No IO, pure computation.
CalculatorTool: Evaluates simple arithmetic using ast.literal_eval. No eval/exec.

Security note on CalculatorTool:
    Uses the `ast` module to parse the expression, then evaluates only
    literals and arithmetic operators. Does NOT allow:
    - Function calls
    - Variable references
    - Import statements
    - String operations
    Any forbidden construct raises ToolError immediately.
"""

from __future__ import annotations

import ast
import operator
import logging
from datetime import datetime, timezone
from typing import Any

from .base_tool import BaseTool, ToolError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ECHO TOOL — testing and debugging
# ─────────────────────────────────────────────────────────────────────────────

class EchoTool(BaseTool):
    """
    Returns its input text unchanged.

    Use case: testing the tool-use pipeline end-to-end without side effects.
    Also useful as a "no-op" confirmation tool.

    Example LLM call:
        {"tool_call": {"name": "echo", "args": {"text": "Hello world"}}}

    Response:
        {"echoed": "Hello world"}
    """

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the input text back unchanged. Useful for testing."

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to echo back."
                }
            },
            "required": ["text"]
        }

    def execute(self, text: str = "", **kwargs: Any) -> dict:
        logger.debug(f"EchoTool: echoing {len(text)} chars")
        return {"echoed": text}


# ─────────────────────────────────────────────────────────────────────────────
# GET DATETIME TOOL — no IO, pure computation
# ─────────────────────────────────────────────────────────────────────────────

class GetDateTimeTool(BaseTool):
    """
    Returns the current UTC datetime.

    Pure computation — no IO, no side effects.

    Example LLM call:
        {"tool_call": {"name": "get_datetime", "args": {}}}

    Response:
        {
            "utc_datetime": "2026-03-01T12:00:00+00:00",
            "utc_date": "2026-03-01",
            "utc_time": "12:00:00",
            "day_of_week": "Sunday"
        }
    """

    @property
    def name(self) -> str:
        return "get_datetime"

    @property
    def description(self) -> str:
        return (
            "Returns the current UTC date and time. "
            "Use when the user asks about the current time, date, or day of week."
        )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def execute(self, **kwargs: Any) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "utc_datetime": now.isoformat(),
            "utc_date": now.strftime("%Y-%m-%d"),
            "utc_time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATOR TOOL — safe arithmetic via AST
# ─────────────────────────────────────────────────────────────────────────────

# Allowed operators for the safe evaluator
_ALLOWED_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod:  operator.mod,
}

_MAX_EXPRESSION_LENGTH = 200


def _safe_eval(node: ast.AST) -> float:
    """
    Recursively evaluate an AST node using only allowed operators.
    Raises ToolError for any unsupported construct.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ToolError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        try:
            result = _ALLOWED_OPS[op_type](left, right)
        except ZeroDivisionError:
            raise ToolError("Division by zero.")
        except OverflowError:
            raise ToolError("Arithmetic overflow.")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ToolError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPS[op_type](operand)

    raise ToolError(
        f"Unsupported expression construct: {type(node).__name__}. "
        f"Only basic arithmetic is allowed."
    )


class CalculatorTool(BaseTool):
    """
    Evaluates basic arithmetic expressions safely via the ast module.

    Supported: +, -, *, /, **, % and parentheses.
    NOT supported: functions, variables, strings, imports, comparisons.

    Example LLM call:
        {"tool_call": {"name": "calculator", "args": {"expression": "2 ** 10 + 5 * 3"}}}

    Response:
        {"expression": "2 ** 10 + 5 * 3", "result": 1039.0}
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluates arithmetic expressions. "
            "Supports: +, -, *, /, **, % and parentheses. "
            "Example: '2 ** 10 + 5 * 3'"
        )

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate. No variables, no functions."
                }
            },
            "required": ["expression"]
        }

    def execute(self, expression: str = "", **kwargs: Any) -> dict:
        expression = expression.strip()

        if len(expression) > _MAX_EXPRESSION_LENGTH:
            raise ToolError(
                f"Expression too long ({len(expression)} chars, max {_MAX_EXPRESSION_LENGTH})."
            )
        if not expression:
            raise ToolError("Expression cannot be empty.")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ToolError(f"Invalid expression syntax: {e}")

        result = _safe_eval(tree.body)
        logger.debug(f"CalculatorTool: {expression!r} = {result}")

        return {
            "expression": expression,
            "result": result,
        }