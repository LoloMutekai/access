"""
A.C.C.E.S.S. — Tool Dispatcher

Handles the full tool-use loop:
    LLM response → detect tool call → execute → inject result → second LLM pass

Tool Call Detection:
    The LLM must respond with ONLY a JSON object at the top level:
        {"tool_call": {"name": "<tool_name>", "args": {<args_dict>}}}

    Detection is strict:
    - Response must be (after strip) a valid JSON object
    - Must have a "tool_call" key
    - Value must have "name" (str) and "args" (dict) keys
    - Any other response format → no tool call detected

Result Injection:
    After execution, the conversation is extended:
        messages + [
            {"role": "assistant", "content": "<original_tool_call_json>"},
            {"role": "user",      "content": "TOOL_RESULT [{name}]: <result_json>"},
        ]
    Then passed to a second LLM call to produce the final response.

Why user role for tool result?
    Most APIs (including Anthropic) support a tool/function role in multi-turn,
    but for maximum compatibility with FakeLLMClient and future providers,
    we inject the result as a user message with a clear TOOL_RESULT: prefix.
    This is simpler and just as effective for most use cases.
    Phase 3 can migrate to native tool_result roles.

Security:
    - All execution is sandbox-checked via the tool's own validate_args()
    - ToolDispatcher never executes raw code, never calls eval/exec
    - Exceptions during execution produce ToolResult(success=False), not crashes
    - Unknown tools → ToolResult(success=False), graceful continuation

Max iterations:
    To prevent infinite tool-call loops, a configurable max_iterations limit
    stops the loop and returns the last response after N tool calls.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .base_tool import ToolError, ToolResult
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL CALL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedToolCall:
    """Result of parsing an LLM response for a tool call."""
    found: bool
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    raw_json: str = ""
    parse_error: Optional[str] = None


def detect_tool_call(response: str) -> ParsedToolCall:
    """
    Parse an LLM response for a tool call directive.

    Expects the response (after strip) to be valid JSON:
        {"tool_call": {"name": "...", "args": {...}}}

    Returns ParsedToolCall with found=True if a valid tool call is found,
    found=False otherwise (including on malformed JSON).

    Design: strict. If the LLM produces anything other than this exact format,
    it's treated as a normal text response. This prevents false positives.
    """
    stripped = response.strip()

    if not stripped.startswith("{"):
        return ParsedToolCall(found=False)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return ParsedToolCall(found=False, parse_error=str(e))

    if not isinstance(data, dict) or "tool_call" not in data:
        return ParsedToolCall(found=False)

    call = data["tool_call"]

    if not isinstance(call, dict):
        return ParsedToolCall(
            found=False,
            parse_error=f"'tool_call' value must be a dict, got: {type(call)}"
        )

    name = call.get("name")
    args = call.get("args", {})

    if not isinstance(name, str) or not name:
        return ParsedToolCall(
            found=False,
            parse_error="'tool_call.name' must be a non-empty string"
        )

    if not isinstance(args, dict):
        return ParsedToolCall(
            found=False,
            parse_error=f"'tool_call.args' must be a dict, got: {type(args)}"
        )

    return ParsedToolCall(
        found=True,
        tool_name=name,
        args=args,
        raw_json=stripped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """
    Full result of a tool dispatch cycle.

    tool_results: All ToolResult objects produced during this call.
    final_response: The final LLM response after tool use (or original if no tool call).
    iterations: Number of tool calls executed.
    """
    final_response: str
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0


class ToolDispatcher:
    """
    Orchestrates the tool-use loop between the LLM and registered tools.

    Lifecycle per message:
    1. Receive LLM response
    2. detect_tool_call() → ParsedToolCall
    3. If found: execute tool → ToolResult
    4. Inject result into messages
    5. Call LLM again → get new response
    6. Go to 2 (until no tool call or max_iterations reached)
    7. Return DispatchResult with final response + all tool results

    Usage:
        dispatcher = ToolDispatcher(registry=registry, max_iterations=3)
        result = dispatcher.dispatch(
            initial_response=llm_response,
            messages=messages,
            llm_call=lambda msgs: llm_client.chat(msgs),
        )
        final_text = result.final_response
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_iterations: int = 5,
    ):
        self._registry = registry
        self._max_iterations = max_iterations

    def dispatch(
        self,
        initial_response: str,
        messages: list[dict],
        llm_call,  # callable(list[dict]) → str
    ) -> DispatchResult:
        """
        Run the tool-use loop starting from an initial LLM response.

        Args:
            initial_response: First LLM response (may or may not contain tool call).
            messages:         Original messages list (will be extended with tool results).
            llm_call:         Callable that sends messages → returns LLM response str.

        Returns:
            DispatchResult with final_response and all tool_results.
        """
        current_response = initial_response
        extended_messages = list(messages)   # copy — don't mutate caller's list
        tool_results: list[ToolResult] = []
        iterations = 0

        while iterations < self._max_iterations:
            parsed = detect_tool_call(current_response)

            if not parsed.found:
                logger.debug(f"ToolDispatcher: no tool call detected (iteration {iterations})")
                break

            logger.info(
                f"ToolDispatcher: tool call detected — "
                f"'{parsed.tool_name}' (iteration {iterations + 1})"
            )

            # Execute tool
            result = self._execute(parsed)
            tool_results.append(result)

            # Inject: assistant message (the tool call) + user message (the result)
            extended_messages.append({
                "role": "assistant",
                "content": parsed.raw_json,
            })
            extended_messages.append({
                "role": "user",
                "content": f"TOOL_RESULT [{parsed.tool_name}]: {result.to_message_content()}",
            })

            iterations += 1

            # Get next LLM response
            try:
                current_response = llm_call(extended_messages)
            except Exception as exc:
                logger.error(f"ToolDispatcher: LLM call failed after tool injection: {exc}")
                current_response = "I encountered an error processing the tool result."
                break

        if iterations >= self._max_iterations:
            logger.warning(
                f"ToolDispatcher: max_iterations ({self._max_iterations}) reached. "
                f"Returning last response."
            )

        return DispatchResult(
            final_response=current_response,
            tool_results=tool_results,
            iterations=iterations,
        )

    def _execute(self, parsed: ParsedToolCall) -> ToolResult:
        """
        Execute a parsed tool call. Always returns ToolResult (never raises).
        """
        t0 = time.perf_counter()

        # Unknown tool
        tool = self._registry.get_or_none(parsed.tool_name)
        if tool is None:
            known = self._registry.list()
            error_msg = (
                f"Unknown tool: '{parsed.tool_name}'. "
                f"Known tools: {known}"
            )
            logger.warning(f"ToolDispatcher: {error_msg}")
            return ToolResult(
                tool_name=parsed.tool_name,
                success=False,
                output={},
                error_msg=error_msg,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Execute
        try:
            if hasattr(tool, "safe_execute"):
                output = tool.safe_execute(**parsed.args)
            else:
                output = tool.execute(**parsed.args)

            if not isinstance(output, dict):
                raise ToolError(
                    f"Tool '{parsed.tool_name}' must return a dict, "
                    f"got: {type(output).__name__}"
                )

            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                f"ToolDispatcher: '{parsed.tool_name}' executed in {latency:.0f}ms"
            )
            return ToolResult(
                tool_name=parsed.tool_name,
                success=True,
                output=output,
                latency_ms=latency,
            )

        except ToolError as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"ToolDispatcher: ToolError from '{parsed.tool_name}': {exc}"
            )
            return ToolResult(
                tool_name=parsed.tool_name,
                success=False,
                output={},
                error_msg=str(exc),
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(
                f"ToolDispatcher: unexpected error from '{parsed.tool_name}': {exc}",
                exc_info=True,
            )
            return ToolResult(
                tool_name=parsed.tool_name,
                success=False,
                output={},
                error_msg=f"Unexpected error: {type(exc).__name__}: {exc}",
                latency_ms=latency,
            )

    def can_dispatch(self, response: str) -> bool:
        """Quick check — does this response contain a tool call?"""
        return detect_tool_call(response).found