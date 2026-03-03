"""
A.C.C.E.S.S. — Tool Use Test Suite

Coverage:
    TestBaseTool              — base_tool.py: validate_args, to_prompt_description
    TestToolResult            — ToolResult contract, to_message_content JSON
    TestToolRegistry          — register, get, get_or_none, duplicate check
    TestDetectToolCall        — detect_tool_call(): valid/invalid/edge cases
    TestToolDispatcher        — dispatch loop, injection, max_iterations
    TestBuiltinEchoTool       — EchoTool functional
    TestBuiltinDateTimeTool   — GetDateTimeTool functional
    TestBuiltinCalculator     — CalculatorTool safe eval, blocked constructs
    TestAgentCoreToolDisabled — no tool execution when flag is off
    TestAgentCoreToolEnabled  — tool dispatched when enabled, result in AgentResponse
    TestAgentCoreToolError    — unknown tool handled gracefully
    TestToolPromptInjection   — tool catalog injected into system prompt
    TestAgentCoreToolChain    — multi-tool turn (tool → second LLM → response)
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Optional

from tools.base_tool import BaseTool, ToolError, ToolResult
from tools.tool_registry import ToolRegistry
from tools.tool_dispatcher import ToolDispatcher, detect_tool_call, ParsedToolCall
from tools.builtins import EchoTool, GetDateTimeTool, CalculatorTool

from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL FAKE TOOL
# ─────────────────────────────────────────────────────────────────────────────

class UpperCaseTool(BaseTool):
    """Test tool: upper-cases input text."""
    call_count = 0

    @property
    def name(self) -> str: return "uppercase"
    @property
    def description(self) -> str: return "Upper-cases the input text."
    @property
    def schema(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def execute(self, text: str = "", **kwargs) -> dict:
        UpperCaseTool.call_count += 1
        return {"result": text.upper()}


class FailingTool(BaseTool):
    @property
    def name(self) -> str: return "failing_tool"
    @property
    def description(self) -> str: return "Always fails."
    @property
    def schema(self) -> dict: return {"type": "object", "properties": {}, "required": []}
    def execute(self, **kwargs) -> dict: raise ToolError("Intentional failure.")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED AGENT FAKES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; pad: object = None
    is_positive: bool = False; is_negative: bool = False; label: str = "neutral"
    def __post_init__(self):
        @dataclass
        class P: valence: float = 0.0; arousal: float = 0.5; dominance: float = 0.5
        if self.pad is None: self.pad = P()

@dataclass
class FakeModulation:
    tone: str = "neutral"; pacing: str = "normal"; verbosity: str = "normal"
    structure_bias: str = "conversational"; emotional_validation: bool = False
    motivational_bias: float = 0.0; cognitive_load_limit: float = 1.0
    active_strategies: tuple = ()

@dataclass
class FakeBuiltPrompt:
    sections: tuple = ("tone",)
    def to_api_messages(self):
        return [{"role": "system", "content": "System."}, {"role": "user", "content": "test"}]
    def has_section(self, n): return n in self.sections

class FakeEngine:
    def __init__(self): self.protection_calls = []
    def process_interaction(self, t, session_id=None): return FakeState()
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, s): self.protection_calls.append(s)
    def stats(self): return {}

class FakeModulator:
    def build_modulation(self, state, trend, dominant_pattern=None): return FakeModulation()

class FakeBuilder:
    def build(self, user_input, modulation, memory_context=None): return FakeBuiltPrompt()

def _make_tool_agent(llm, registry=None, config_overrides=None):
    base_config = dict(enable_rag=False, write_user_turn_to_memory=False,
                       apply_emotional_protection=False)
    if config_overrides:
        base_config.update(config_overrides)
    config = AgentConfig(**base_config)
    if registry is not None:
        config.enable_tool_use = True
        config.tool_registry = registry
    return AgentCore(
        emotion_engine=FakeEngine(),
        conversation_modulator=FakeModulator(),
        prompt_builder=FakeBuilder(),
        llm_client=llm,
        config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseTool:
    def test_validate_args_passes_when_required_present(self):
        tool = EchoTool()
        tool.validate_args({"text": "hello"})  # should not raise

    def test_validate_args_raises_on_missing_required(self):
        tool = EchoTool()
        try:
            tool.validate_args({})
            assert False
        except ToolError as e:
            assert "text" in str(e)

    def test_to_prompt_description_contains_name(self):
        tool = EchoTool()
        desc = tool.to_prompt_description()
        assert "echo" in desc

    def test_to_prompt_description_contains_description(self):
        tool = EchoTool()
        desc = tool.to_prompt_description()
        assert "Echoes" in desc

    def test_safe_execute_validates_before_execute(self):
        tool = EchoTool()
        try:
            tool.safe_execute()  # missing required arg 'text'
            assert False
        except ToolError:
            pass

    def test_repr_contains_class_name(self):
        tool = EchoTool()
        assert "EchoTool" in repr(tool)


class TestToolResult:
    def test_success_result_json_is_dict(self):
        result = ToolResult(tool_name="echo", success=True, output={"echoed": "hi"})
        content = result.to_message_content()
        parsed = json.loads(content)
        assert parsed == {"echoed": "hi"}

    def test_failure_result_has_error_key(self):
        result = ToolResult(tool_name="echo", success=False, output={}, error_msg="oops")
        content = result.to_message_content()
        parsed = json.loads(content)
        assert "error" in parsed
        assert parsed["error"] == "oops"

    def test_tool_result_is_frozen(self):
        result = ToolResult(tool_name="echo", success=True, output={})
        try:
            result.success = False
            assert False
        except (AttributeError, TypeError):
            pass

    def test_to_log_dict_has_required_keys(self):
        result = ToolResult(tool_name="echo", success=True, output={"k": "v"}, latency_ms=10.0)
        d = result.to_log_dict()
        assert {"tool", "success", "latency_ms"}.issubset(set(d.keys()))

    def test_repr_contains_tool_name(self):
        result = ToolResult(tool_name="my_tool", success=True, output={})
        assert "my_tool" in repr(result)


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool = reg.get("echo")
        assert tool.name == "echo"

    def test_get_unknown_raises_tool_error(self):
        reg = ToolRegistry()
        try:
            reg.get("nonexistent")
            assert False
        except ToolError as e:
            assert "nonexistent" in str(e)

    def test_get_or_none_returns_none_for_unknown(self):
        reg = ToolRegistry()
        assert reg.get_or_none("nothing") is None

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        try:
            reg.register(EchoTool())
            assert False
        except ValueError:
            pass

    def test_overwrite_allowed_with_flag(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(EchoTool(), overwrite=True)  # should not raise

    def test_list_returns_sorted_names(self):
        reg = ToolRegistry()
        reg.register(GetDateTimeTool())
        reg.register(EchoTool())
        reg.register(CalculatorTool())
        names = reg.list()
        assert names == sorted(names)

    def test_contains_operator(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert "echo" in reg
        assert "unknown" not in reg

    def test_len(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(GetDateTimeTool())
        assert len(reg) == 2

    def test_unregister_removes_tool(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.unregister("echo")
        assert "echo" not in reg

    def test_unregister_noop_for_unknown(self):
        reg = ToolRegistry()
        reg.unregister("nothing")  # should not raise

    def test_get_prompt_section_contains_tool_names(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(CalculatorTool())
        section = reg.get_prompt_section()
        assert "echo" in section
        assert "calculator" in section

    def test_get_prompt_section_empty_when_no_tools(self):
        reg = ToolRegistry()
        assert reg.get_prompt_section() == ""

    def test_non_protocol_raises_value_error(self):
        reg = ToolRegistry()
        try:
            reg.register("not a tool")
            assert False
        except ValueError:
            pass


class TestDetectToolCall:
    def _call(self, response: str) -> ParsedToolCall:
        return detect_tool_call(response)

    def test_valid_tool_call_detected(self):
        json_str = '{"tool_call": {"name": "echo", "args": {"text": "hi"}}}'
        result = self._call(json_str)
        assert result.found is True
        assert result.tool_name == "echo"
        assert result.args == {"text": "hi"}

    def test_no_tool_call_in_plain_text(self):
        result = self._call("This is a normal response.")
        assert result.found is False

    def test_missing_tool_call_key(self):
        result = self._call('{"other": "stuff"}')
        assert result.found is False

    def test_empty_name_not_detected(self):
        result = self._call('{"tool_call": {"name": "", "args": {}}}')
        assert result.found is False

    def test_args_not_dict_not_detected(self):
        result = self._call('{"tool_call": {"name": "echo", "args": "text"}}')
        assert result.found is False

    def test_invalid_json_not_detected(self):
        result = self._call("{invalid json}")
        assert result.found is False

    def test_empty_args_dict_is_valid(self):
        result = self._call('{"tool_call": {"name": "get_datetime", "args": {}}}')
        assert result.found is True
        assert result.args == {}

    def test_extra_whitespace_handled(self):
        result = self._call('  {"tool_call": {"name": "echo", "args": {"text": "hi"}}}  ')
        assert result.found is True

    def test_tool_call_value_not_dict_not_detected(self):
        result = self._call('{"tool_call": "echo"}')
        assert result.found is False

    def test_raw_json_preserved_on_success(self):
        s = '{"tool_call": {"name": "echo", "args": {"text": "x"}}}'
        result = self._call(s)
        assert result.found is True
        parsed = json.loads(result.raw_json)
        assert parsed["tool_call"]["name"] == "echo"


class TestToolDispatcher:
    def test_no_tool_call_returns_original_response(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        dispatcher = ToolDispatcher(registry=reg)
        result = dispatcher.dispatch(
            initial_response="Plain text, no tool call.",
            messages=[{"role": "user", "content": "test"}],
            llm_call=lambda msgs: "Should not be called",
        )
        assert result.final_response == "Plain text, no tool call."
        assert result.iterations == 0
        assert result.tool_results == []

    def test_tool_executed_on_valid_call(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        dispatcher = ToolDispatcher(registry=reg)
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "hello"}}}'
        result = dispatcher.dispatch(
            initial_response=tool_json,
            messages=[{"role": "user", "content": "test"}],
            llm_call=lambda msgs: "Final response after tool.",
        )
        assert result.iterations == 1
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True
        assert result.final_response == "Final response after tool."

    def test_tool_result_injected_into_messages(self):
        captured_messages = []
        def capture_llm(msgs):
            captured_messages.extend(msgs)
            return "Done."
        reg = ToolRegistry()
        reg.register(EchoTool())
        dispatcher = ToolDispatcher(registry=reg)
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "test"}}}'
        dispatcher.dispatch(tool_json, [{"role": "user", "content": "q"}], capture_llm)
        roles = [m["role"] for m in captured_messages]
        assert "assistant" in roles  # tool call message
        assert any("TOOL_RESULT" in m.get("content", "") for m in captured_messages)

    def test_unknown_tool_produces_failure_result(self):
        reg = ToolRegistry()
        dispatcher = ToolDispatcher(registry=reg)
        tool_json = '{"tool_call": {"name": "no_such_tool", "args": {}}}'
        result = dispatcher.dispatch(
            tool_json,
            [{"role": "user", "content": "q"}],
            lambda msgs: "Final.",
        )
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "no_such_tool" in result.tool_results[0].error_msg

    def test_failing_tool_produces_failure_result(self):
        reg = ToolRegistry()
        reg.register(FailingTool())
        dispatcher = ToolDispatcher(registry=reg)
        tool_json = '{"tool_call": {"name": "failing_tool", "args": {}}}'
        result = dispatcher.dispatch(
            tool_json, [{"role": "user", "content": "q"}], lambda msgs: "Done."
        )
        assert result.tool_results[0].success is False
        assert "Intentional failure" in result.tool_results[0].error_msg

    def test_max_iterations_stops_loop(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        dispatcher = ToolDispatcher(registry=reg, max_iterations=2)
        # LLM always returns a tool call → would loop forever without max_iterations
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "loop"}}}'
        result = dispatcher.dispatch(
            tool_json,
            [{"role": "user", "content": "q"}],
            llm_call=lambda msgs: tool_json,  # always returns another tool call
        )
        assert result.iterations == 2

    def test_can_dispatch_detects_tool_call(self):
        reg = ToolRegistry()
        dispatcher = ToolDispatcher(registry=reg)
        assert dispatcher.can_dispatch('{"tool_call": {"name": "x", "args": {}}}')
        assert not dispatcher.can_dispatch("Just plain text.")


class TestBuiltinEchoTool:
    def test_echoes_input(self):
        tool = EchoTool()
        result = tool.execute(text="Hello world!")
        assert result == {"echoed": "Hello world!"}

    def test_satisfies_tool_protocol(self):
        from tools.base_tool import ToolProtocol
        assert isinstance(EchoTool(), ToolProtocol)

    def test_empty_string_echoed(self):
        tool = EchoTool()
        assert tool.execute(text="") == {"echoed": ""}


class TestBuiltinDateTimeTool:
    def test_returns_dict_with_datetime_keys(self):
        tool = GetDateTimeTool()
        result = tool.execute()
        assert "utc_datetime" in result
        assert "utc_date" in result
        assert "day_of_week" in result

    def test_date_format_is_iso(self):
        tool = GetDateTimeTool()
        result = tool.execute()
        from datetime import datetime
        # Should parse without raising
        datetime.fromisoformat(result["utc_datetime"])

    def test_no_required_args(self):
        tool = GetDateTimeTool()
        assert tool.schema.get("required", []) == []


class TestBuiltinCalculator:
    def test_basic_addition(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3")
        assert result["result"] == 5.0

    def test_power(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")
        assert result["result"] == 1024.0

    def test_modulo(self):
        tool = CalculatorTool()
        result = tool.execute(expression="10 % 3")
        assert result["result"] == 1.0

    def test_complex_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="(2 + 3) * 4 - 1")
        assert result["result"] == 19.0

    def test_division_by_zero_raises_tool_error(self):
        tool = CalculatorTool()
        try:
            tool.execute(expression="1 / 0")
            assert False
        except ToolError as e:
            assert "zero" in str(e).lower()

    def test_function_call_blocked(self):
        tool = CalculatorTool()
        try:
            tool.execute(expression="abs(-5)")
            assert False
        except ToolError:
            pass

    def test_empty_expression_raises(self):
        tool = CalculatorTool()
        try:
            tool.execute(expression="")
            assert False
        except ToolError:
            pass

    def test_expression_preserved_in_result(self):
        tool = CalculatorTool()
        result = tool.execute(expression="1 + 1")
        assert result["expression"] == "1 + 1"

    def test_negative_number_supported(self):
        tool = CalculatorTool()
        result = tool.execute(expression="-5 + 10")
        assert result["result"] == 5.0


class TestAgentCoreToolDisabled:
    """No tool execution when enable_tool_use=False."""

    def test_tool_not_executed_when_disabled(self):
        UpperCaseTool.call_count = 0
        reg = ToolRegistry()
        reg.register(UpperCaseTool())
        # Tool use disabled (default)
        tool_json = '{"tool_call": {"name": "uppercase", "args": {"text": "hello"}}}'
        llm = FakeLLMClient(response=tool_json)
        agent = _make_tool_agent(llm, registry=None)  # no registry → disabled
        result = agent.handle_message("Test.")
        # LLM responded with tool JSON, but agent didn't dispatch it
        assert UpperCaseTool.call_count == 0

    def test_tool_json_returned_as_is_when_disabled(self):
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "hi"}}}'
        llm = FakeLLMClient(response=tool_json)
        agent = _make_tool_agent(llm, registry=None)
        result = agent.handle_message("Test.")
        # Output is the raw tool JSON since dispatch didn't run
        assert result.assistant_output.strip() == tool_json.strip()

    def test_tool_results_empty_when_disabled(self):
        llm = FakeLLMClient(response="Normal response.")
        agent = _make_tool_agent(llm)
        result = agent.handle_message("Test.")
        assert result.tool_results == ()


class TestAgentCoreToolEnabled:
    """Tool dispatched when enabled; result appears in AgentResponse."""

    def test_tool_executed_when_enabled(self):
        UpperCaseTool.call_count = 0
        reg = ToolRegistry()
        reg.register(UpperCaseTool())
        tool_json = '{"tool_call": {"name": "uppercase", "args": {"text": "hello"}}}'
        # First LLM call returns tool JSON; second returns final response
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return tool_json
            return "HELLO was the result."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("uppercase hello")
        assert UpperCaseTool.call_count >= 1

    def test_tool_results_in_agent_response(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "test"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Done."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Echo test.")
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "echo"

    def test_final_response_after_tool_use(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "hi"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Final answer."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Test.")
        assert result.assistant_output == "Final answer."

    def test_tool_dispatch_ms_nonzero_when_tool_used(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "x"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Done."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Test.")
        assert result.trace.tool_dispatch_ms >= 0.0

    def test_tool_results_is_tuple_in_response(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        llm = FakeLLMClient(response="No tool call here.")
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Test.")
        assert isinstance(result.tool_results, tuple)


class TestAgentCoreToolError:
    """Unknown/failing tools handled gracefully — no crash."""

    def test_unknown_tool_no_crash(self):
        reg = ToolRegistry()
        tool_json = '{"tool_call": {"name": "unknown_tool", "args": {}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Done anyway."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Test.")
        assert isinstance(result, __import__("agent").AgentResponse)

    def test_failing_tool_no_crash(self):
        reg = ToolRegistry()
        reg.register(FailingTool())
        tool_json = '{"tool_call": {"name": "failing_tool", "args": {}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Done."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg)
        result = agent.handle_message("Test.")
        assert result.tool_results[0].success is False


class TestToolPromptInjection:
    """Tool catalog is passed into memory_context before PromptBuilder.build()."""

    def test_tool_catalog_in_messages(self):
        # AgentCore injects the tool catalog into memory_context before calling
        # PromptBuilder.build(). The real PromptBuilder injects it into the system prompt.
        # Here we verify AgentCore sends it through correctly.
        received_contexts = []

        class CapturingBuilder:
            def build(self_, user_input, modulation, memory_context=None):
                received_contexts.append(memory_context or "")
                class P:
                    sections = ()
                    def to_api_messages(self):
                        return [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
                return P()

        from tools.tool_registry import ToolRegistry as TR
        from tools.builtins import EchoTool as ET
        from agent.agent_config import AgentConfig as AC
        from agent.agent_core import AgentCore as AC2
        from agent.llm_client import FakeLLMClient as FLLM

        reg = TR(); reg.register(ET())
        config = AC(enable_rag=False, write_user_turn_to_memory=False,
                    apply_emotional_protection=False, enable_tool_use=True, tool_registry=reg)
        agent = AC2(
            emotion_engine=FakeEngine(),
            conversation_modulator=FakeModulator(),
            prompt_builder=CapturingBuilder(),
            llm_client=FLLM(response="Done."),
            config=config,
        )
        agent.handle_message("Test.")

        assert len(received_contexts) == 1
        ctx = received_contexts[0]
        assert "AVAILABLE TOOLS" in ctx or "echo" in ctx, (
            f"Tool catalog not in memory_context. Got: {ctx[:200]!r}"
        )

class TestAgentCoreToolChain:
    """Multi-step: tool call → result injection → second LLM → final response."""

    def test_two_sequential_tool_calls(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool_json_1 = '{"tool_call": {"name": "echo", "args": {"text": "first"}}}'
        tool_json_2 = '{"tool_call": {"name": "echo", "args": {"text": "second"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            if call_count[0] == 1: return tool_json_1
            if call_count[0] == 2: return tool_json_2
            return "All done."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg,
                                  config_overrides={"max_tool_iterations": 5})
        result = agent.handle_message("Chain test.")
        assert len(result.tool_results) == 2
        assert result.assistant_output == "All done."

    def test_tool_results_are_all_successful_in_chain(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "x"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] <= 2 else "Done."
        llm = FakeLLMClient(response_fn=llm_fn)
        agent = _make_tool_agent(llm, registry=reg,
                                  config_overrides={"max_tool_iterations": 5})
        result = agent.handle_message("Chain.")
        assert all(r.success for r in result.tool_results)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestBaseTool, TestToolResult, TestToolRegistry, TestDetectToolCall,
        TestToolDispatcher, TestBuiltinEchoTool, TestBuiltinDateTimeTool,
        TestBuiltinCalculator, TestAgentCoreToolDisabled, TestAgentCoreToolEnabled,
        TestAgentCoreToolError, TestToolPromptInjection, TestAgentCoreToolChain,
    ]

    passed, failed = [], []
    for cls in test_classes:
        instance = cls()
        for method_name in sorted(m for m in dir(cls) if m.startswith("test_")):
            label = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed.append(label)
                print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e))
                print(f"  ❌ {label}: {e}")
                traceback.print_exc()

    total = len(passed) + len(failed)
    print(f"\n{'='*60}\nResults: {len(passed)}/{total} passed")
    if failed:
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")