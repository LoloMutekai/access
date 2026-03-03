"""
A.C.C.E.S.S. — Telemetry Hook Test Suite

TestTelemetryHookBasics     — hook called on key events
TestTelemetryHookIsolation  — crash never propagates to agent
TestTelemetryHookPayload    — payload structure per event type
TestTelemetryHookOptional   — no hook → no crash
TestTelemetryEventTypes     — all promised event types fired
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Optional
from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient
from agent.memory_loop import MemoryLoop


# ─── Fakes ───────────────────────────────────────────────────────────────────

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; intensity: float = 0.5; label: str = "neutral"
    is_positive: bool = False; is_negative: bool = False; pad: object = None
    is_high_arousal: bool = False
    def __post_init__(self):
        @dataclass
        class P: valence: float = 0.0; arousal: float = 0.5; dominance: float = 0.5
        if self.pad is None: self.pad = P()

@dataclass
class FakeModulation:
    tone: str = "neutral"; pacing: str = "normal"; verbosity: str = "normal"
    structure_bias: str = "conversational"; emotional_validation: bool = False
    motivational_bias: float = 0.0; cognitive_load_limit: float = 1.0; active_strategies: tuple = ()

@dataclass
class FakeBuiltPrompt:
    sections: tuple = ()
    def to_api_messages(self): return [{"role":"system","content":"S"},{"role":"user","content":"u"}]

class FakeEngine:
    def __init__(self): self.protection_calls = []
    def process_interaction(self, t, session_id=None): return FakeState()
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, s): self.protection_calls.append(s)
    def stats(self): return {}

class FakeMod:
    def build_modulation(self, state, trend, dominant_pattern=None): return FakeModulation()

class FakeBuilder:
    def build(self, user_input, modulation, memory_context=None): return FakeBuiltPrompt()

class FakeMemory:
    def __init__(self): self.add_calls = []
    def add_memory(self, **kw): self.add_calls.append(kw)
    def retrieve_relevant_memories(self, **kw): return []
    def format_for_rag(self, m): return ""

def _make_agent(hook=None, config=None, memory_loop=None):
    config = config or AgentConfig(enable_rag=False, apply_emotional_protection=False)
    return AgentCore(
        emotion_engine=FakeEngine(), conversation_modulator=FakeMod(),
        prompt_builder=FakeBuilder(), llm_client=FakeLLMClient(response="Response."),
        config=config, telemetry_hook=hook, memory_loop=memory_loop,
    )

class TelemetryCollector:
    """Collects all telemetry calls for assertion."""
    def __init__(self):
        self.calls = []

    def hook(self, event_name: str, metadata: dict):
        self.calls.append((event_name, metadata))

    def event_names(self): return [c[0] for c in self.calls]
    def events_of_type(self, name): return [c for c in self.calls if c[0] == name]


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTelemetryHookBasics:
    def test_hook_called_on_turn_completed(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        agent.handle_message("Test.")
        assert "turn_completed" in collector.event_names()

    def test_hook_called_on_stream_finalized(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        assert "turn_completed" in collector.event_names()

    def test_hook_receives_event_name_as_string(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        agent.handle_message("Test.")
        for name, _ in collector.calls:
            assert isinstance(name, str)

    def test_hook_receives_metadata_as_dict(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        agent.handle_message("Test.")
        for _, metadata in collector.calls:
            assert isinstance(metadata, dict)

    def test_hook_called_multiple_times_for_multiple_turns(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        turn_events = collector.events_of_type("turn_completed")
        assert len(turn_events) == 2

    def test_hook_called_on_reflection_done(self):
        collector = TelemetryCollector()
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(hook=collector.hook, config=config)
        agent.handle_message("Test.")
        assert "reflection_done" in collector.event_names()


class TestTelemetryHookIsolation:
    def test_crashing_hook_does_not_crash_agent(self):
        def bad_hook(name, meta): raise RuntimeError("telemetry system down")
        agent = _make_agent(hook=bad_hook)
        # Should not raise
        result = agent.handle_message("Test.")
        assert result is not None

    def test_crashing_hook_does_not_affect_response(self):
        def bad_hook(name, meta): raise RuntimeError("down")
        agent = _make_agent(hook=bad_hook)
        result = agent.handle_message("My question.")
        assert result.assistant_output == "Response."

    def test_crashing_hook_allows_multiple_turns(self):
        def bad_hook(name, meta): raise RuntimeError("down")
        agent = _make_agent(hook=bad_hook)
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")  # should not raise
        assert agent.turn_index == 2

    def test_none_hook_does_not_crash(self):
        agent = _make_agent(hook=None)
        result = agent.handle_message("Test.")
        assert result is not None


class TestTelemetryHookPayload:
    def test_turn_completed_payload_has_output(self):
        collector = TelemetryCollector()
        agent = _make_agent(hook=collector.hook)
        agent.handle_message("Test.")
        events = collector.events_of_type("turn_completed")
        assert len(events) == 1
        payload = events[0][1]
        assert "output_chars" in payload or "latency_ms" in payload

    def test_reflection_done_payload_has_importance(self):
        collector = TelemetryCollector()
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(hook=collector.hook, config=config)
        agent.handle_message("Test.")
        events = collector.events_of_type("reflection_done")
        assert len(events) >= 1
        payload = events[0][1]
        assert "importance_score" in payload

    def test_tool_used_payload_has_tool_name(self):
        from tools.tool_registry import ToolRegistry
        from tools.builtins import EchoTool
        collector = TelemetryCollector()
        reg = ToolRegistry(); reg.register(EchoTool())
        tool_json = '{"tool_call": {"name": "echo", "args": {"text": "hi"}}}'
        call_count = [0]
        def llm_fn(msgs):
            call_count[0] += 1
            return tool_json if call_count[0] == 1 else "Done."
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             enable_tool_use=True, tool_registry=reg)
        agent = AgentCore(
            emotion_engine=FakeEngine(), conversation_modulator=FakeMod(),
            prompt_builder=FakeBuilder(), llm_client=FakeLLMClient(response_fn=llm_fn),
            config=config, telemetry_hook=collector.hook,
        )
        agent.handle_message("Use echo tool.")
        events = collector.events_of_type("tool_used")
        assert len(events) >= 1
        assert "tool_name" in events[0][1]


class TestTelemetryHookOptional:
    def test_no_hook_handle_message_works(self):
        agent = _make_agent(hook=None)
        result = agent.handle_message("Test.")
        assert result.assistant_output == "Response."

    def test_no_hook_streaming_works(self):
        agent = _make_agent(hook=None)
        tokens = list(agent.stream_message("Test."))
        result = agent.finalize_stream("".join(tokens))
        assert isinstance(result.assistant_output, str)

    def test_no_hook_maintenance_works(self):
        from agent.memory_loop import MemoryLoop
        class FakeMem:
            def run_decay(self): return type('R', (), {'processed':0,'updated':0,'below_floor':0})()
            _store = None
        loop = MemoryLoop(FakeMem())
        agent = _make_agent(hook=None, memory_loop=loop)
        result = agent.run_memory_maintenance(
            run_decay=True, run_consolidation=False, run_topics=False, run_repetition=False
        )
        assert result is not None


class TestTelemetryEventTypes:
    def test_memory_maintenance_done_fired_after_run_maintenance(self):
        collector = TelemetryCollector()
        from agent.memory_loop import MemoryLoop
        class FakeMem:
            def run_decay(self): return type('R', (), {'processed':0,'updated':0,'below_floor':0})()
            _store = None
        loop = MemoryLoop(FakeMem())
        agent = _make_agent(hook=collector.hook, memory_loop=loop)
        agent.run_memory_maintenance(run_decay=True, run_consolidation=False,
                                      run_topics=False, run_repetition=False)
        assert "memory_maintenance_done" in collector.event_names()

    def test_all_promised_event_types_fired_in_full_turn(self):
        """turn_completed and reflection_done are fired in a normal turn."""
        collector = TelemetryCollector()
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(hook=collector.hook, config=config)
        agent.handle_message("Test.")
        events = set(collector.event_names())
        # Must include at minimum these two
        assert "turn_completed" in events
        assert "reflection_done" in events

    def test_hook_count_per_turn(self):
        """Each turn fires at least turn_completed + reflection_done = 2 events."""
        collector = TelemetryCollector()
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(hook=collector.hook, config=config)
        agent.handle_message("Test.")
        assert len(collector.calls) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestTelemetryHookBasics, TestTelemetryHookIsolation,
        TestTelemetryHookPayload, TestTelemetryHookOptional, TestTelemetryEventTypes,
    ]
    passed, failed = [], []
    for cls in test_classes:
        instance = cls()
        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            label = f"{cls.__name__}.{name}"
            try:
                getattr(instance, name)(); passed.append(label); print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e)); print(f"  ❌ {label}: {e}"); traceback.print_exc()
    total = len(passed) + len(failed)
    print(f"\n{'='*60}\nResults: {len(passed)}/{total} passed")
    if failed:
        for l, e in failed: print(f"  {l}: {e}")
        sys.exit(1)
    else: print("All checks passed ✅")