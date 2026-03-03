"""
A.C.C.E.S.S. — Structured Logging Test Suite

TestStructuredLogger        — core StructuredLogger behavior
TestLoggerSink              — external sink callback
TestLoggerEventFiltering    — get_logs_by_type, by_session, recent
TestLoggerMaxEvents         — memory limit enforcement
TestAgentCoreLogging        — integration: events logged per turn
TestLoggingDisabled         — enable_structured_logging=False: no events
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from datetime import UTC
from agent.logger import StructuredLogger
from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient


# ─── Agent fakes ─────────────────────────────────────────────────────────────

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

def _make_agent(config=None, telemetry_hook=None):
    config = config or AgentConfig(enable_rag=False, apply_emotional_protection=False,
                                    enable_structured_logging=True)
    return AgentCore(
        emotion_engine=FakeEngine(), conversation_modulator=FakeMod(),
        prompt_builder=FakeBuilder(), llm_client=FakeLLMClient(response="Response."),
        config=config, telemetry_hook=telemetry_hook,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredLogger:
    def test_initial_event_count_zero(self):
        log = StructuredLogger()
        assert log.event_count == 0

    def test_log_event_increments_count(self):
        log = StructuredLogger()
        log.log_event("test", {"key": "val"})
        assert log.event_count == 1

    def test_get_logs_returns_list(self):
        log = StructuredLogger()
        log.log_event("test", {})
        events = log.get_logs()
        assert isinstance(events, list)
        assert len(events) == 1

    def test_event_has_required_fields(self):
        log = StructuredLogger()
        log.log_event("turn_completed", {"x": 1}, session_id="s1", turn_index=0)
        event = log.get_logs()[0]
        assert {"event_type", "timestamp", "session_id", "turn_index", "payload"} == set(event.keys())

    def test_event_type_stored_correctly(self):
        log = StructuredLogger()
        log.log_event("reflection_done", {})
        assert log.get_logs()[0]["event_type"] == "reflection_done"

    def test_payload_stored_correctly(self):
        log = StructuredLogger()
        log.log_event("test", {"answer": 42})
        assert log.get_logs()[0]["payload"] == {"answer": 42}

    def test_timestamp_is_utc_iso(self):
        from datetime import datetime
        log = StructuredLogger()
        log.log_event("test", {})
        ts = log.get_logs()[0]["timestamp"]
        # Should parse without error and have timezone info
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_get_logs_returns_copy(self):
        log = StructuredLogger()
        log.log_event("test", {})
        logs = log.get_logs()
        logs.clear()
        assert log.event_count == 1  # original not affected

    def test_clear_removes_all_events(self):
        log = StructuredLogger()
        for _ in range(5): log.log_event("test", {})
        log.clear()
        assert log.event_count == 0

    def test_repr_contains_event_count(self):
        log = StructuredLogger()
        log.log_event("test", {})
        assert "1" in repr(log)


class TestLoggerSink:
    def test_sink_called_on_each_event(self):
        received = []
        log = StructuredLogger(sink=received.append)
        log.log_event("test1", {"a": 1})
        log.log_event("test2", {"b": 2})
        assert len(received) == 2

    def test_sink_receives_correct_event(self):
        received = []
        log = StructuredLogger(sink=received.append)
        log.log_event("my_event", {"x": 99})
        assert received[0]["event_type"] == "my_event"
        assert received[0]["payload"]["x"] == 99

    def test_failing_sink_does_not_crash_logger(self):
        def bad_sink(event): raise RuntimeError("sink down")
        log = StructuredLogger(sink=bad_sink)
        log.log_event("test", {})  # should not raise
        assert log.event_count == 1

    def test_no_sink_no_crash(self):
        log = StructuredLogger(sink=None)
        log.log_event("test", {})  # should not raise
        assert log.event_count == 1


class TestLoggerEventFiltering:
    def _populated_logger(self):
        log = StructuredLogger()
        log.log_event("turn_completed", {"t": 1}, session_id="s1", turn_index=0)
        log.log_event("reflection_done", {"r": 1}, session_id="s1", turn_index=0)
        log.log_event("turn_completed", {"t": 2}, session_id="s2", turn_index=1)
        log.log_event("tool_used", {"tool": "echo"}, session_id="s1", turn_index=1)
        return log

    def test_get_logs_by_type(self):
        log = self._populated_logger()
        turns = log.get_logs_by_type("turn_completed")
        assert len(turns) == 2
        assert all(e["event_type"] == "turn_completed" for e in turns)

    def test_get_logs_by_type_empty_when_none(self):
        log = self._populated_logger()
        assert log.get_logs_by_type("nonexistent") == []

    def test_get_logs_by_session(self):
        log = self._populated_logger()
        s1_logs = log.get_logs_by_session("s1")
        assert len(s1_logs) == 3
        assert all(e["session_id"] == "s1" for e in s1_logs)

    def test_get_recent_n(self):
        log = self._populated_logger()
        recent = log.get_recent(2)
        assert len(recent) == 2
        # Most recent events
        assert recent[-1]["event_type"] == "tool_used"

    def test_get_recent_more_than_total(self):
        log = StructuredLogger()
        log.log_event("test", {})
        assert len(log.get_recent(100)) == 1


class TestLoggerMaxEvents:
    def test_max_events_enforced(self):
        log = StructuredLogger(max_events=5)
        for i in range(8): log.log_event("test", {"i": i})
        assert log.event_count <= 5

    def test_oldest_events_dropped(self):
        log = StructuredLogger(max_events=3)
        for i in range(5): log.log_event("event", {"i": i})
        remaining = log.get_logs()
        payloads = [e["payload"]["i"] for e in remaining]
        # Newest events should be kept
        assert max(payloads) == 4

    def test_max_events_one(self):
        log = StructuredLogger(max_events=1)
        log.log_event("first", {})
        log.log_event("second", {})
        assert log.event_count == 1
        assert log.get_logs()[0]["event_type"] == "second"


class TestAgentCoreLogging:
    def test_turn_completed_logged_after_handle_message(self):
        agent = _make_agent()
        agent.handle_message("Test.")
        logs = agent.get_logs_by_type("turn_completed")
        assert len(logs) >= 1

    def test_stream_finalized_logged_after_stream(self):
        agent = _make_agent()
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        logs = agent.get_logs_by_type("stream_finalized")
        assert len(logs) >= 1

    def test_reflection_logged_after_turn(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False, enable_structured_logging=True)
        agent = _make_agent(config=config)
        agent.handle_message("Test.")
        logs = agent.get_logs_by_type("reflection_done")
        assert len(logs) >= 1

    def test_get_logs_empty_list_when_disabled(self):
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             enable_structured_logging=False)
        agent = _make_agent(config=config)
        agent.handle_message("Test.")
        assert agent.get_logs() == []

    def test_get_logs_by_type_returns_empty_when_disabled(self):
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             enable_structured_logging=False)
        agent = _make_agent(config=config)
        agent.handle_message("Test.")
        assert agent.get_logs_by_type("turn_completed") == []

    def test_multiple_turns_accumulate_logs(self):
        agent = _make_agent()
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        logs = agent.get_logs_by_type("turn_completed")
        assert len(logs) == 2

    def test_event_payload_contains_turn_index(self):
        agent = _make_agent()
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        logs = agent.get_logs_by_type("turn_completed")
        # turn_index in the event itself
        assert any(e.get("turn_index") is not None for e in logs)


class TestLoggingDisabled:
    def test_no_logs_when_disabled(self):
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             enable_structured_logging=False)
        agent = AgentCore(
            emotion_engine=FakeEngine(), conversation_modulator=FakeMod(),
            prompt_builder=FakeBuilder(), llm_client=FakeLLMClient(response="R."),
            config=config,
        )
        agent.handle_message("Test.")
        assert agent.get_logs() == []

    def test_log_count_zero_when_disabled(self):
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             enable_structured_logging=False)
        agent = AgentCore(
            emotion_engine=FakeEngine(), conversation_modulator=FakeMod(),
            prompt_builder=FakeBuilder(), llm_client=FakeLLMClient(response="R."),
            config=config,
        )
        assert agent.stats()["log_events"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestStructuredLogger, TestLoggerSink, TestLoggerEventFiltering,
        TestLoggerMaxEvents, TestAgentCoreLogging, TestLoggingDisabled,
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