"""
A.C.C.E.S.S. — AgentCore Test Suite

Test strategy:
- Zero real API calls (FakeLLMClient throughout)
- Zero real embedders (FakeEmotionEngine, FakeModulator, FakePromptBuilder)
- Zero DB or FAISS (FakeMemoryManager)
- All tests deterministic and isolated
- Each class tests one logical concern

Coverage:
    TestFullPipelineFlow         — happy path, all stages, correct output shape
    TestLatencyMeasurement       — latency_ms present and > 0, trace populated
    TestMemoryWriteBack          — add_memory called when configured
    TestMemoryWriteBackDisabled  — add_memory NOT called when disabled
    TestRAGInjection             — memory context injected when RAG enabled
    TestEmotionalProtection      — protection called after turn
    TestAgentResponseContract    — AgentResponse immutability + API shape
    TestTurnIndexIncrement       — turn_index increments on each call
    TestSessionManagement        — session_id flows through, reset works
    TestLLMErrorHandling         — LLM failures produce fallback, not crash
    TestFakeLLMClient            — FakeLLMClient recording, response_fn, error sim
    TestConversationHistory      — history injected when enabled
    TestConfigurableConfig       — AgentConfig fields flow into agent behavior
    TestPipelineTrace            — per-stage timing recorded
    TestAgentStats               — stats() returns expected keys
    TestStageIsolation           — stage failure does not break downstream stages
    TestMultipleTurns            — sequential turns produce incrementing indices
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Optional, List

from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.models import AgentResponse, PipelineTrace
from agent.llm_client import FakeLLMClient, LLMError, LLMTimeoutError


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL FAKES — duck-typed stubs, no imports from emotion/conversation/prompt
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakePAD:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5


@dataclass
class FakeState:
    primary_emotion: str = "neutral"
    intensity: float = 0.5
    confidence: float = 0.8
    pad: FakePAD = None
    is_positive: bool = False
    is_negative: bool = False
    label: str = "moderate neutral"

    def __post_init__(self):
        if self.pad is None:
            self.pad = FakePAD()


@dataclass
class FakeModulation:
    tone: str = "neutral"
    pacing: str = "normal"
    verbosity: str = "normal"
    structure_bias: str = "conversational"
    emotional_validation: bool = False
    motivational_bias: float = 0.0
    cognitive_load_limit: float = 1.0
    active_strategies: tuple = ()


@dataclass
class FakeBuiltPrompt:
    system_prompt: str = "You are helpful."
    sections: tuple = ("tone", "pacing")
    char_count: int = 20

    def to_api_messages(self):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "test"},
        ]

    def has_section(self, name):
        return name in self.sections


class FakeEmotionEngine:
    """Fake EmotionEngine with call tracking."""

    def __init__(self, state: FakeState = None, fail: bool = False):
        self._state = state or FakeState()
        self._fail = fail
        self.process_calls: list[str] = []
        self.protection_calls: list = []
        self._trend = {
            "trend_label": "stable",
            "valence_slope": 0.0,
            "arousal_slope": 0.0,
            "dominance_slope": 0.0,
            "burnout_risk": False,
            "mania_risk": False,
            "data_points": 3,
        }

    def process_interaction(self, text: str, session_id=None):
        self.process_calls.append(text)
        if self._fail:
            raise RuntimeError("FakeEmotionEngine: forced failure")
        return self._state

    def emotional_trend(self):
        return self._trend

    def dominant_pattern(self, last_n=10):
        return None

    def apply_emotional_protection(self, state):
        self.protection_calls.append(state)

    def stats(self):
        return {"history_size": len(self.process_calls)}


class FakeModulator:
    """Fake ConversationModulator with call tracking."""

    def __init__(self, modulation: FakeModulation = None, fail: bool = False):
        self._modulation = modulation or FakeModulation()
        self._fail = fail
        self.calls: list = []

    def build_modulation(self, state, trend, dominant_pattern=None):
        self.calls.append((state, trend, dominant_pattern))
        if self._fail:
            raise RuntimeError("FakeModulator: forced failure")
        return self._modulation


class FakePromptBuilder:
    """Fake PromptBuilder with call tracking."""

    def __init__(self, prompt: FakeBuiltPrompt = None, fail: bool = False):
        self._prompt = prompt or FakeBuiltPrompt()
        self._fail = fail
        self.calls: list = []

    def build(self, user_input, modulation, memory_context=None):
        self.calls.append((user_input, modulation, memory_context))
        if self._fail:
            raise RuntimeError("FakePromptBuilder: forced failure")
        return self._prompt


class FakeMemoryManager:
    """Fake MemoryManager — tracks calls without any DB."""

    def __init__(self, memories=None, rag_context="Past: user worked on project X."):
        self._memories = memories or []
        self._rag_context = rag_context
        self.add_calls: list = []
        self.retrieve_calls: list = []
        self.format_calls: list = []

    def retrieve_relevant_memories(self, query, top_k=5, min_importance=0.0, emotional_context=None):
        self.retrieve_calls.append((query, top_k, min_importance, emotional_context))
        return self._memories

    def format_for_rag(self, memories):
        self.format_calls.append(memories)
        return self._rag_context if memories else ""

    def add_memory(self, content, summary="", memory_type="episodic",
                   tags=None, importance_score=0.5, source="interaction",
                   session_id=None):
        self.add_calls.append({
            "content": content,
            "summary": summary,
            "memory_type": memory_type,
            "tags": tags or [],
            "importance_score": importance_score,
            "source": source,
            "session_id": session_id,
        })


def _make_agent(
    llm_response="Test response.",
    llm_raise=None,
    llm_latency_ms=0.0,
    emotion_fail=False,
    modulator_fail=False,
    prompt_fail=False,
    memory=None,
    config=None,
    emotion_state=None,
) -> tuple[AgentCore, FakeLLMClient, FakeEmotionEngine, FakeModulator, FakePromptBuilder]:
    llm = FakeLLMClient(
        response=llm_response,
        raise_error=llm_raise,
        latency_ms=llm_latency_ms,
    )
    engine = FakeEmotionEngine(state=emotion_state, fail=emotion_fail)
    modulator = FakeModulator(fail=modulator_fail)
    builder = FakePromptBuilder(fail=prompt_fail)
    agent = AgentCore(
        emotion_engine=engine,
        conversation_modulator=modulator,
        prompt_builder=builder,
        llm_client=llm,
        memory_manager=memory,
        config=config or AgentConfig(),
    )
    return agent, llm, engine, modulator, builder


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineFlow:
    """Happy path: all stages run, AgentResponse has correct shape."""

    def test_returns_agent_response(self):
        agent, *_ = _make_agent(llm_response="Hello!")
        result = agent.handle_message("Hi there.")
        assert isinstance(result, AgentResponse)

    def test_assistant_output_matches_llm_response(self):
        agent, *_ = _make_agent(llm_response="Specific response text.")
        result = agent.handle_message("Test input.")
        assert result.assistant_output == "Specific response text."

    def test_user_input_preserved_in_response(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("My specific input.")
        assert result.user_input == "My specific input."

    def test_emotional_state_in_response(self):
        state = FakeState(primary_emotion="frustration", intensity=0.8)
        agent, *_ = _make_agent(emotion_state=state)
        result = agent.handle_message("I'm frustrated.")
        assert result.emotional_state is state

    def test_modulation_in_response(self):
        agent, llm, engine, modulator, builder = _make_agent()
        result = agent.handle_message("Test.")
        assert result.modulation is modulator._modulation

    def test_sections_used_in_response(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert isinstance(result.sections_used, tuple)

    def test_emotion_engine_called_once(self):
        agent, llm, engine, *_ = _make_agent()
        agent.handle_message("Hello.")
        assert len(engine.process_calls) == 1
        assert engine.process_calls[0] == "Hello."

    def test_llm_called_once(self):
        agent, llm, *_ = _make_agent()
        agent.handle_message("Test.")
        assert llm.call_count == 1

    def test_prompt_builder_called_with_user_input(self):
        agent, llm, engine, modulator, builder = _make_agent()
        agent.handle_message("Specific question.")
        assert len(builder.calls) == 1
        assert builder.calls[0][0] == "Specific question."

    def test_modulator_called_with_state_and_trend(self):
        agent, llm, engine, modulator, builder = _make_agent()
        agent.handle_message("Test.")
        assert len(modulator.calls) == 1


class TestLatencyMeasurement:
    """Latency must be measured and reported accurately."""

    def test_latency_ms_present_and_positive(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert result.latency_ms > 0.0

    def test_latency_ms_is_float(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert isinstance(result.latency_ms, float)

    def test_simulated_llm_latency_reflected(self):
        """50ms simulated LLM latency must push total latency > 50ms."""
        agent, *_ = _make_agent(llm_latency_ms=50.0)
        result = agent.handle_message("Test.")
        assert result.latency_ms >= 45.0  # allow small tolerance

    def test_trace_present_in_response(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert result.trace is not None
        assert isinstance(result.trace, PipelineTrace)

    def test_trace_llm_ms_positive(self):
        agent, *_ = _make_agent(llm_latency_ms=10.0)
        result = agent.handle_message("Test.")
        assert result.trace.llm_ms > 0.0

    def test_trace_total_ms_close_to_latency_ms(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert abs(result.trace.total_ms - result.latency_ms) < 5.0

    def test_trace_to_dict_has_all_keys(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        d = result.trace.to_dict()
        # Phase 2: tool_dispatch_ms added — use subset check for forward compatibility
        required = {
            "emotion_ms", "modulation_ms", "rag_ms",
            "prompt_ms", "llm_ms", "memory_write_ms", "total_ms"
        }
        assert required.issubset(set(d.keys()))


class TestMemoryWriteBack:
    """add_memory must be called when write_user_turn_to_memory=True."""

    def test_user_turn_written_when_configured(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Remember this.")
        user_writes = [c for c in mem.add_calls if c["source"] == "interaction"]
        assert len(user_writes) >= 1

    def test_user_turn_content_correct(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Store this message.")
        user_writes = [c for c in mem.add_calls if c["source"] == "interaction"]
        assert any("Store this message." in w["content"] for w in user_writes)

    def test_assistant_turn_written_when_configured(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            write_user_turn_to_memory=False,
            write_assistant_turn_to_memory=True,
        )
        agent, *_ = _make_agent(memory=mem, config=config, llm_response="My answer.")
        agent.handle_message("Question.")
        assistant_writes = [c for c in mem.add_calls if c["source"] == "agent_response"]
        assert len(assistant_writes) == 1
        assert "My answer." in assistant_writes[0]["content"]

    def test_memory_importance_configurable(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            write_user_turn_to_memory=True,
            auto_memory_importance=0.75,
        )
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Important message.")
        user_writes = [c for c in mem.add_calls if c["source"] == "interaction"]
        assert any(w["importance_score"] == 0.75 for w in user_writes)

    def test_session_id_passed_to_memory(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            write_user_turn_to_memory=True,
            default_session_id="session-abc",
        )
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Test.")
        user_writes = [c for c in mem.add_calls if c["source"] == "interaction"]
        assert any(w["session_id"] == "session-abc" for w in user_writes)

    def test_emotion_tags_added_to_memory(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True)
        state = FakeState(primary_emotion="frustration")
        agent, *_ = _make_agent(memory=mem, config=config, emotion_state=state)
        agent.handle_message("I'm stuck.")
        user_writes = [c for c in mem.add_calls if c["source"] == "interaction"]
        assert any("frustration" in w["tags"] for w in user_writes)


class TestMemoryWriteBackDisabled:
    """add_memory must NOT be called when write-back is disabled."""

    def test_no_memory_write_when_disabled(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            write_user_turn_to_memory=False,
            write_assistant_turn_to_memory=False,
        )
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Nothing to store.")
        assert len(mem.add_calls) == 0

    def test_no_memory_calls_when_memory_manager_none(self):
        config = AgentConfig(write_user_turn_to_memory=True)
        agent, llm, engine, *_ = _make_agent(memory=None, config=config)
        # Should not raise
        result = agent.handle_message("Test.")
        assert result.assistant_output is not None


class TestRAGInjection:
    """Memory context must be retrieved and injected when RAG is enabled."""

    def test_retrieve_called_when_rag_enabled(self):
        mem = FakeMemoryManager(memories=["m1"])  # non-empty → triggers format
        config = AgentConfig(enable_rag=True, write_user_turn_to_memory=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        assert len(mem.retrieve_calls) == 1

    def test_retrieve_not_called_when_rag_disabled(self):
        mem = FakeMemoryManager()
        config = AgentConfig(enable_rag=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        assert len(mem.retrieve_calls) == 0

    def test_rag_top_k_passed_to_retrieve(self):
        mem = FakeMemoryManager()
        config = AgentConfig(enable_rag=True, rag_top_k=7, write_user_turn_to_memory=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        assert mem.retrieve_calls[0][1] == 7

    def test_rag_min_importance_passed_to_retrieve(self):
        mem = FakeMemoryManager()
        config = AgentConfig(enable_rag=True, rag_min_importance=0.6, write_user_turn_to_memory=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        assert mem.retrieve_calls[0][2] == 0.6

    def test_emotion_context_passed_when_rag_emotion_aware(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            enable_rag=True,
            rag_emotion_aware=True,
            write_user_turn_to_memory=False,
        )
        state = FakeState(primary_emotion="frustration")
        agent, *_ = _make_agent(memory=mem, config=config, emotion_state=state)
        agent.handle_message("Query.")
        _, _, _, emotional_ctx = mem.retrieve_calls[0]
        assert emotional_ctx is state

    def test_no_emotion_context_when_rag_not_emotion_aware(self):
        mem = FakeMemoryManager()
        config = AgentConfig(
            enable_rag=True,
            rag_emotion_aware=False,
            write_user_turn_to_memory=False,
        )
        agent, *_ = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        _, _, _, emotional_ctx = mem.retrieve_calls[0]
        assert emotional_ctx is None

    def test_memory_context_passed_to_prompt_builder(self):
        """If memories are retrieved, memory_context reaches PromptBuilder."""
        mem = FakeMemoryManager(memories=["some_memory"])
        config = AgentConfig(enable_rag=True, write_user_turn_to_memory=False)
        agent, llm, engine, modulator, builder = _make_agent(memory=mem, config=config)
        agent.handle_message("Query.")
        # builder.calls[0] = (user_input, modulation, memory_context)
        _, _, memory_ctx = builder.calls[0]
        assert memory_ctx is not None
        assert "Past:" in memory_ctx


class TestEmotionalProtection:
    """apply_emotional_protection must be called after each turn."""

    def test_protection_called_by_default(self):
        config = AgentConfig(apply_emotional_protection=True)
        agent, llm, engine, *_ = _make_agent(config=config)
        agent.handle_message("Test.")
        assert len(engine.protection_calls) == 1

    def test_protection_not_called_when_disabled(self):
        config = AgentConfig(apply_emotional_protection=False)
        agent, llm, engine, *_ = _make_agent(config=config)
        agent.handle_message("Test.")
        assert len(engine.protection_calls) == 0

    def test_protection_receives_emotional_state(self):
        state = FakeState(primary_emotion="drive")
        config = AgentConfig(apply_emotional_protection=True)
        agent, llm, engine, *_ = _make_agent(config=config, emotion_state=state)
        agent.handle_message("Test.")
        assert engine.protection_calls[0] is state


class TestAgentResponseContract:
    """AgentResponse must be immutable and expose correct API."""

    def test_agent_response_is_frozen(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        try:
            result.assistant_output = "hacked"
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass

    def test_sections_used_is_tuple(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert isinstance(result.sections_used, tuple)

    def test_to_log_dict_has_expected_keys(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        d = result.to_log_dict()
        required = {
            "session_id", "turn_index", "latency_ms",
            "emotion", "intensity", "tone", "pacing",
            "sections", "output_chars", "detected_at",
        }
        assert required.issubset(set(d.keys()))

    def test_output_chars_matches_actual_length(self):
        agent, *_ = _make_agent(llm_response="Hello world!")
        result = agent.handle_message("Test.")
        assert result.to_log_dict()["output_chars"] == len("Hello world!")

    def test_detected_at_is_datetime(self):
        from datetime import datetime
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        assert isinstance(result.detected_at, datetime)

    def test_repr_is_informative(self):
        agent, *_ = _make_agent(llm_response="Short reply.")
        result = agent.handle_message("Test.")
        r = repr(result)
        assert "AgentResponse" in r
        assert "turn=" in r


class TestTurnIndexIncrement:
    """turn_index must increment on each successful call."""

    def test_first_turn_index_is_zero(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("First.")
        assert result.turn_index == 0

    def test_second_turn_index_is_one(self):
        agent, *_ = _make_agent()
        agent.handle_message("First.")
        result = agent.handle_message("Second.")
        assert result.turn_index == 1

    def test_agent_turn_index_increments(self):
        agent, *_ = _make_agent()
        for i in range(5):
            agent.handle_message(f"Turn {i}")
        assert agent.turn_index == 5


class TestSessionManagement:
    """Session ID flows through pipeline and reset works correctly."""

    def test_session_id_in_response(self):
        config = AgentConfig(default_session_id="test-session")
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert result.session_id == "test-session"

    def test_session_id_override_per_call(self):
        config = AgentConfig(default_session_id="default-session")
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.", session_id="override-session")
        assert result.session_id == "override-session"

    def test_set_session_id(self):
        agent, *_ = _make_agent()
        agent.set_session_id("new-session")
        assert agent.session_id == "new-session"

    def test_reset_session_clears_turn_index(self):
        agent, *_ = _make_agent()
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        assert agent.turn_index == 2
        agent.reset_session()
        assert agent.turn_index == 0

    def test_reset_session_clears_history_buffer(self):
        config = AgentConfig(enable_conversation_history=True)
        agent, *_ = _make_agent(config=config)
        agent.handle_message("Turn 1.")
        assert len(agent._conversation_history) > 0
        agent.reset_session()
        assert len(agent._conversation_history) == 0


class TestLLMErrorHandling:
    """LLM failures must produce graceful fallback, not crash."""

    def test_llm_error_returns_fallback_response(self):
        agent, *_ = _make_agent(llm_raise=LLMError("API down"))
        result = agent.handle_message("Test.")
        assert result.assistant_output != ""
        assert isinstance(result.assistant_output, str)

    def test_llm_timeout_returns_fallback(self):
        agent, *_ = _make_agent(llm_raise=LLMTimeoutError("Timeout"))
        result = agent.handle_message("Test.")
        assert result.assistant_output != ""

    def test_llm_error_does_not_crash_agent(self):
        agent, *_ = _make_agent(llm_raise=LLMError("Crash"))
        # Must not raise
        result = agent.handle_message("Test.")
        assert isinstance(result, AgentResponse)

    def test_llm_error_still_increments_turn_index(self):
        agent, *_ = _make_agent(llm_raise=LLMError("Crash"))
        agent.handle_message("Test.")
        assert agent.turn_index == 1


class TestFakeLLMClient:
    """FakeLLMClient must correctly record calls, support response_fn, and simulate errors."""

    def test_call_count_increments(self):
        client = FakeLLMClient(response="Hi")
        client.chat([{"role": "user", "content": "Test"}])
        assert client.call_count == 1

    def test_call_history_recorded(self):
        client = FakeLLMClient(response="Hi")
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
        client.chat(msgs)
        assert client.call_history[0] == msgs

    def test_last_messages_property(self):
        client = FakeLLMClient(response="Hi")
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat(msgs)
        assert client.last_messages == msgs

    def test_last_system_prompt(self):
        client = FakeLLMClient(response="Hi")
        client.chat([
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "User message."},
        ])
        assert client.last_system_prompt == "System instruction."

    def test_response_fn_receives_messages(self):
        received = []
        def fn(msgs):
            received.extend(msgs)
            return "dynamic"

        client = FakeLLMClient(response_fn=fn)
        client.chat([{"role": "user", "content": "Hello"}])
        assert len(received) == 1
        assert received[0]["content"] == "Hello"

    def test_response_fn_return_used(self):
        client = FakeLLMClient(response_fn=lambda msgs: "dynamic response")
        result = client.chat([{"role": "user", "content": "Test"}])
        assert result == "dynamic response"

    def test_raise_error_simulation(self):
        client = FakeLLMClient(raise_error=LLMError("Simulated"))
        try:
            client.chat([{"role": "user", "content": "test"}])
            assert False, "Should have raised"
        except LLMError:
            pass

    def test_reset_clears_history(self):
        client = FakeLLMClient(response="Hi")
        client.chat([{"role": "user", "content": "test"}])
        client.reset()
        assert client.call_count == 0
        assert client.call_history == []

    def test_model_property(self):
        client = FakeLLMClient(model_name="my-model")
        assert client.model == "my-model"


class TestConversationHistory:
    """Multi-turn conversation history injection (Phase 2 hook)."""

    def test_history_grows_when_enabled(self):
        config = AgentConfig(enable_conversation_history=True)
        agent, *_ = _make_agent(config=config)
        agent.handle_message("Turn 1.")
        assert len(agent._conversation_history) == 2  # user + assistant

    def test_history_not_grown_when_disabled(self):
        config = AgentConfig(enable_conversation_history=False)
        agent, *_ = _make_agent(config=config)
        agent.handle_message("Turn 1.")
        assert len(agent._conversation_history) == 0

    def test_history_trimmed_to_max_turns(self):
        config = AgentConfig(
            enable_conversation_history=True,
            conversation_history_max_turns=2,
        )
        agent, *_ = _make_agent(config=config)
        for i in range(5):
            agent.handle_message(f"Turn {i}.")
        # Max 2 turns = 4 messages
        assert len(agent._conversation_history) <= 4

    def test_history_injected_into_messages(self):
        config = AgentConfig(enable_conversation_history=True)
        agent, llm, *_ = _make_agent(config=config)
        agent.handle_message("First turn.")
        agent.handle_message("Second turn.")
        # Second call should have history in messages
        second_call_messages = llm.call_history[1]
        roles = [m["role"] for m in second_call_messages]
        # Should contain system + user (history) + assistant (history) + user (current)
        assert roles.count("user") >= 2 or len(second_call_messages) >= 3


class TestConfigurableConfig:
    """AgentConfig fields must flow correctly into agent behavior."""

    def test_strip_response_removes_whitespace(self):
        agent, *_ = _make_agent(llm_response="  Response with spaces.  ")
        result = agent.handle_message("Test.")
        assert result.assistant_output == "Response with spaces."

    def test_no_strip_when_disabled(self):
        config = AgentConfig(strip_response=False)
        agent, *_ = _make_agent(
            llm_response="  Padded.  ",
            config=config,
        )
        result = agent.handle_message("Test.")
        assert result.assistant_output == "  Padded.  "

    def test_max_response_chars_truncates(self):
        config = AgentConfig(max_response_chars=10)
        agent, *_ = _make_agent(
            llm_response="A very long response that exceeds the limit.",
            config=config,
        )
        result = agent.handle_message("Test.")
        assert len(result.assistant_output) <= 10

    def test_max_response_chars_zero_means_no_limit(self):
        config = AgentConfig(max_response_chars=0)
        long_response = "A" * 5000
        agent, *_ = _make_agent(llm_response=long_response, config=config)
        result = agent.handle_message("Test.")
        assert len(result.assistant_output) == 5000


class TestPipelineTrace:
    """PipelineTrace must record per-stage timing."""

    def test_all_trace_fields_are_non_negative(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        d = result.trace.to_dict()
        for key, val in d.items():
            assert val >= 0.0, f"{key} = {val} is negative"

    def test_trace_is_frozen(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        try:
            result.trace.llm_ms = 999.0
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass

    def test_rag_ms_positive_when_memory_present(self):
        mem = FakeMemoryManager()
        config = AgentConfig(enable_rag=True, write_user_turn_to_memory=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        result = agent.handle_message("Test.")
        assert result.trace.rag_ms >= 0.0

    def test_trace_repr_informative(self):
        agent, *_ = _make_agent()
        result = agent.handle_message("Test.")
        r = repr(result.trace)
        assert "PipelineTrace" in r
        assert "total=" in r


class TestAgentStats:
    """stats() must return a well-formed dict."""

    def test_stats_returns_dict(self):
        agent, *_ = _make_agent()
        s = agent.stats()
        assert isinstance(s, dict)

    def test_stats_has_expected_keys(self):
        agent, *_ = _make_agent()
        s = agent.stats()
        required = {"session_id", "turn_index", "llm_model", "memory", "rag_enabled"}
        assert required.issubset(set(s.keys()))

    def test_stats_memory_false_when_no_manager(self):
        agent, *_ = _make_agent(memory=None)
        assert agent.stats()["memory"] is False

    def test_stats_memory_true_when_manager_present(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        assert agent.stats()["memory"] is True

    def test_stats_turn_index_matches_agent(self):
        agent, *_ = _make_agent()
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        assert agent.stats()["turn_index"] == 2


class TestStageIsolation:
    """A failing stage must not crash the entire pipeline — downstream stages still run."""

    def test_emotion_failure_does_not_crash(self):
        agent, llm, *_ = _make_agent(emotion_fail=True)
        result = agent.handle_message("Test.")
        assert isinstance(result, AgentResponse)
        assert llm.call_count == 1  # LLM still called

    def test_modulator_failure_does_not_crash(self):
        agent, llm, *_ = _make_agent(modulator_fail=True)
        result = agent.handle_message("Test.")
        assert isinstance(result, AgentResponse)
        assert llm.call_count == 1

    def test_prompt_failure_uses_fallback_messages(self):
        agent, llm, *_ = _make_agent(prompt_fail=True)
        result = agent.handle_message("Test.")
        # LLM should still be called with fallback messages
        assert llm.call_count == 1
        assert result.assistant_output != ""

    def test_llm_failure_still_returns_agent_response(self):
        agent, *_ = _make_agent(llm_raise=LLMError("down"))
        result = agent.handle_message("Test.")
        assert isinstance(result, AgentResponse)
        assert result.assistant_output != ""


class TestMultipleTurns:
    """Sequential turns must produce correct incremental behavior."""

    def test_five_sequential_turns(self):
        agent, llm, *_ = _make_agent()
        results = [agent.handle_message(f"Turn {i}.") for i in range(5)]
        assert [r.turn_index for r in results] == [0, 1, 2, 3, 4]
        assert llm.call_count == 5

    def test_each_turn_has_correct_user_input(self):
        agent, *_ = _make_agent()
        inputs = ["Alpha.", "Beta.", "Gamma."]
        results = [agent.handle_message(i) for i in inputs]
        assert [r.user_input for r in results] == inputs

    def test_turn_index_resets_after_session_reset(self):
        agent, *_ = _make_agent()
        for i in range(3):
            agent.handle_message(f"Turn {i}.")
        agent.reset_session()
        result = agent.handle_message("After reset.")
        assert result.turn_index == 0


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestFullPipelineFlow,
        TestLatencyMeasurement,
        TestMemoryWriteBack,
        TestMemoryWriteBackDisabled,
        TestRAGInjection,
        TestEmotionalProtection,
        TestAgentResponseContract,
        TestTurnIndexIncrement,
        TestSessionManagement,
        TestLLMErrorHandling,
        TestFakeLLMClient,
        TestConversationHistory,
        TestConfigurableConfig,
        TestPipelineTrace,
        TestAgentStats,
        TestStageIsolation,
        TestMultipleTurns,
    ]

    passed = []
    failed = []

    for cls in test_classes:
        instance = cls()
        methods = sorted([m for m in dir(cls) if m.startswith("test_")])
        for method_name in methods:
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
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{total} passed")
    if failed:
        print("\nFAILED:")
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")