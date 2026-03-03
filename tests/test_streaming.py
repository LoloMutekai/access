"""
A.C.C.E.S.S. — Streaming Test Suite

Coverage:
    TestFakeLLMStreaming          — FakeLLMClient.stream() deterministic behavior
    TestStreamMessageYieldsTokens — tokens yielded progressively, llm.stream() called
    TestFinalizeStream            — builds valid AgentResponse from collected tokens
    TestStreamStateMachine        — is_streaming flag, double-stream guard, context clearing
    TestStreamErrorHandling       — stream errors yield fallback, finalize still works
    TestStreamNoWriteBackDuring   — memory not written during streaming, only on finalize
    TestStreamProtection          — emotional protection called in finalize, not during stream
    TestStreamTurnIndex           — turn_index increments only on finalize, not stream_message
    TestStreamVsBlocking          — same pre-LLM stages fired for both modes
    TestStreamFinalizePairings    — finalize without stream raises, stream without finalize
    TestStreamConversationHistory — history updated only on finalize
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Optional, Iterator

from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.models import AgentResponse, StreamContext
from agent.llm_client import FakeLLMClient, LLMStreamError


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FAKES (same as test_agent.py — no imports between test files)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakePAD:
    valence: float = 0.0; arousal: float = 0.5; dominance: float = 0.5

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; intensity: float = 0.5
    confidence: float = 0.8; pad: FakePAD = None
    is_positive: bool = False; is_negative: bool = False; label: str = "moderate neutral"
    def __post_init__(self):
        if self.pad is None: self.pad = FakePAD()

@dataclass
class FakeModulation:
    tone: str = "neutral"; pacing: str = "normal"; verbosity: str = "normal"
    structure_bias: str = "conversational"; emotional_validation: bool = False
    motivational_bias: float = 0.0; cognitive_load_limit: float = 1.0
    active_strategies: tuple = ()

@dataclass
class FakeBuiltPrompt:
    system_prompt: str = "System."; sections: tuple = ("tone",); char_count: int = 10
    def to_api_messages(self):
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "test"}]
    def has_section(self, name): return name in self.sections

class FakeEmotionEngine:
    def __init__(self, state=None, fail=False):
        self._state = state or FakeState()
        self._fail = fail
        self.process_calls = []
        self.protection_calls = []
    def process_interaction(self, text, session_id=None):
        self.process_calls.append(text)
        if self._fail: raise RuntimeError("forced")
        return self._state
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, state): self.protection_calls.append(state)
    def stats(self): return {}

class FakeModulator:
    def __init__(self): self.calls = []
    def build_modulation(self, state, trend, dominant_pattern=None):
        self.calls.append((state, trend)); return FakeModulation()

class FakePromptBuilder:
    def __init__(self): self.calls = []
    def build(self, user_input, modulation, memory_context=None):
        self.calls.append((user_input, memory_context)); return FakeBuiltPrompt()

class FakeMemoryManager:
    def __init__(self):
        self.add_calls = []; self.retrieve_calls = []
    def retrieve_relevant_memories(self, query, top_k=5, min_importance=0.0, emotional_context=None):
        self.retrieve_calls.append(query); return []
    def format_for_rag(self, memories): return ""
    def add_memory(self, **kwargs): self.add_calls.append(kwargs)

def _make_agent(llm=None, config=None, memory=None, engine=None, modulator=None, builder=None):
    llm = llm or FakeLLMClient(response="Default response.")
    engine = engine or FakeEmotionEngine()
    modulator = modulator or FakeModulator()
    builder = builder or FakePromptBuilder()
    return AgentCore(
        emotion_engine=engine,
        conversation_modulator=modulator,
        prompt_builder=builder,
        llm_client=llm,
        memory_manager=memory,
        config=config or AgentConfig(),
    ), llm, engine, modulator, builder


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFakeLLMStreaming:
    """FakeLLMClient.stream() deterministic behavior."""

    def test_stream_yields_tokens(self):
        client = FakeLLMClient(response="Hello world!")
        tokens = list(client.stream([{"role": "user", "content": "Hi"}]))
        assert len(tokens) >= 1
        assert "".join(tokens) == "Hello world!"

    def test_stream_call_count_increments(self):
        client = FakeLLMClient(response="Hello.")
        list(client.stream([{"role": "user", "content": "Hi"}]))
        assert client.stream_call_count == 1

    def test_stream_separate_from_chat_count(self):
        client = FakeLLMClient(response="X")
        client.chat([{"role": "user", "content": "a"}])
        list(client.stream([{"role": "user", "content": "b"}]))
        assert client.call_count == 1
        assert client.stream_call_count == 1

    def test_stream_call_history_recorded(self):
        client = FakeLLMClient(response="X")
        msgs = [{"role": "user", "content": "Test"}]
        list(client.stream(msgs))
        assert client.last_stream_messages == msgs

    def test_stream_char_level_splitter(self):
        client = FakeLLMClient(response="ABC", token_splitter=list)
        tokens = list(client.stream([{"role": "user", "content": "Hi"}]))
        assert tokens == ["A", "B", "C"]

    def test_stream_error_raises_immediately(self):
        client = FakeLLMClient(stream_raise_error=LLMStreamError("cut"))
        try:
            list(client.stream([{"role": "user", "content": "test"}]))
            assert False, "Should have raised"
        except LLMStreamError:
            pass

    def test_stream_response_fn_used(self):
        client = FakeLLMClient(response_fn=lambda msgs: "Dynamic!")
        tokens = list(client.stream([{"role": "user", "content": "X"}]))
        assert "".join(tokens) == "Dynamic!"

    def test_reset_clears_stream_history(self):
        client = FakeLLMClient(response="Hi")
        list(client.stream([{"role": "user", "content": "test"}]))
        client.reset()
        assert client.stream_call_count == 0
        assert client.stream_call_history == []

    def test_default_word_splitter_multi_word(self):
        client = FakeLLMClient(response="One two three")
        tokens = list(client.stream([{"role": "user", "content": "Hi"}]))
        reconstructed = "".join(tokens)
        assert reconstructed == "One two three"
        assert len(tokens) >= 2  # at least word-level splitting


class TestStreamMessageYieldsTokens:
    """stream_message() runs pre-LLM stages and yields tokens."""

    def test_yields_non_empty_tokens(self):
        llm = FakeLLMClient(response="Hello streaming world!")
        agent, *_ = _make_agent(llm=llm)
        tokens = list(agent.stream_message("Test."))
        assert len(tokens) >= 1

    def test_tokens_reconstruct_full_response(self):
        llm = FakeLLMClient(response="Full response text.")
        agent, *_ = _make_agent(llm=llm)
        tokens = list(agent.stream_message("Test."))
        assert "".join(tokens) == "Full response text."

    def test_llm_stream_called_once(self):
        llm = FakeLLMClient(response="Response.")
        agent, *_ = _make_agent(llm=llm)
        list(agent.stream_message("Test."))
        assert llm.stream_call_count == 1
        assert llm.call_count == 0  # blocking NOT called

    def test_emotion_engine_called(self):
        llm = FakeLLMClient(response="R.")
        engine = FakeEmotionEngine()
        agent, *_ = _make_agent(llm=llm, engine=engine)
        list(agent.stream_message("Input."))
        assert len(engine.process_calls) == 1
        assert engine.process_calls[0] == "Input."

    def test_prompt_builder_called(self):
        llm = FakeLLMClient(response="R.")
        builder = FakePromptBuilder()
        agent, *_ = _make_agent(llm=llm, builder=builder)
        list(agent.stream_message("Input."))
        assert len(builder.calls) == 1

    def test_stream_context_stored_during_streaming(self):
        """is_streaming flag is True while stream is active."""
        llm = FakeLLMClient(response="Token by token.")
        agent, *_ = _make_agent(llm=llm)
        # Consume first token then check
        gen = agent.stream_message("Test.")
        next(gen)  # consume first token — context should be set at this point
        assert agent.is_streaming

    def test_stream_context_none_before_stream(self):
        agent, *_ = _make_agent()
        assert not agent.is_streaming


class TestFinalizeStream:
    """finalize_stream() builds a valid AgentResponse."""

    def _stream_and_finalize(self, agent, user_input="Test."):
        tokens = list(agent.stream_message(user_input))
        full = "".join(tokens)
        return agent.finalize_stream(full), full

    def test_returns_agent_response(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent)
        assert isinstance(response, AgentResponse)

    def test_assistant_output_matches_collected_tokens(self):
        llm = FakeLLMClient(response="Streamed output.")
        agent, *_ = _make_agent(llm=llm)
        response, full = self._stream_and_finalize(agent)
        assert response.assistant_output == full.strip()

    def test_user_input_preserved(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent, "My specific input.")
        assert response.user_input == "My specific input."

    def test_emotional_state_in_response(self):
        state = FakeState(primary_emotion="confidence")
        engine = FakeEmotionEngine(state=state)
        agent, *_ = _make_agent(engine=engine)
        response, _ = self._stream_and_finalize(agent)
        assert response.emotional_state is state

    def test_sections_used_is_tuple(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent)
        assert isinstance(response.sections_used, tuple)

    def test_latency_ms_positive(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent)
        assert response.latency_ms > 0.0

    def test_trace_present(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent)
        assert response.trace is not None

    def test_stream_context_cleared_after_finalize(self):
        agent, *_ = _make_agent()
        self._stream_and_finalize(agent)
        assert not agent.is_streaming

    def test_agent_response_is_frozen(self):
        agent, *_ = _make_agent()
        response, _ = self._stream_and_finalize(agent)
        try:
            response.assistant_output = "hacked"
            assert False
        except (AttributeError, TypeError):
            pass


class TestStreamStateMachine:
    """is_streaming flag and context state machine."""

    def test_is_streaming_false_initially(self):
        agent, *_ = _make_agent()
        assert not agent.is_streaming

    def test_is_streaming_true_after_stream_starts(self):
        llm = FakeLLMClient(response="A B C")
        agent, *_ = _make_agent(llm=llm)
        gen = agent.stream_message("Test.")
        next(gen)  # start the generator
        assert agent.is_streaming

    def test_is_streaming_false_after_finalize(self):
        agent, *_ = _make_agent()
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        assert not agent.is_streaming

    def test_finalize_without_stream_raises(self):
        agent, *_ = _make_agent()
        try:
            agent.finalize_stream("orphaned response")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "finalize_stream" in str(e)

    def test_reset_clears_stream_context(self):
        agent, *_ = _make_agent()
        list(agent.stream_message("Test."))  # start but don't finalize
        agent.reset_session()
        assert not agent.is_streaming


class TestStreamErrorHandling:
    """Stream errors should yield fallback, not crash; finalize should still work."""

    def test_stream_error_yields_fallback_not_crash(self):
        llm = FakeLLMClient(stream_raise_error=LLMStreamError("Network cut"))
        agent, *_ = _make_agent(llm=llm)
        tokens = list(agent.stream_message("Test."))
        assert len(tokens) > 0
        assert isinstance(tokens[0], str)

    def test_stream_error_context_still_stored(self):
        """Even when stream fails, StreamContext must be stored for finalize."""
        llm = FakeLLMClient(stream_raise_error=LLMStreamError("err"))
        agent, *_ = _make_agent(llm=llm)
        tokens = list(agent.stream_message("Test."))
        # Context should still be set so finalize can run
        assert agent._stream_context is not None

    def test_finalize_works_after_stream_error(self):
        llm = FakeLLMClient(stream_raise_error=LLMStreamError("err"))
        agent, *_ = _make_agent(llm=llm)
        tokens = list(agent.stream_message("Test."))
        fallback = "".join(tokens)
        # Should not raise
        response = agent.finalize_stream(fallback)
        assert isinstance(response, AgentResponse)

    def test_stream_error_does_not_increment_turn_index(self):
        """Turn index only increments on finalize, not on stream error."""
        llm = FakeLLMClient(stream_raise_error=LLMStreamError("err"))
        agent, *_ = _make_agent(llm=llm)
        assert agent.turn_index == 0
        list(agent.stream_message("Test."))
        assert agent.turn_index == 0  # not yet finalized


class TestStreamNoWriteBackDuring:
    """Memory write-back happens only on finalize, not during streaming."""

    def test_no_memory_write_during_stream_message(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True, enable_rag=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        list(agent.stream_message("Store me."))
        # No write-back until finalize
        assert len(mem.add_calls) == 0

    def test_memory_written_on_finalize(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True, enable_rag=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        tokens = list(agent.stream_message("Store me."))
        agent.finalize_stream("".join(tokens))
        assert len(mem.add_calls) >= 1

    def test_user_content_in_memory_write(self):
        mem = FakeMemoryManager()
        config = AgentConfig(write_user_turn_to_memory=True, enable_rag=False)
        agent, *_ = _make_agent(memory=mem, config=config)
        tokens = list(agent.stream_message("Specific content to store."))
        agent.finalize_stream("".join(tokens))
        written_contents = [c["content"] for c in mem.add_calls]
        assert any("Specific content to store." in c for c in written_contents)


class TestStreamProtection:
    """Emotional protection called in finalize, not during streaming."""

    def test_no_protection_during_stream_message(self):
        engine = FakeEmotionEngine()
        agent, *_ = _make_agent(engine=engine)
        list(agent.stream_message("Test."))
        assert len(engine.protection_calls) == 0

    def test_protection_called_on_finalize(self):
        engine = FakeEmotionEngine()
        config = AgentConfig(apply_emotional_protection=True)
        agent, *_ = _make_agent(engine=engine, config=config)
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        assert len(engine.protection_calls) == 1

    def test_protection_not_called_when_disabled(self):
        engine = FakeEmotionEngine()
        config = AgentConfig(apply_emotional_protection=False)
        agent, *_ = _make_agent(engine=engine, config=config)
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        assert len(engine.protection_calls) == 0


class TestStreamTurnIndex:
    """turn_index increments only on finalize, not on stream_message."""

    def test_turn_index_zero_before_anything(self):
        agent, *_ = _make_agent()
        assert agent.turn_index == 0

    def test_turn_index_unchanged_after_stream_message(self):
        agent, *_ = _make_agent()
        list(agent.stream_message("Test."))
        assert agent.turn_index == 0  # not incremented yet

    def test_turn_index_incremented_after_finalize(self):
        agent, *_ = _make_agent()
        tokens = list(agent.stream_message("Test."))
        agent.finalize_stream("".join(tokens))
        assert agent.turn_index == 1

    def test_turn_index_in_finalize_response_matches_pre_increment(self):
        agent, *_ = _make_agent()
        tokens = list(agent.stream_message("Test."))
        response = agent.finalize_stream("".join(tokens))
        assert response.turn_index == 0  # the turn index AT stream time

    def test_sequential_stream_turns_increment_correctly(self):
        agent, *_ = _make_agent()
        for _ in range(3):
            tokens = list(agent.stream_message("Turn."))
            agent.finalize_stream("".join(tokens))
        assert agent.turn_index == 3


class TestStreamVsBlocking:
    """Same pre-LLM stages are invoked for both streaming and blocking modes."""

    def test_emotion_called_same_number_of_times(self):
        engine_block = FakeEmotionEngine()
        engine_stream = FakeEmotionEngine()
        llm = FakeLLMClient(response="R.")

        agent_block, *_ = _make_agent(llm=llm, engine=engine_block)
        agent_block.handle_message("Test.")

        llm2 = FakeLLMClient(response="R.")
        agent_stream, *_ = _make_agent(llm=llm2, engine=engine_stream)
        tokens = list(agent_stream.stream_message("Test."))
        agent_stream.finalize_stream("".join(tokens))

        assert len(engine_block.process_calls) == len(engine_stream.process_calls) == 1

    def test_prompt_builder_called_in_both_modes(self):
        builder_b = FakePromptBuilder()
        builder_s = FakePromptBuilder()

        agent_b, *_ = _make_agent(builder=builder_b)
        agent_b.handle_message("Test.")

        agent_s, *_ = _make_agent(builder=builder_s)
        tokens = list(agent_s.stream_message("Test."))
        agent_s.finalize_stream("".join(tokens))

        assert len(builder_b.calls) == 1
        assert len(builder_s.calls) == 1


class TestStreamFinalizePairings:
    """Correct pairing enforcement between stream_message and finalize_stream."""

    def test_second_stream_message_warns_but_does_not_crash(self):
        """Calling stream_message while streaming active should warn, not crash."""
        agent, *_ = _make_agent()
        list(agent.stream_message("First."))
        # Don't finalize — call stream again (should warn, not crash)
        tokens = list(agent.stream_message("Second."))
        assert len(tokens) >= 1

    def test_finalize_after_double_stream_uses_latest_context(self):
        llm = FakeLLMClient(response="R2.")
        agent, *_ = _make_agent(llm=llm)
        list(agent.stream_message("First."))
        tokens = list(agent.stream_message("Second."))
        response = agent.finalize_stream("".join(tokens))
        assert response.user_input == "Second."


class TestStreamConversationHistory:
    """History buffer updated only on finalize_stream, not during streaming."""

    def test_history_empty_during_streaming(self):
        config = AgentConfig(enable_conversation_history=True)
        agent, *_ = _make_agent(config=config)
        list(agent.stream_message("Turn 1."))
        # History not yet updated
        assert len(agent._conversation_history) == 0

    def test_history_populated_after_finalize(self):
        config = AgentConfig(enable_conversation_history=True)
        agent, *_ = _make_agent(config=config)
        tokens = list(agent.stream_message("Turn 1."))
        agent.finalize_stream("".join(tokens))
        assert len(agent._conversation_history) == 2  # user + assistant


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestFakeLLMStreaming,
        TestStreamMessageYieldsTokens,
        TestFinalizeStream,
        TestStreamStateMachine,
        TestStreamErrorHandling,
        TestStreamNoWriteBackDuring,
        TestStreamProtection,
        TestStreamTurnIndex,
        TestStreamVsBlocking,
        TestStreamFinalizePairings,
        TestStreamConversationHistory,
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
        print("\nFAILED:")
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")