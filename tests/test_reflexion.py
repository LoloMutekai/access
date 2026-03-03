"""
A.C.C.E.S.S. — Reflection Test Suite

TestReflectionResult        — frozen, fields, to_log_dict, UTC timestamp
TestReflectionEngineScoring — importance score heuristics
TestReflectionEngineGoal    — goal signal detection
TestReflectionEngineTrajectory — trajectory signal detection
TestReflectionEngineSummary — summary builder
TestReflectionEngineError   — failure isolation
TestReflectionProtocol      — structural typing
TestAgentCoreReflection     — integration: called per turn, stored in response
TestReflectionDisabled      — enable_reflection=False: no reflection
TestReflectionMemory        — adaptive_importance, use_reflection_summary
TestReflectionStreaming      — reflection triggered on finalize_stream, not stream_message
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Optional

from agent.reflection_engine import ReflectionEngine, ReflectionResult, ReflectionConfig, ReflectionEngineProtocol
from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient


# ─── Fakes ───────────────────────────────────────────────────────────────────

@dataclass
class FakePAD:
    valence: float = 0.0; arousal: float = 0.5; dominance: float = 0.5

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; intensity: float = 0.5
    confidence: float = 0.8; pad: FakePAD = None
    is_positive: bool = False; is_negative: bool = False
    is_high_arousal: bool = False; label: str = "moderate neutral"
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
    sections: tuple = ("tone",)
    def to_api_messages(self): return [{"role":"system","content":"S"},{"role":"user","content":"u"}]

class FakeEngine:
    def __init__(self, state=None):
        self._state = state or FakeState()
        self.protection_calls = []
    def process_interaction(self, t, session_id=None): return self._state
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, s): self.protection_calls.append(s)
    def stats(self): return {}

class FakeModulator:
    def build_modulation(self, state, trend, dominant_pattern=None): return FakeModulation()

class FakeBuilder:
    def build(self, user_input, modulation, memory_context=None): return FakeBuiltPrompt()

class FakeMemory:
    def __init__(self): self.add_calls = []
    def retrieve_relevant_memories(self, **kw): return []
    def format_for_rag(self, m): return ""
    def add_memory(self, **kw): self.add_calls.append(kw)

def _make_agent(llm=None, config=None, state=None, memory=None,
                reflection_engine=None, engine=None):
    llm = llm or FakeLLMClient(response="Agent response.")
    engine = engine or FakeEngine(state=state)
    return AgentCore(
        emotion_engine=engine,
        conversation_modulator=FakeModulator(),
        prompt_builder=FakeBuilder(),
        llm_client=llm,
        memory_manager=memory,
        config=config or AgentConfig(enable_rag=False, apply_emotional_protection=False),
        reflection_engine=reflection_engine,
    ), llm, engine


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestReflectionResult:
    def test_frozen(self):
        r = ReflectionResult(summary="s", importance_score=0.5,
                             emotional_tags=[], goal_signal=None, trajectory_signal=None)
        try: r.summary = "x"; assert False
        except (AttributeError, TypeError): pass

    def test_importance_score_present(self):
        r = ReflectionResult(summary="s", importance_score=0.7,
                             emotional_tags=["frustration"], goal_signal="stabilize",
                             trajectory_signal="declining")
        assert r.importance_score == 0.7

    def test_to_log_dict_keys(self):
        r = ReflectionResult(summary="s", importance_score=0.5, emotional_tags=[],
                             goal_signal=None, trajectory_signal=None)
        d = r.to_log_dict()
        assert {"summary", "importance_score", "emotional_tags",
                "goal_signal", "trajectory_signal", "reflected_at"}.issubset(set(d.keys()))

    def test_reflected_at_is_utc(self):
        r = ReflectionResult(summary="s", importance_score=0.5, emotional_tags=[],
                             goal_signal=None, trajectory_signal=None)
        assert r.reflected_at.tzinfo is not None

    def test_repr_contains_importance(self):
        r = ReflectionResult(summary="test", importance_score=0.88,
                             emotional_tags=[], goal_signal="push_forward", trajectory_signal=None)
        assert "0.88" in repr(r)


class TestReflectionEngineScoring:
    def _engine(self): return ReflectionEngine()

    def _state(self, intensity=0.5, is_negative=False, is_positive=False, is_high_arousal=False):
        return FakeState(intensity=intensity, is_negative=is_negative,
                        is_positive=is_positive, is_high_arousal=is_high_arousal)

    def _mod(self, bias=0.0, validation=False):
        return FakeModulation(motivational_bias=bias, emotional_validation=validation)

    def test_base_importance(self):
        r = self._engine().reflect("input", "output", self._state(), self._mod(), ())
        assert 0.0 <= r.importance_score <= 1.0

    def test_high_intensity_increases_score(self):
        low = self._engine().reflect("x", "y", self._state(intensity=0.3), self._mod(), ())
        high = self._engine().reflect("x", "y", self._state(intensity=0.9), self._mod(), ())
        assert high.importance_score > low.importance_score

    def test_tool_used_increases_score(self):
        from tools.base_tool import ToolResult
        no_tool = self._engine().reflect("x", "y", self._state(), self._mod(), ())
        tool_result = ToolResult(tool_name="echo", success=True, output={})
        with_tool = self._engine().reflect("x", "y", self._state(), self._mod(), (tool_result,))
        assert with_tool.importance_score > no_tool.importance_score

    def test_strong_bias_increases_score(self):
        neutral = self._engine().reflect("x", "y", self._state(), self._mod(bias=0.0), ())
        strong = self._engine().reflect("x", "y", self._state(), self._mod(bias=0.8), ())
        assert strong.importance_score > neutral.importance_score

    def test_long_output_increases_score(self):
        short = self._engine().reflect("x", "a", self._state(), self._mod(), ())
        long_out = "x" * 500
        long = self._engine().reflect("x", long_out, self._state(), self._mod(), ())
        assert long.importance_score > short.importance_score

    def test_negative_emotion_increases_score(self):
        pos = self._engine().reflect("x", "y", self._state(is_negative=False), self._mod(), ())
        neg = self._engine().reflect("x", "y", self._state(is_negative=True), self._mod(), ())
        assert neg.importance_score > pos.importance_score

    def test_validation_increases_score(self):
        no_val = self._engine().reflect("x", "y", self._state(), self._mod(validation=False), ())
        val = self._engine().reflect("x", "y", self._state(), self._mod(validation=True), ())
        assert val.importance_score > no_val.importance_score

    def test_importance_clamped_to_one(self):
        # Maximize all bonuses
        from tools.base_tool import ToolResult
        state = self._state(intensity=0.9, is_negative=True)
        mod = self._mod(bias=0.9, validation=True)
        long_out = "x" * 600
        r = self._engine().reflect("x", long_out, state, mod,
                                    (ToolResult(tool_name="e", success=True, output={}),))
        assert r.importance_score <= 1.0

    def test_importance_minimum_zero(self):
        r = self._engine().reflect("", "", None, None, ())
        assert r.importance_score >= 0.0

    def test_custom_config(self):
        cfg = ReflectionConfig(base_importance=0.9)
        engine = ReflectionEngine(config=cfg)
        r = engine.reflect("x", "y", self._state(), self._mod(), ())
        assert r.importance_score >= 0.9

    def test_deterministic(self):
        engine = ReflectionEngine()
        state = self._state(intensity=0.7, is_negative=True)
        mod = self._mod(bias=0.4)
        r1 = engine.reflect("input", "output", state, mod, ())
        r2 = engine.reflect("input", "output", state, mod, ())
        assert r1.importance_score == r2.importance_score
        assert r1.goal_signal == r2.goal_signal


class TestReflectionEngineGoal:
    def _engine(self): return ReflectionEngine()

    def test_fatigue_returns_recover(self):
        state = FakeState(primary_emotion="fatigue", is_negative=True)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.goal_signal == "recover"

    def test_drive_high_arousal_returns_execute(self):
        state = FakeState(primary_emotion="drive", is_positive=True, is_high_arousal=True)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.goal_signal == "execute"

    def test_positive_high_bias_returns_push_forward(self):
        state = FakeState(is_positive=True)
        mod = FakeModulation(motivational_bias=0.5)
        r = self._engine().reflect("x", "y", state, mod, ())
        assert r.goal_signal == "push_forward"

    def test_negative_calm_tone_returns_stabilize(self):
        state = FakeState(is_negative=True)
        mod = FakeModulation(tone="calm")
        r = self._engine().reflect("x", "y", state, mod, ())
        assert r.goal_signal == "stabilize"

    def test_doubt_returns_explore(self):
        state = FakeState(primary_emotion="doubt")
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.goal_signal == "explore"

    def test_none_state_returns_none_goal(self):
        r = self._engine().reflect("x", "y", None, FakeModulation(), ())
        assert r.goal_signal is None

    def test_neutral_returns_none_or_valid(self):
        state = FakeState(primary_emotion="neutral", is_positive=False, is_negative=False)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        # Could be None or any valid signal
        assert r.goal_signal in (None, "push_forward", "execute", "stabilize", "recover", "explore")


class TestReflectionEngineTrajectory:
    def _engine(self): return ReflectionEngine()

    def test_fatigue_high_intensity_returns_declining(self):
        state = FakeState(primary_emotion="fatigue", is_negative=True, intensity=0.8)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.trajectory_signal == "declining"

    def test_drive_high_arousal_high_intensity_returns_escalating(self):
        state = FakeState(is_positive=True, is_high_arousal=True, intensity=0.8)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.trajectory_signal == "escalating"

    def test_confidence_returns_progressing(self):
        state = FakeState(primary_emotion="confidence", is_positive=True, intensity=0.5)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.trajectory_signal == "progressing"

    def test_low_intensity_returns_stable(self):
        state = FakeState(intensity=0.2)
        r = self._engine().reflect("x", "y", state, FakeModulation(), ())
        assert r.trajectory_signal == "stable"

    def test_none_state_returns_none_trajectory(self):
        r = self._engine().reflect("x", "y", None, FakeModulation(), ())
        assert r.trajectory_signal is None


class TestReflectionEngineSummary:
    def _engine(self): return ReflectionEngine()

    def test_summary_non_empty(self):
        state = FakeState(primary_emotion="frustration")
        r = self._engine().reflect("I feel stuck.", "Let me help.", state, FakeModulation(), ())
        assert len(r.summary) > 0

    def test_summary_contains_emotion(self):
        state = FakeState(primary_emotion="confidence")
        r = self._engine().reflect("test", "output", state, FakeModulation(), ())
        assert "confidence" in r.summary

    def test_summary_max_200_chars(self):
        state = FakeState()
        r = self._engine().reflect("x" * 300, "y" * 300, state, FakeModulation(), ())
        assert len(r.summary) <= 200

    def test_summary_contains_tool_name(self):
        from tools.base_tool import ToolResult
        state = FakeState()
        tr = ToolResult(tool_name="echo", success=True, output={})
        r = self._engine().reflect("test", "output", state, FakeModulation(), (tr,))
        assert "echo" in r.summary or "TOOL" in r.summary


class TestReflectionEngineError:
    def test_none_state_does_not_crash(self):
        engine = ReflectionEngine()
        result = engine.reflect("input", "output", None, None, ())
        assert isinstance(result, ReflectionResult)

    def test_missing_fields_do_not_crash(self):
        engine = ReflectionEngine()
        class Partial: pass
        result = engine.reflect("input", "output", Partial(), Partial(), ())
        assert isinstance(result, ReflectionResult)

    def test_empty_strings_do_not_crash(self):
        engine = ReflectionEngine()
        result = engine.reflect("", "", None, None, ())
        assert isinstance(result, ReflectionResult)

    def test_importance_always_in_range(self):
        engine = ReflectionEngine()
        result = engine.reflect("", "", None, None, ())
        assert 0.0 <= result.importance_score <= 1.0


class TestReflectionProtocol:
    def test_reflection_engine_satisfies_protocol(self):
        engine = ReflectionEngine()
        assert isinstance(engine, ReflectionEngineProtocol)


class TestAgentCoreReflection:
    def test_reflection_in_response_when_enabled(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert result.reflection is not None

    def test_reflection_is_reflection_result(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert isinstance(result.reflection, ReflectionResult)

    def test_reflection_called_once_per_turn(self):
        reflect_calls = []
        class CountingEngine:
            def reflect(self, **kw):
                reflect_calls.append(kw)
                return ReflectionResult(summary="s", importance_score=0.5,
                                       emotional_tags=[], goal_signal=None, trajectory_signal=None)
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config, reflection_engine=CountingEngine())
        agent.handle_message("Turn 1.")
        agent.handle_message("Turn 2.")
        assert len(reflect_calls) == 2

    def test_reflection_importance_in_range(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert 0.0 <= result.reflection.importance_score <= 1.0

    def test_trajectory_in_response(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert result.trajectory is not None

    def test_trajectory_turn_count_increments(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        agent.handle_message("Turn 1.")
        result = agent.handle_message("Turn 2.")
        # trajectory.turn_count should be 2 after 2 turns with reflection
        from agent.trajectory import TrajectoryState
        assert isinstance(result.trajectory, TrajectoryState)
        assert result.trajectory.turn_count == 2


class TestReflectionDisabled:
    def test_reflection_none_when_disabled(self):
        config = AgentConfig(enable_reflection=False, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        assert result.reflection is None

    def test_trajectory_still_updated_when_reflection_disabled(self):
        config = AgentConfig(enable_reflection=False, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        result = agent.handle_message("Test.")
        # Trajectory is always present
        assert result.trajectory is not None


class TestReflectionMemory:
    def test_adaptive_importance_uses_reflection_score(self):
        mem = FakeMemory()
        # Use custom engine that returns fixed importance
        class FixedEngine:
            def reflect(self, **kw):
                return ReflectionResult(summary="s", importance_score=0.99,
                                       emotional_tags=[], goal_signal=None, trajectory_signal=None)
        config = AgentConfig(
            enable_reflection=True, adaptive_importance=True,
            write_user_turn_to_memory=True, enable_rag=False,
            apply_emotional_protection=False,
        )
        agent, *_ = _make_agent(config=config, reflection_engine=FixedEngine(), memory=mem)
        agent.handle_message("Test.")
        user_writes = [c for c in mem.add_calls if c.get("source") == "interaction"]
        assert len(user_writes) >= 1
        assert any(abs(w["importance_score"] - 0.99) < 0.01 for w in user_writes)

    def test_reflection_summary_used_when_configured(self):
        mem = FakeMemory()
        class SummaryEngine:
            def reflect(self, **kw):
                return ReflectionResult(summary="REFLECTION SUMMARY",
                                       importance_score=0.5, emotional_tags=[],
                                       goal_signal=None, trajectory_signal=None)
        config = AgentConfig(
            enable_reflection=True, use_reflection_summary_for_memory=True,
            write_user_turn_to_memory=True, enable_rag=False,
            apply_emotional_protection=False,
        )
        agent, *_ = _make_agent(config=config, reflection_engine=SummaryEngine(), memory=mem)
        agent.handle_message("Original user input.")
        user_writes = [c for c in mem.add_calls if c.get("source") == "interaction"]
        assert any("REFLECTION SUMMARY" in w["content"] for w in user_writes)

    def test_non_adaptive_uses_config_importance(self):
        mem = FakeMemory()
        config = AgentConfig(
            enable_reflection=True, adaptive_importance=False,
            auto_memory_importance=0.77,
            write_user_turn_to_memory=True, enable_rag=False,
            apply_emotional_protection=False,
        )
        agent, *_ = _make_agent(config=config, memory=mem)
        agent.handle_message("Test.")
        user_writes = [c for c in mem.add_calls if c.get("source") == "interaction"]
        assert any(abs(w["importance_score"] - 0.77) < 0.01 for w in user_writes)


class TestReflectionStreaming:
    def _stream_and_finalize(self, agent, text="Test."):
        tokens = list(agent.stream_message(text))
        return agent.finalize_stream("".join(tokens))

    def test_reflection_not_in_stream_message_phase(self):
        reflect_calls = []
        class CountingEngine:
            def reflect(self, **kw):
                reflect_calls.append(kw)
                return ReflectionResult(summary="s", importance_score=0.5,
                                       emotional_tags=[], goal_signal=None, trajectory_signal=None)
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config, reflection_engine=CountingEngine())
        gen = agent.stream_message("Test.")
        next(gen, None)  # start generating
        assert len(reflect_calls) == 0  # not yet

    def test_reflection_in_finalize_response(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config)
        response = self._stream_and_finalize(agent)
        assert response.reflection is not None

    def test_reflection_called_once_in_streaming(self):
        reflect_calls = []
        class CountingEngine:
            def reflect(self, **kw):
                reflect_calls.append(kw)
                return ReflectionResult(summary="s", importance_score=0.5,
                                       emotional_tags=[], goal_signal=None, trajectory_signal=None)
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent, *_ = _make_agent(config=config, reflection_engine=CountingEngine())
        self._stream_and_finalize(agent)
        assert len(reflect_calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestReflectionResult, TestReflectionEngineScoring,
        TestReflectionEngineGoal, TestReflectionEngineTrajectory,
        TestReflectionEngineSummary, TestReflectionEngineError,
        TestReflectionProtocol, TestAgentCoreReflection,
        TestReflectionDisabled, TestReflectionMemory, TestReflectionStreaming,
    ]
    passed, failed = [], []
    for cls in test_classes:
        instance = cls()
        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            label = f"{cls.__name__}.{name}"
            try:
                getattr(instance, name)()
                passed.append(label); print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e)); print(f"  ❌ {label}: {e}")
                traceback.print_exc()
    total = len(passed) + len(failed)
    print(f"\n{'='*60}\nResults: {len(passed)}/{total} passed")
    if failed:
        for l, e in failed: print(f"  {l}: {e}")
        sys.exit(1)
    else:
        print("All checks passed ✅")