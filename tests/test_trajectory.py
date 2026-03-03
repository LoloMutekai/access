"""
A.C.C.E.S.S. — Trajectory Test Suite

TestTrajectoryState         — frozen-ish, to_dict, UTC
TestTrajectoryTrackerBasics — update, state, window
TestTrajectoryTrackerDrift  — drift computation
TestTrajectoryTrackerDominant — dominant signal selection
TestTrajectoryTrackerReset  — reset clears state
TestTrajectoryIntegration   — AgentCore updates trajectory after each turn
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from agent.trajectory import TrajectoryTracker, TrajectoryState
from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient
from agent.reflection_engine import ReflectionResult


# ─── Fakes ───────────────────────────────────────────────────────────────────

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; intensity: float = 0.5
    pad: object = None; is_positive: bool = False; is_negative: bool = False
    is_high_arousal: bool = False; label: str = "neutral"
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
    sections: tuple = ("tone",)
    def to_api_messages(self): return [{"role":"system","content":"S"},{"role":"user","content":"u"}]

class FakeEngine:
    def __init__(self, state=None): self._state = state or FakeState(); self.protection_calls = []
    def process_interaction(self, t, session_id=None): return self._state
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, s): self.protection_calls.append(s)
    def stats(self): return {}

class FakeModulator:
    def build_modulation(self, state, trend, dominant_pattern=None): return FakeModulation()

class FakeBuilder:
    def build(self, user_input, modulation, memory_context=None): return FakeBuiltPrompt()

def _make_agent(llm=None, config=None):
    llm = llm or FakeLLMClient(response="Response.")
    config = config or AgentConfig(enable_rag=False, apply_emotional_protection=False)
    return AgentCore(
        emotion_engine=FakeEngine(), conversation_modulator=FakeModulator(),
        prompt_builder=FakeBuilder(), llm_client=llm, config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryState:
    def test_to_dict_has_required_keys(self):
        state = TrajectoryState()
        d = state.to_dict()
        assert {"dominant_goal_signal", "recent_goal_signals", "drift_score",
                "dominant_trajectory", "turn_count", "updated_at"}.issubset(set(d.keys()))

    def test_updated_at_is_utc(self):
        from datetime import UTC
        state = TrajectoryState()
        assert state.updated_at.tzinfo is not None

    def test_repr_contains_drift(self):
        state = TrajectoryState(drift_score=0.42)
        assert "0.42" in repr(state)

    def test_none_signals_stored_as_none_str_in_dict(self):
        state = TrajectoryState(recent_goal_signals=[None, "push_forward"])
        d = state.to_dict()
        assert "none" in d["recent_goal_signals"]


class TestTrajectoryTrackerBasics:
    def test_initial_state_is_empty(self):
        tracker = TrajectoryTracker(window_size=5)
        state = tracker.state
        assert state.turn_count == 0
        assert state.dominant_goal_signal is None

    def test_update_increments_turn_count(self):
        tracker = TrajectoryTracker()
        tracker.update("push_forward", "progressing")
        assert tracker.state.turn_count == 1

    def test_update_returns_trajectory_state(self):
        tracker = TrajectoryTracker()
        state = tracker.update("push_forward", "progressing")
        assert isinstance(state, TrajectoryState)

    def test_window_trims_to_size(self):
        tracker = TrajectoryTracker(window_size=3)
        for _ in range(5):
            tracker.update("push_forward", "progressing")
        state = tracker.state
        assert len(state.recent_goal_signals) == 3

    def test_window_size_too_small_raises(self):
        try: TrajectoryTracker(window_size=1); assert False
        except ValueError: pass

    def test_state_property_matches_last_update(self):
        tracker = TrajectoryTracker()
        update_state = tracker.update("stabilize", "declining")
        state_prop = tracker.state
        assert state_prop.dominant_goal_signal == update_state.dominant_goal_signal
        assert state_prop.drift_score == update_state.drift_score


class TestTrajectoryTrackerDrift:
    def test_zero_drift_with_uniform_signals(self):
        tracker = TrajectoryTracker(window_size=5)
        for _ in range(5):
            tracker.update("push_forward", "progressing")
        assert tracker.state.drift_score == 0.0

    def test_max_drift_with_all_different_signals(self):
        tracker = TrajectoryTracker(window_size=4)
        signals = ["push_forward", "stabilize", "explore", "recover"]
        for s in signals:
            tracker.update(s, None)
        # 4 distinct signals in 4 slots → max drift
        assert tracker.state.drift_score > 0.5

    def test_partial_drift_with_mixed_signals(self):
        tracker = TrajectoryTracker(window_size=4)
        tracker.update("push_forward", None)
        tracker.update("push_forward", None)
        tracker.update("stabilize", None)
        tracker.update("push_forward", None)
        drift = tracker.state.drift_score
        assert 0.0 < drift < 1.0

    def test_none_signals_not_counted_in_drift(self):
        tracker = TrajectoryTracker(window_size=4)
        tracker.update(None, None)
        tracker.update(None, None)
        tracker.update(None, None)
        tracker.update(None, None)
        # All None → no distinct signals → 0 drift
        assert tracker.state.drift_score == 0.0

    def test_drift_clamped_to_one(self):
        tracker = TrajectoryTracker(window_size=5)
        signals = ["push_forward", "stabilize", "explore", "recover", "execute"]
        for s in signals:
            tracker.update(s, None)
        assert tracker.state.drift_score <= 1.0


class TestTrajectoryTrackerDominant:
    def test_dominant_signal_is_most_frequent(self):
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(3): tracker.update("push_forward", None)
        for _ in range(2): tracker.update("stabilize", None)
        assert tracker.state.dominant_goal_signal == "push_forward"

    def test_dominant_trajectory_ignores_none(self):
        tracker = TrajectoryTracker()
        tracker.update(None, "progressing")
        tracker.update(None, "progressing")
        tracker.update(None, "declining")
        assert tracker.state.dominant_trajectory == "progressing"

    def test_dominant_none_when_all_none(self):
        tracker = TrajectoryTracker()
        tracker.update(None, None)
        tracker.update(None, None)
        assert tracker.state.dominant_goal_signal is None

    def test_tie_breaks_deterministically(self):
        tracker = TrajectoryTracker(window_size=4)
        tracker.update("push_forward", None)
        tracker.update("stabilize", None)
        tracker.update("push_forward", None)
        tracker.update("stabilize", None)
        # Tie — result is one of the two (deterministic for Counter.most_common)
        assert tracker.state.dominant_goal_signal in ("push_forward", "stabilize")


class TestTrajectoryTrackerReset:
    def test_reset_clears_all_state(self):
        tracker = TrajectoryTracker()
        for _ in range(5): tracker.update("push_forward", "progressing")
        tracker.reset()
        state = tracker.state
        assert state.turn_count == 0
        assert state.dominant_goal_signal is None
        assert state.drift_score == 0.0

    def test_update_after_reset_starts_fresh(self):
        tracker = TrajectoryTracker()
        for _ in range(3): tracker.update("stabilize", "declining")
        tracker.reset()
        tracker.update("push_forward", "progressing")
        assert tracker.state.turn_count == 1
        assert tracker.state.dominant_goal_signal == "push_forward"


class TestTrajectoryIntegration:
    def test_trajectory_populated_after_turn(self):
        agent = _make_agent()
        result = agent.handle_message("Test.")
        assert result.trajectory is not None
        assert isinstance(result.trajectory, TrajectoryState)

    def test_trajectory_turn_count_increments_each_turn(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(config=config)
        agent.handle_message("Turn 1.")
        result = agent.handle_message("Turn 2.")
        assert result.trajectory.turn_count == 2

    def test_trajectory_accessible_via_property(self):
        agent = _make_agent()
        agent.handle_message("Test.")
        state = agent.trajectory
        assert isinstance(state, TrajectoryState)

    def test_trajectory_reset_on_session_reset(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(config=config)
        for _ in range(3): agent.handle_message("Turn.")
        agent.reset_session()
        assert agent.trajectory.turn_count == 0

    def test_trajectory_goal_signal_in_to_dict(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(config=config)
        result = agent.handle_message("Test.")
        d = result.trajectory.to_dict()
        assert "dominant_goal_signal" in d
        assert "drift_score" in d

    def test_trajectory_drift_score_in_range(self):
        config = AgentConfig(enable_reflection=True, enable_rag=False,
                             apply_emotional_protection=False)
        agent = _make_agent(config=config)
        for _ in range(5): agent.handle_message("Turn.")
        assert 0.0 <= agent.trajectory.drift_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestTrajectoryState, TestTrajectoryTrackerBasics, TestTrajectoryTrackerDrift,
        TestTrajectoryTrackerDominant, TestTrajectoryTrackerReset, TestTrajectoryIntegration,
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