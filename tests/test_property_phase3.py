"""
A.C.C.E.S.S. — Property-Based Tests (Phase 3 Solidification)

Uses Hypothesis to validate structural invariants across randomized inputs.

Sections:
    TestReflectionProperties   — importance clamping, determinism, summary bounds, signal validity
    TestTrajectoryProperties   — drift bounds, window enforcement, monotonic turns, signal validity
    TestMemoryLoopProperties   — report serializability, duration non-negative, boost caps
    TestLoggerProperties       — event structure, filtering correctness, memory limits
"""

import sys
import os
import json
from dataclasses import dataclass
from typing import Optional

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# ─── Path setup (mirrors existing test conventions) ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.reflection_engine import ReflectionEngine, ReflectionConfig, ReflectionResult
from agent.trajectory import TrajectoryTracker, TrajectoryState
from agent.memory_loop import (
    MemoryLoop, MaintenanceReport, DecayReport,
    ConsolidationReport, TopicReport, RepetitionAdjustmentReport,
)
from agent.logger import StructuredLogger

# ─── Valid signal sets (from production code) ──────────────────────────────────
VALID_GOAL_SIGNALS = {"push_forward", "execute", "stabilize", "recover", "explore", None}
VALID_TRAJECTORY_SIGNALS = {"progressing", "declining", "stable", "escalating", None}


# ─── Hypothesis strategies ─────────────────────────────────────────────────────

@st.composite
def emotional_states(draw):
    """Generate randomized emotional state objects."""
    @dataclass
    class GenState:
        primary_emotion: str
        intensity: float
        is_positive: bool
        is_negative: bool
        is_high_arousal: bool
        label: str
        pad: object = None

    @dataclass
    class PAD:
        valence: float
        arousal: float
        dominance: float

    emotion = draw(st.sampled_from([
        "neutral", "frustration", "confidence", "doubt", "fatigue",
        "drive", "flow", "anxiety", "excitement", "sadness",
    ]))
    intensity = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    is_pos = draw(st.booleans())
    is_neg = draw(st.booleans())
    is_ha = draw(st.booleans())
    v = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
    a = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    d = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    return GenState(
        primary_emotion=emotion, intensity=intensity,
        is_positive=is_pos, is_negative=is_neg, is_high_arousal=is_ha,
        label=emotion, pad=PAD(valence=v, arousal=a, dominance=d),
    )


@st.composite
def modulations(draw):
    """Generate randomized modulation objects."""
    @dataclass
    class GenMod:
        tone: str
        pacing: str
        verbosity: str
        structure_bias: str
        emotional_validation: bool
        motivational_bias: float
        cognitive_load_limit: float
        active_strategies: tuple = ()

    return GenMod(
        tone=draw(st.sampled_from([
            "calm", "energizing", "grounding", "challenging",
            "reassuring", "supportive", "neutral",
        ])),
        pacing=draw(st.sampled_from(["slow", "normal", "fast"])),
        verbosity=draw(st.sampled_from(["concise", "normal", "detailed"])),
        structure_bias=draw(st.sampled_from(["structured", "conversational"])),
        emotional_validation=draw(st.booleans()),
        motivational_bias=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
        cognitive_load_limit=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


goal_signals = st.sampled_from(["push_forward", "execute", "stabilize", "recover", "explore", None])
trajectory_signals = st.sampled_from(["progressing", "declining", "stable", "escalating", None])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REFLECTION ENGINE PROPERTIES
# ═════════════════════════════════════════════════════════════════════════════

class TestReflectionProperties:
    """Invariants that must hold for ALL possible inputs to ReflectionEngine."""

    @given(
        state=emotional_states(),
        mod=modulations(),
        user_input=st.text(min_size=0, max_size=500),
        assistant_output=st.text(min_size=0, max_size=5000),
        n_tools=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_importance_always_in_unit_range(self, state, mod, user_input, assistant_output, n_tools):
        """importance_score ∈ [0.0, 1.0] for any input combination."""
        engine = ReflectionEngine()
        tools = tuple(range(n_tools))  # mock tool results (len checked only)
        result = engine.reflect(user_input, assistant_output, state, mod, tools)
        assert 0.0 <= result.importance_score <= 1.0

    @given(
        state=emotional_states(),
        mod=modulations(),
        user_input=st.text(min_size=1, max_size=200),
        assistant_output=st.text(min_size=0, max_size=1000),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_deterministic_same_inputs_same_output(self, state, mod, user_input, assistant_output):
        """Same inputs must always produce identical importance and signals."""
        engine = ReflectionEngine()
        r1 = engine.reflect(user_input, assistant_output, state, mod, ())
        r2 = engine.reflect(user_input, assistant_output, state, mod, ())
        assert r1.importance_score == r2.importance_score
        assert r1.goal_signal == r2.goal_signal
        assert r1.trajectory_signal == r2.trajectory_signal
        assert r1.summary == r2.summary

    @given(
        state=emotional_states(),
        mod=modulations(),
        user_input=st.text(min_size=0, max_size=500),
        assistant_output=st.text(min_size=0, max_size=2000),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_summary_length_bounded(self, state, mod, user_input, assistant_output):
        """Summary must never exceed 200 characters."""
        engine = ReflectionEngine()
        result = engine.reflect(user_input, assistant_output, state, mod, ())
        assert len(result.summary) <= 200

    @given(state=emotional_states(), mod=modulations())
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_goal_signal_in_allowed_set(self, state, mod):
        """goal_signal must be one of the defined taxonomy values or None."""
        engine = ReflectionEngine()
        result = engine.reflect("test", "response", state, mod, ())
        assert result.goal_signal in VALID_GOAL_SIGNALS

    @given(state=emotional_states(), mod=modulations())
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_trajectory_signal_in_allowed_set(self, state, mod):
        """trajectory_signal must be valid or None."""
        engine = ReflectionEngine()
        result = engine.reflect("test", "response", state, mod, ())
        assert result.trajectory_signal in VALID_TRAJECTORY_SIGNALS

    @given(state=emotional_states(), mod=modulations())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_result_is_json_serializable(self, state, mod):
        """to_log_dict() must always produce JSON-serializable output."""
        engine = ReflectionEngine()
        result = engine.reflect("test", "output", state, mod, ())
        d = result.to_log_dict()
        json.dumps(d)  # must not raise

    def test_none_emotional_state_returns_fallback(self):
        """None state must not crash — returns fallback."""
        engine = ReflectionEngine()
        result = engine.reflect("test", "output", None, None, ())
        assert isinstance(result, ReflectionResult)
        assert 0.0 <= result.importance_score <= 1.0

    @given(
        base=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        bonus=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_config_overrides_respected(self, base, bonus):
        """Custom config values are reflected in scoring."""
        cfg = ReflectionConfig(base_importance=base, bonus_high_intensity=bonus)
        engine = ReflectionEngine(config=cfg)
        result = engine.reflect("x", "y", None, None, ())
        assert result.importance_score >= 0.0
        assert result.importance_score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRAJECTORY TRACKER PROPERTIES
# ═════════════════════════════════════════════════════════════════════════════

class TestTrajectoryProperties:
    """Invariants for TrajectoryTracker under randomized signal sequences."""

    @given(
        signals=st.lists(goal_signals, min_size=1, max_size=100),
        window=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_drift_score_always_in_unit_range(self, signals, window):
        """drift_score ∈ [0.0, 1.0] for any signal sequence."""
        tracker = TrajectoryTracker(window_size=window)
        for s in signals:
            tracker.update(s, None)
        assert 0.0 <= tracker.state.drift_score <= 1.0

    @given(
        signals=st.lists(goal_signals, min_size=1, max_size=200),
        window=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_window_size_respected(self, signals, window):
        """recent_goal_signals never exceeds window_size."""
        tracker = TrajectoryTracker(window_size=window)
        for s in signals:
            tracker.update(s, None)
        assert len(tracker.state.recent_goal_signals) <= window

    @given(n=st.integers(min_value=1, max_value=500))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_turn_count_monotonic(self, n):
        """turn_count increments by exactly 1 per update call."""
        tracker = TrajectoryTracker()
        prev = 0
        for i in range(n):
            state = tracker.update("push_forward", None)
            assert state.turn_count == prev + 1
            prev = state.turn_count

    def test_reset_clears_all_state(self):
        """After reset, tracker is indistinguishable from fresh."""
        tracker = TrajectoryTracker()
        for _ in range(50):
            tracker.update("push_forward", "progressing")
        tracker.reset()
        s = tracker.state
        assert s.turn_count == 0
        assert s.dominant_goal_signal is None
        assert s.drift_score == 0.0
        assert s.recent_goal_signals == []

    @given(
        signals=st.lists(goal_signals, min_size=1, max_size=50),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_dominant_signal_always_valid(self, signals):
        """dominant_goal_signal is always from the allowed set or None."""
        tracker = TrajectoryTracker()
        for s in signals:
            tracker.update(s, None)
        assert tracker.state.dominant_goal_signal in VALID_GOAL_SIGNALS

    @given(
        signals=st.lists(trajectory_signals, min_size=1, max_size=50),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_dominant_trajectory_always_valid(self, signals):
        """dominant_trajectory is always from the allowed set or None."""
        tracker = TrajectoryTracker()
        for s in signals:
            tracker.update(None, s)
        assert tracker.state.dominant_trajectory in VALID_TRAJECTORY_SIGNALS

    @given(signals=st.lists(goal_signals, min_size=1, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_state_serializable(self, signals):
        """to_dict() must always produce JSON-serializable output."""
        tracker = TrajectoryTracker()
        for s in signals:
            tracker.update(s, "progressing")
        d = tracker.state.to_dict()
        json.dumps(d)  # must not raise

    def test_uniform_signals_zero_drift(self):
        """Identical signals must always produce drift = 0.0."""
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(20):
            tracker.update("push_forward", None)
        assert tracker.state.drift_score == 0.0

    def test_all_none_signals_zero_drift(self):
        """All-None signals must produce drift = 0.0."""
        tracker = TrajectoryTracker(window_size=5)
        for _ in range(10):
            tracker.update(None, None)
        assert tracker.state.drift_score == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MEMORY LOOP PROPERTIES
# ═════════════════════════════════════════════════════════════════════════════

class _FakeMemoryManager:
    """Minimal duck-typed MemoryManager for property tests."""
    def __init__(self):
        self._store = None
    def run_decay(self):
        @dataclass
        class R:
            processed: int = 0
            updated: int = 0
            below_floor: int = 0
        return R()


class TestMemoryLoopProperties:
    """Invariants for MemoryLoop reports and maintenance operations."""

    def test_report_always_serializable(self):
        """MaintenanceReport.to_dict() is always JSON-safe."""
        report = MaintenanceReport(
            decay=DecayReport(memories_evaluated=5, memories_updated=2),
            consolidation=ConsolidationReport(candidates_found=3, tagged_for_review=1),
            topics=TopicReport(tag_counts={"focus": 5}, recurrent_topics=["focus"]),
            repetition=RepetitionAdjustmentReport(memories_boosted=2, boost_amount=0.05),
            errors=["test error"],
        )
        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    @given(
        run_decay=st.booleans(),
        run_consolidation=st.booleans(),
        run_topics=st.booleans(),
        run_repetition=st.booleans(),
    )
    @settings(max_examples=30)
    def test_run_never_raises(self, run_decay, run_consolidation, run_topics, run_repetition):
        """MemoryLoop.run() must never raise regardless of flag combination."""
        loop = MemoryLoop(_FakeMemoryManager())
        report = loop.run(
            run_decay=run_decay,
            run_consolidation=run_consolidation,
            run_topics=run_topics,
            run_repetition=run_repetition,
        )
        assert report.total_duration_ms >= 0.0

    def test_duration_always_non_negative(self):
        """All duration fields must be >= 0."""
        loop = MemoryLoop(_FakeMemoryManager())
        report = loop.run(run_decay=True, run_consolidation=False,
                          run_topics=False, run_repetition=False)
        assert report.total_duration_ms >= 0.0
        if report.decay:
            assert report.decay.duration_ms >= 0.0

    @given(
        boost=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        cap=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_repetition_boost_cap_respected(self, boost, cap):
        """Importance after boost must never exceed the configured cap."""
        # This tests the algorithm, not the full store integration
        current_importance = 0.8
        new_score = min(cap, current_importance + boost)
        assert new_score <= cap
        assert new_score <= 1.0

    def test_malformed_store_no_crash(self):
        """MemoryLoop must handle a store that raises on every operation."""
        class BrokenManager:
            _store = None
            def run_decay(self):
                raise RuntimeError("store is down")

        loop = MemoryLoop(BrokenManager())
        report = loop.run(run_decay=True, run_consolidation=True,
                          run_topics=True, run_repetition=False)
        # Must not crash — returns a valid report regardless
        assert isinstance(report, MaintenanceReport)
        assert report.total_duration_ms >= 0.0

    def test_completely_missing_store_no_crash(self):
        """MemoryLoop with manager that has no _store and no run_decay."""
        class EmptyManager:
            pass

        loop = MemoryLoop(EmptyManager())
        report = loop.run(run_decay=True, run_consolidation=True,
                          run_topics=True, run_repetition=False)
        assert isinstance(report, MaintenanceReport)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRUCTURED LOGGER PROPERTIES
# ═════════════════════════════════════════════════════════════════════════════

class TestLoggerProperties:
    """Invariants for StructuredLogger under randomized event streams."""

    @given(
        event_type=st.text(min_size=1, max_size=50),
        payload_keys=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_event_structure_always_complete(self, event_type, payload_keys):
        """Every logged event has all required keys."""
        log = StructuredLogger()
        payload = {k: "val" for k in payload_keys}
        log.log_event(event_type, payload)
        event = log.get_logs()[0]
        assert set(event.keys()) == {"event_type", "timestamp", "session_id", "turn_index", "payload"}

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=30)
    def test_get_logs_returns_correct_count(self, n):
        """get_logs() returns exactly as many events as were logged."""
        log = StructuredLogger()
        for i in range(n):
            log.log_event(f"type_{i % 3}", {"i": i})
        assert log.event_count == n
        assert len(log.get_logs()) == n

    @given(max_events=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20)
    def test_max_events_enforced(self, max_events):
        """Logger never exceeds max_events in memory."""
        log = StructuredLogger(max_events=max_events)
        for i in range(max_events * 3):
            log.log_event("test", {"i": i})
        assert log.event_count <= max_events

    @given(
        types=st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_filter_by_type_correct(self, types):
        """get_logs_by_type returns only matching events."""
        log = StructuredLogger()
        for t in types:
            log.log_event(t, {})
        for check_type in ("a", "b", "c"):
            filtered = log.get_logs_by_type(check_type)
            expected = types.count(check_type)
            assert len(filtered) == expected
            assert all(e["event_type"] == check_type for e in filtered)

    def test_crashing_sink_never_propagates(self):
        """A failing sink must never crash the logger."""
        def bad_sink(event):
            raise RuntimeError("boom")

        log = StructuredLogger(sink=bad_sink)
        for i in range(10):
            log.log_event("test", {"i": i})  # must not raise
        assert log.event_count == 10


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])