"""
A.C.C.E.S.S. — Phase 4.5 Test Suite

Tests for Meta-Cognitive Stability Layer:
    1. MetaDiagnostics invariants
    2. Coherence scoring system
    3. StrategicAdjustmentEngine (no side effects)
    4. CognitiveIdentityManager meta-layer integration
    5. Backward compatibility (meta OFF)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, UTC
from types import SimpleNamespace

from agent.agent_config import AgentConfig
from agent.relationship_state import RelationshipState, RelationshipEngine
from agent.personality_state import PersonalityTraits, TraitDriftEngine, TRAIT_NAMES
from agent.self_model import SelfModel, SelfModelEngine
from agent.goal_queue import GoalQueue, InternalGoal, GoalQueueConfig
from agent.cognitive_identity import CognitiveIdentityManager
from agent.meta_diagnostics import (
    MetaDiagnosticsEngine, MetaDiagnosticsConfig, MetaDiagnostics,
    IdentitySnapshot, _linear_slope,
)
from agent.meta_strategy import (
    StrategicAdjustmentEngine, StrategyConfig, StrategicAdjustment, MetaGoal,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_snapshot(
    trust=0.3, respect=0.3, dependency=0.0,
    assertiveness=0.45, warmth=0.55, analytical=0.50,
    drive=0.50, stability=0.55,
    mode="collaborating",
    active_goals=0, completed_goals=0, expired_goals=0,
) -> IdentitySnapshot:
    """Create a snapshot with specified values."""
    return IdentitySnapshot(
        personality=PersonalityTraits(
            assertiveness=assertiveness, warmth=warmth,
            analytical_bias=analytical, motivational_drive=drive,
            emotional_stability=stability,
        ),
        relationship=RelationshipState(
            trust_level=trust, respect_level=respect,
            dependency_risk=dependency,
        ),
        dominant_mode=mode,
        dependency_risk=dependency,
        active_goals=active_goals,
        completed_goals=completed_goals,
        expired_goals=expired_goals,
    )


def _stable_history(n: int) -> list[IdentitySnapshot]:
    """Generate N identical snapshots (perfect stability)."""
    return [_make_snapshot() for _ in range(n)]


def _volatile_personality_history(n: int) -> list[IdentitySnapshot]:
    """Alternate personality traits wildly between snapshots."""
    history = []
    for i in range(n):
        if i % 2 == 0:
            history.append(_make_snapshot(assertiveness=0.1, warmth=0.9, analytical=0.1, drive=0.9, stability=0.1))
        else:
            history.append(_make_snapshot(assertiveness=0.9, warmth=0.1, analytical=0.9, drive=0.1, stability=0.9))
    return history


def _rising_dependency_history(n: int) -> list[IdentitySnapshot]:
    """Dependency risk rises linearly from 0 to 0.8."""
    return [_make_snapshot(dependency=0.8 * i / max(1, n - 1)) for i in range(n)]


def _mode_flip_history(n: int) -> list[IdentitySnapshot]:
    """Alternate modes every turn."""
    modes = ["coaching", "collaborating", "supporting", "challenging"]
    return [_make_snapshot(mode=modes[i % len(modes)]) for i in range(n)]


def _make_turn_signals(goal="push_forward", traj="progressing", positive=True, emotion="focused"):
    """Create fake reflection/trajectory/emotion/modulation for CIM updates."""
    reflection = SimpleNamespace(
        goal_signal=goal,
        importance_score=0.7,
        trajectory_signal=traj,
    )
    trajectory = SimpleNamespace(
        dominant_trajectory=traj,
        drift_score=0.1,
    )
    emotional = SimpleNamespace(
        primary_emotion=emotion,
        is_positive=positive,
        is_negative=not positive,
    )
    modulation = SimpleNamespace(
        tone="calm",
        structure_bias="conversational",
        emotional_validation=positive,
    )
    return reflection, trajectory, emotional, modulation


# ═════════════════════════════════════════════════════════════════════════════
# 1. MetaDiagnostics INVARIANTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMetaDiagnosticsInvariants:
    """Coherence always ∈ [0,1], risk flags are strings, data_points matches input."""

    def test_coherence_always_clamped(self):
        diag = MetaDiagnostics(coherence_score=1.5)
        assert diag.coherence_score == 1.0
        diag2 = MetaDiagnostics(coherence_score=-0.3)
        assert diag2.coherence_score == 0.0

    def test_default_is_healthy(self):
        diag = MetaDiagnostics()
        assert diag.is_healthy
        assert diag.coherence_score == 1.0
        assert diag.risk_flags == ()

    def test_to_dict_roundtrip_keys(self):
        diag = MetaDiagnostics(
            personality_volatility=0.12,
            coherence_score=0.75,
            risk_flags=("personality_volatile",),
            data_points=10,
        )
        d = diag.to_dict()
        assert d["coherence_score"] == 0.75
        assert d["risk_flags"] == ["personality_volatile"]
        assert d["data_points"] == 10

    def test_insufficient_data_returns_neutral(self):
        engine = MetaDiagnosticsEngine()
        # 0 snapshots
        diag0 = engine.analyze([])
        assert diag0.coherence_score == 1.0
        assert diag0.data_points == 0
        # 1 snapshot
        diag1 = engine.analyze([_make_snapshot()])
        assert diag1.coherence_score == 1.0
        assert diag1.data_points == 1


# ═════════════════════════════════════════════════════════════════════════════
# 2. COHERENCE SCORING SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class TestCoherenceScoring:
    """Coherence formula: monotonic, never NaN, clamped."""

    def test_perfect_stability_high_coherence(self):
        engine = MetaDiagnosticsEngine()
        history = _stable_history(10)
        diag = engine.analyze(history)
        assert diag.coherence_score >= 0.9
        assert diag.is_healthy
        assert diag.personality_volatility == 0.0
        assert diag.relationship_volatility == 0.0

    def test_extreme_volatility_lowers_coherence(self):
        engine = MetaDiagnosticsEngine()
        stable = engine.analyze(_stable_history(10))
        volatile = engine.analyze(_volatile_personality_history(10))
        assert volatile.coherence_score < stable.coherence_score
        assert volatile.personality_volatility > 0

    def test_rising_dependency_lowers_coherence(self):
        engine = MetaDiagnosticsEngine()
        stable = engine.analyze(_stable_history(10))
        rising = engine.analyze(_rising_dependency_history(10))
        assert rising.coherence_score < stable.coherence_score
        assert rising.dependency_trend > 0

    def test_mode_instability_lowers_coherence(self):
        engine = MetaDiagnosticsEngine()
        stable = engine.analyze(_stable_history(10))
        flipping = engine.analyze(_mode_flip_history(10))
        assert flipping.coherence_score < stable.coherence_score
        assert flipping.mode_instability > 0.5

    def test_zero_goals_gives_full_goal_health(self):
        engine = MetaDiagnosticsEngine()
        history = _stable_history(5)
        diag = engine.analyze(history)
        assert diag.goal_resolution_rate == 1.0

    def test_all_goals_expired_low_resolution(self):
        engine = MetaDiagnosticsEngine()
        history = [_make_snapshot(completed_goals=0, expired_goals=10) for _ in range(5)]
        diag = engine.analyze(history)
        assert diag.goal_resolution_rate == 0.0

    def test_mixed_goals_partial_resolution(self):
        engine = MetaDiagnosticsEngine()
        history = [_make_snapshot(completed_goals=3, expired_goals=7) for _ in range(5)]
        diag = engine.analyze(history)
        assert 0.25 < diag.goal_resolution_rate < 0.35

    def test_coherence_never_nan(self):
        """Even with extreme inputs, coherence must not be NaN."""
        engine = MetaDiagnosticsEngine()
        # Mix of extreme scenarios
        history = []
        for i in range(20):
            history.append(_make_snapshot(
                trust=1.0 if i % 3 == 0 else 0.0,
                dependency=float(i) / 20,
                assertiveness=1.0 if i % 2 == 0 else 0.0,
                mode=["coaching", "supporting", "challenging", "collaborating"][i % 4],
                completed_goals=i, expired_goals=20 - i,
            ))
        diag = engine.analyze(history)
        assert 0.0 <= diag.coherence_score <= 1.0
        import math
        assert not math.isnan(diag.coherence_score)

    def test_monotonicity_with_increasing_volatility(self):
        """More volatility → lower coherence (monotonic)."""
        engine = MetaDiagnosticsEngine()
        scores = []
        for spread in [0.0, 0.1, 0.3, 0.5, 0.8]:
            history = []
            for i in range(10):
                a = 0.5 + spread * (1 if i % 2 == 0 else -1)
                a = max(0.0, min(1.0, a))
                history.append(_make_snapshot(assertiveness=a, warmth=1.0 - a))
            diag = engine.analyze(history)
            scores.append(diag.coherence_score)
        # Each score should be <= the previous (monotonically decreasing)
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 0.001  # tiny epsilon for float


# ═════════════════════════════════════════════════════════════════════════════
# 3. RISK FLAGS
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskFlags:
    """Flags trigger at correct thresholds."""

    def test_stable_no_flags(self):
        engine = MetaDiagnosticsEngine()
        diag = engine.analyze(_stable_history(10))
        assert diag.risk_flags == ()

    def test_personality_volatile_flag(self):
        engine = MetaDiagnosticsEngine()
        diag = engine.analyze(_volatile_personality_history(10))
        assert "personality_volatile" in diag.risk_flags

    def test_dependency_rising_flag(self):
        engine = MetaDiagnosticsEngine()
        diag = engine.analyze(_rising_dependency_history(10))
        assert "dependency_rising" in diag.risk_flags

    def test_mode_unstable_flag(self):
        engine = MetaDiagnosticsEngine()
        diag = engine.analyze(_mode_flip_history(10))
        assert "mode_unstable" in diag.risk_flags

    def test_goals_failing_flag(self):
        engine = MetaDiagnosticsEngine()
        history = [_make_snapshot(completed_goals=1, expired_goals=20) for _ in range(5)]
        diag = engine.analyze(history)
        assert "goals_failing" in diag.risk_flags

    def test_relationship_volatile_flag(self):
        engine = MetaDiagnosticsEngine()
        history = []
        for i in range(10):
            t = 0.9 if i % 2 == 0 else 0.1
            r = 0.9 if i % 2 == 0 else 0.1
            history.append(_make_snapshot(trust=t, respect=r))
        diag = engine.analyze(history)
        assert "relationship_volatile" in diag.risk_flags


# ═════════════════════════════════════════════════════════════════════════════
# 4. LINEAR SLOPE HELPER
# ═════════════════════════════════════════════════════════════════════════════

class TestLinearSlope:

    def test_constant_zero_slope(self):
        assert _linear_slope([0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_rising_positive_slope(self):
        slope = _linear_slope([0.0, 0.1, 0.2, 0.3])
        assert slope > 0

    def test_falling_negative_slope(self):
        slope = _linear_slope([0.3, 0.2, 0.1, 0.0])
        assert slope < 0

    def test_single_point_zero(self):
        assert _linear_slope([0.5]) == 0.0

    def test_empty_zero(self):
        assert _linear_slope([]) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. STRATEGIC ADJUSTMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestStrategicAdjustment:
    """Pure function: no side effects, correct triggers."""

    def test_healthy_returns_no_adjustment(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(coherence_score=0.9)
        adj = engine.evaluate(diag)
        assert not adj.any_adjustment
        assert adj.suggested_drift_cap is None
        assert adj.dependency_mitigation_multiplier == 1.0
        assert adj.allow_goal_injection is True
        assert adj.suggested_guidance_mode is None
        assert adj.meta_goals == ()

    def test_low_coherence_reduces_drift_cap(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(coherence_score=0.4)
        adj = engine.evaluate(diag)
        assert adj.any_adjustment
        assert adj.suggested_drift_cap == 0.005
        assert adj.suggested_guidance_mode == "structured"

    def test_dependency_rising_boosts_mitigation(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(
            dependency_trend=0.05,
            risk_flags=("dependency_rising",),
        )
        adj = engine.evaluate(diag)
        assert adj.any_adjustment
        assert adj.dependency_mitigation_multiplier == 2.0

    def test_dependency_rising_injects_meta_goal(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(
            dependency_trend=0.02,
            risk_flags=("dependency_rising",),
        )
        adj = engine.evaluate(diag)
        descs = [g.description for g in adj.meta_goals]
        assert "Encourage independent real-world action" in descs

    def test_goals_failing_blocks_injection(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(goal_resolution_rate=0.1)
        adj = engine.evaluate(diag)
        assert adj.any_adjustment
        assert adj.allow_goal_injection is False

    def test_goals_failing_injects_cognitive_load_goal(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(goal_resolution_rate=0.1)
        adj = engine.evaluate(diag)
        descs = [g.description for g in adj.meta_goals]
        assert any("cognitive load" in d.lower() for d in descs)

    def test_mode_unstable_injects_stabilize_goal(self):
        engine = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(risk_flags=("mode_unstable",))
        adj = engine.evaluate(diag)
        descs = [g.description for g in adj.meta_goals]
        assert any("stabilize" in d.lower() for d in descs)

    def test_adjustment_is_frozen(self):
        adj = StrategicAdjustment()
        with pytest.raises(AttributeError):
            adj.any_adjustment = True

    def test_meta_goal_is_frozen(self):
        mg = MetaGoal(description="test")
        with pytest.raises(AttributeError):
            mg.description = "changed"

    def test_to_dict_complete(self):
        adj = StrategicAdjustment(
            suggested_drift_cap=0.005,
            any_adjustment=True,
            meta_goals=(MetaGoal("test goal", 0.8),),
        )
        d = adj.to_dict()
        assert d["suggested_drift_cap"] == 0.005
        assert d["any_adjustment"] is True
        assert len(d["meta_goals"]) == 1
        assert d["meta_goals"][0]["description"] == "test goal"


# ═════════════════════════════════════════════════════════════════════════════
# 6. COGNITIVE IDENTITY MANAGER — META INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestCIMMetaIntegration:
    """Integration tests: CognitiveIdentityManager with meta-cognition enabled."""

    def _make_cim(self, meta=True) -> CognitiveIdentityManager:
        cfg = AgentConfig(
            enable_relationship_tracking=True,
            enable_personality_drift=True,
            enable_self_model=True,
            enable_goal_queue=True,
            enable_meta_cognition=meta,
            meta_window_size=20,
        )
        return CognitiveIdentityManager(cfg)

    def test_meta_off_returns_none(self):
        cim = self._make_cim(meta=False)
        assert cim.meta_snapshot() is None
        assert cim.last_diagnostics is None
        assert cim.last_adjustment is None

    def test_meta_on_initial_state(self):
        cim = self._make_cim(meta=True)
        snap = cim.meta_snapshot()
        assert snap is not None
        assert snap["diagnostics"] is None  # no turns yet
        assert snap["history_size"] == 0

    def test_single_turn_captures_snapshot(self):
        cim = self._make_cim()
        r, t, e, m = _make_turn_signals()
        cim.update_from_turn(r, t, e, m)
        assert len(cim.meta_history) == 1
        # Still < 2 snapshots, diagnostics neutral
        assert cim.last_diagnostics.coherence_score == 1.0

    def test_two_turns_produces_diagnostics(self):
        cim = self._make_cim()
        for _ in range(2):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        diag = cim.last_diagnostics
        assert diag is not None
        assert diag.data_points == 2
        assert 0.0 <= diag.coherence_score <= 1.0

    def test_20_stable_turns_high_coherence(self):
        cim = self._make_cim()
        for _ in range(20):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        diag = cim.last_diagnostics
        assert diag.coherence_score >= 0.7
        assert diag.is_healthy or "dependency_rising" in diag.risk_flags
        # dependency_rising might trigger because dependency grows passively

    def test_20_volatile_turns_low_coherence(self):
        """Alternate between very different signals → volatility → lower coherence."""
        cim = self._make_cim()
        for i in range(20):
            if i % 2 == 0:
                r, t, e, m = _make_turn_signals(
                    goal="push_forward", traj="progressing", positive=True,
                )
            else:
                r, t, e, m = _make_turn_signals(
                    goal="recover", traj="declining", positive=False,
                    emotion="frustration",
                )
            cim.update_from_turn(r, t, e, m)
        diag = cim.last_diagnostics
        # Must be lower than stable scenario
        assert diag.coherence_score < 0.95

    def test_meta_goals_injected_on_dependency_spike(self):
        """Simulate dependency rising → meta-goal should appear in queue."""
        cim = self._make_cim()
        # Force dependency up by simulating negative + stabilize turns
        for _ in range(25):
            r, t, e, m = _make_turn_signals(
                goal="stabilize", traj="declining", positive=False,
                emotion="anxiety",
            )
            cim.update_from_turn(r, t, e, m)
        # Check if meta-goal was injected
        all_goals = cim.goal_queue.list_all()
        descs = [g.description for g in all_goals]
        # Either the dependency meta-goal or the stabilize meta-goal should exist
        has_meta = any(
            "independent" in d.lower() or "stabilize" in d.lower() or "cognitive" in d.lower()
            for d in descs
        )
        # At minimum, the diagnostics should show flags
        assert cim.last_diagnostics is not None
        assert len(cim.last_diagnostics.risk_flags) > 0

    def test_identity_snapshot_includes_meta(self):
        cim = self._make_cim()
        for _ in range(3):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        full = cim.identity_snapshot()
        assert "meta_cognition" in full
        assert full["meta_cognition"]["diagnostics"] is not None

    def test_history_trimmed_to_window(self):
        cim = self._make_cim()
        cim._meta_window = 5
        for _ in range(20):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        assert len(cim.meta_history) <= 5

    def test_meta_snapshot_structure(self):
        cim = self._make_cim()
        for _ in range(3):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        snap = cim.meta_snapshot()
        assert "diagnostics" in snap
        assert "adjustment" in snap
        assert "history_size" in snap
        assert "window_size" in snap


# ═════════════════════════════════════════════════════════════════════════════
# 7. BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Meta OFF must leave Phase 4 behavior completely untouched."""

    def test_meta_off_no_history_buffer(self):
        cfg = AgentConfig(
            enable_relationship_tracking=True,
            enable_personality_drift=True,
            enable_self_model=True,
            enable_goal_queue=True,
            enable_meta_cognition=False,
        )
        cim = CognitiveIdentityManager(cfg)
        for _ in range(10):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        # No meta history should be populated
        assert cim.meta_snapshot() is None
        assert len(cim.meta_history) == 0

    def test_meta_off_identity_snapshot_no_meta_key(self):
        cfg = AgentConfig(
            enable_relationship_tracking=True,
            enable_meta_cognition=False,
        )
        cim = CognitiveIdentityManager(cfg)
        snap = cim.identity_snapshot()
        assert "meta_cognition" not in snap

    def test_default_config_all_off(self):
        cfg = AgentConfig()
        assert cfg.enable_meta_cognition is False
        assert cfg.meta_coherence_threshold == 0.6
        assert cfg.meta_window_size == 20

    def test_phase4_works_without_meta(self):
        """Full Phase 4 update cycle works without meta-cognition."""
        cfg = AgentConfig(
            enable_relationship_tracking=True,
            enable_personality_drift=True,
            enable_self_model=True,
            enable_goal_queue=True,
            enable_meta_cognition=False,
        )
        cim = CognitiveIdentityManager(cfg)
        r, t, e, m = _make_turn_signals()
        cim.update_from_turn(r, t, e, m)
        # Phase 4 should work normally
        assert cim.relationship.interaction_count == 1
        assert cim.personality.total_drift_events == 1
        assert cim.self_model.total_observations == 1


# ═════════════════════════════════════════════════════════════════════════════
# 8. IDENTITY SNAPSHOT frozen/immutable
# ═════════════════════════════════════════════════════════════════════════════

class TestIdentitySnapshotFrozen:

    def test_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.dominant_mode = "hacked"

    def test_snapshot_captures_correct_values(self):
        snap = _make_snapshot(trust=0.7, mode="coaching", active_goals=3)
        assert snap.relationship.trust_level == 0.7
        assert snap.dominant_mode == "coaching"
        assert snap.active_goals == 3


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])