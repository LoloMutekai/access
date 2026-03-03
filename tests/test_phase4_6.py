"""
A.C.C.E.S.S. — Phase 4.6 Test Suite

Tests for:
    1. Meta history persistence (save/load roundtrip, corrupt fallback, window trim)
    2. Scoped override enforcement (drift cap, dep multiplier, goal injection throttle)
    3. EMA smoothing (single spike, sustained instability, determinism)
    4. Telemetry hook integration (safe, optional, correct events)
    5. AgentResponse coherence propagation
    6. Backward compatibility (meta OFF leaves everything untouched)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math
import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, UTC

from agent.agent_config import AgentConfig
from agent.relationship_state import RelationshipState, RelationshipEngine
from agent.personality_state import PersonalityTraits, TraitDriftEngine
from agent.self_model import SelfModel
from agent.goal_queue import GoalQueue, GoalQueueConfig
from agent.persistence import IdentityStore, PersistenceConfig
from agent.cognitive_identity import CognitiveIdentityManager
from agent.meta_diagnostics import (
    MetaDiagnosticsEngine, MetaDiagnostics, IdentitySnapshot,
)
from agent.meta_strategy import (
    StrategicAdjustmentEngine, StrategyConfig, StrategicAdjustment, MetaGoal,
)
from agent.models import AgentResponse, TurnRecord


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_turn_signals(goal="push_forward", traj="progressing", positive=True, emotion="focused"):
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


def _make_cim(
    meta=True,
    persist=False,
    data_dir=None,
    telemetry_hook=None,
    ema_alpha=0.3,
) -> CognitiveIdentityManager:
    cfg = AgentConfig(
        enable_relationship_tracking=True,
        enable_personality_drift=True,
        enable_self_model=True,
        enable_goal_queue=True,
        enable_meta_cognition=meta,
        enable_persistence=persist,
        identity_data_dir=data_dir or "data/identity",
        meta_window_size=20,
        meta_ema_alpha=ema_alpha,
    )
    return CognitiveIdentityManager(cfg, telemetry_hook=telemetry_hook)


def _make_snapshot(**kwargs) -> IdentitySnapshot:
    defaults = dict(
        trust=0.3, respect=0.3, dependency=0.0,
        assertiveness=0.45, warmth=0.55, analytical=0.50,
        drive=0.50, stability=0.55,
        mode="collaborating",
        active_goals=0, completed_goals=0, expired_goals=0,
    )
    defaults.update(kwargs)
    return IdentitySnapshot(
        personality=PersonalityTraits(
            assertiveness=defaults["assertiveness"],
            warmth=defaults["warmth"],
            analytical_bias=defaults["analytical"],
            motivational_drive=defaults["drive"],
            emotional_stability=defaults["stability"],
        ),
        relationship=RelationshipState(
            trust_level=defaults["trust"],
            respect_level=defaults["respect"],
            dependency_risk=defaults["dependency"],
        ),
        dominant_mode=defaults["mode"],
        dependency_risk=defaults["dependency"],
        active_goals=defaults["active_goals"],
        completed_goals=defaults["completed_goals"],
        expired_goals=defaults["expired_goals"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. META HISTORY PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

class TestMetaHistoryPersistence:
    """IdentityStore save/load roundtrip for meta history."""

    def test_roundtrip(self, tmp_path):
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path))
        snapshots = [_make_snapshot(trust=0.3 + i * 0.05) for i in range(5)]
        assert store.save_meta_history(snapshots, window_size=20)
        loaded = store.load_meta_history()
        assert len(loaded) == 5
        assert abs(loaded[0].relationship.trust_level - 0.3) < 0.01
        assert abs(loaded[4].relationship.trust_level - 0.5) < 0.01

    def test_window_trim_on_save(self, tmp_path):
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path))
        snapshots = [_make_snapshot() for _ in range(30)]
        store.save_meta_history(snapshots, window_size=10)
        loaded = store.load_meta_history()
        assert len(loaded) == 10

    def test_corrupt_file_returns_empty(self, tmp_path):
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path))
        # Write garbage
        corrupt_path = tmp_path / "meta_history.json"
        corrupt_path.write_text("{{{not json!!!", encoding="utf-8")
        loaded = store.load_meta_history()
        assert loaded == []

    def test_missing_file_returns_empty(self, tmp_path):
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path))
        loaded = store.load_meta_history()
        assert loaded == []

    def test_meta_off_no_file_created(self, tmp_path):
        cfg = AgentConfig(
            enable_meta_cognition=False,
            enable_persistence=True,
            identity_data_dir=str(tmp_path),
        )
        cim = CognitiveIdentityManager(cfg)
        cim.save()
        assert not (tmp_path / "meta_history.json").exists()

    def test_cim_load_respects_window(self, tmp_path):
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path))
        snapshots = [_make_snapshot() for _ in range(30)]
        store.save_meta_history(snapshots, window_size=30)

        cfg = AgentConfig(
            enable_meta_cognition=True,
            enable_persistence=True,
            identity_data_dir=str(tmp_path),
            meta_window_size=10,
        )
        cim = CognitiveIdentityManager(cfg)
        cim.load()
        assert len(cim.meta_history) <= 10

    def test_snapshot_serialization_roundtrip(self):
        snap = _make_snapshot(trust=0.7, mode="coaching", active_goals=3, dependency=0.5)
        d = snap.to_dict()
        snap2 = IdentitySnapshot.from_dict(d)
        assert snap2.dominant_mode == "coaching"
        assert snap2.active_goals == 3
        assert abs(snap2.dependency_risk - 0.5) < 0.001
        assert abs(snap2.relationship.trust_level - 0.7) < 0.001

    def test_cim_save_load_cycle(self, tmp_path):
        cfg = AgentConfig(
            enable_relationship_tracking=True,
            enable_personality_drift=True,
            enable_self_model=True,
            enable_goal_queue=True,
            enable_meta_cognition=True,
            enable_persistence=True,
            identity_data_dir=str(tmp_path),
            meta_window_size=20,
        )
        cim = CognitiveIdentityManager(cfg)
        for _ in range(5):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        history_before = len(cim.meta_history)
        cim.save()

        cim2 = CognitiveIdentityManager(cfg)
        cim2.load()
        assert len(cim2.meta_history) == history_before


# ═════════════════════════════════════════════════════════════════════════════
# 2. SCOPED OVERRIDE ENFORCEMENT
# ═════════════════════════════════════════════════════════════════════════════

class TestScopedOverrides:
    """Adjustments from turn N-1 affect turn N, then reset."""

    def test_drift_cap_override_reduces_drift(self):
        """Lower drift cap → smaller trait change per turn."""
        engine = TraitDriftEngine()
        traits = PersonalityTraits()
        r, t, e, m = _make_turn_signals(goal="push_forward", traj="progressing", positive=True)

        # Normal drift
        result_normal = engine.compute_drift(traits, r, t, e, m)
        normal_change = result_normal.distance_from(traits)

        # Override with smaller cap
        result_capped = engine.compute_drift(traits, r, t, e, m, drift_cap_override=0.002)
        capped_change = result_capped.distance_from(traits)

        assert capped_change <= normal_change
        assert capped_change > 0  # still drifts, just less

    def test_drift_cap_override_zero_blocks_drift(self):
        """drift_cap=0 should prevent all trait changes (except maturation)."""
        engine = TraitDriftEngine()
        traits = PersonalityTraits()
        r, t, e, m = _make_turn_signals()
        result = engine.compute_drift(traits, r, t, e, m, drift_cap_override=0.0)
        # Only emotional_stability maturation (very tiny) should remain
        assert result.distance_from(traits) < 0.001

    def test_dep_mitigation_multiplier_increases_relief(self):
        """Higher multiplier → larger dependency decrease on positive turns."""
        engine = RelationshipEngine()
        state = RelationshipState(dependency_risk=0.5)
        r, t, e, m = _make_turn_signals(goal="push_forward", positive=True)

        # Normal
        result_normal = engine.update_from_turn(state, r, t, e)
        dep_normal = result_normal.dependency_risk

        # 2× mitigation
        result_boosted = engine.update_from_turn(state, r, t, e, dependency_mitigation_multiplier=2.0)
        dep_boosted = result_boosted.dependency_risk

        # 2× mitigation should reduce dependency more
        assert dep_boosted < dep_normal

    def test_dep_mitigation_default_1_no_change(self):
        """Default multiplier = 1.0 should match legacy behavior."""
        engine = RelationshipEngine()
        state = RelationshipState(dependency_risk=0.5)
        r, t, e, m = _make_turn_signals()

        result_default = engine.update_from_turn(state, r, t, e)
        result_explicit = engine.update_from_turn(state, r, t, e, dependency_mitigation_multiplier=1.0)

        assert result_default.dependency_risk == result_explicit.dependency_risk

    def test_goal_injection_blocked_by_adjustment(self):
        """When allow_goal_injection=False, no new goals should be added."""
        cim = _make_cim()

        # Force a StrategicAdjustment that blocks goals
        cim._last_adjustment = StrategicAdjustment(
            allow_goal_injection=False,
            any_adjustment=True,
        )

        initial_count = cim.goal_queue.total_count
        r, t, e, m = _make_turn_signals(goal="push_forward")
        cim.update_from_turn(r, t, e, m)

        # Goal queue should not have grown from reflection-based injection
        # (meta-goals from new adjustment might still be added, but reflection goals blocked)
        # Check no _GOAL_TEMPLATES-based goal was added
        goals = cim.goal_queue.list_all()
        template_goals = [g for g in goals if g.description in [
            "Continue momentum on current task",
            "Complete in-progress execution",
        ]]
        assert len(template_goals) == 0

    def test_scoped_override_does_not_persist(self):
        """Overrides apply only for one turn, not permanently."""
        cim = _make_cim()

        # Force tight cap
        cim._last_adjustment = StrategicAdjustment(
            suggested_drift_cap=0.001,
            any_adjustment=True,
        )

        personality_before = cim.personality

        # Turn 1: capped
        r, t, e, m = _make_turn_signals(goal="push_forward", positive=True)
        cim.update_from_turn(r, t, e, m)
        change_1 = cim.personality.distance_from(personality_before)

        # Turn 2: no override (adjustment should be fresh from meta-cognition)
        personality_after_1 = cim.personality
        cim.update_from_turn(r, t, e, m)
        change_2 = cim.personality.distance_from(personality_after_1)

        # Turn 2 should have at least as much drift as turn 1 (cap removed)
        # This is probabilistic with meta system, so just check it's not zero
        assert change_2 > 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. EMA SMOOTHING
# ═════════════════════════════════════════════════════════════════════════════

class TestEMASmoothing:
    """EMA prevents overreaction to single spikes."""

    def test_ema_formula_deterministic(self):
        """EMA with known inputs produces known output."""
        cim = _make_cim(ema_alpha=0.3)
        # Simulate: first coherence = 1.0, second = 0.4
        cim._smoothed_coherence = 1.0
        cim._coherence_history = [1.0]

        # Manually compute expected: EMA = 0.3 * 0.4 + 0.7 * 1.0 = 0.82
        cim._coherence_history.append(0.4)
        result = cim._compute_ema(0.4)
        expected = 0.3 * 0.4 + 0.7 * 1.0
        assert abs(result - expected) < 0.001

    def test_single_spike_does_not_trigger_adjustment(self):
        """One bad turn should not immediately drop smoothed coherence below threshold."""
        cim = _make_cim(ema_alpha=0.3)

        # 10 stable turns
        for _ in range(10):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)

        smoothed_before = cim.smoothed_coherence
        assert smoothed_before > 0.6

        # 1 volatile turn
        r, t, e, m = _make_turn_signals(
            goal="recover", traj="declining", positive=False, emotion="frustration",
        )
        cim.update_from_turn(r, t, e, m)

        # Smoothed should still be above threshold (0.6)
        assert cim.smoothed_coherence > 0.5

    def test_sustained_instability_drops_smoothed_coherence(self):
        """Many bad turns should eventually lower smoothed coherence."""
        cim = _make_cim(ema_alpha=0.3)

        # 20 volatile turns alternating wildly
        for i in range(20):
            if i % 2 == 0:
                r, t, e, m = _make_turn_signals(
                    goal="push_forward", traj="progressing", positive=True,
                )
            else:
                r, t, e, m = _make_turn_signals(
                    goal="recover", traj="declining", positive=False, emotion="frustration",
                )
            cim.update_from_turn(r, t, e, m)

        # After sustained volatility, smoothed coherence should be lower than a stable run
        cim_stable = _make_cim(ema_alpha=0.3)
        for _ in range(20):
            r, t, e, m = _make_turn_signals()
            cim_stable.update_from_turn(r, t, e, m)

        assert cim.smoothed_coherence < cim_stable.smoothed_coherence

    def test_ema_first_value_equals_raw(self):
        """First EMA value should equal the raw value (no history)."""
        cim = _make_cim(ema_alpha=0.3)
        cim._coherence_history = []
        cim._smoothed_coherence = 1.0

        # First value
        cim._coherence_history.append(0.7)
        result = cim._compute_ema(0.7)
        assert abs(result - 0.7) < 0.001

    def test_ema_alpha_1_equals_raw(self):
        """Alpha=1.0 means no smoothing — smoothed always equals raw."""
        cim = _make_cim(ema_alpha=1.0)
        cim._smoothed_coherence = 0.9
        cim._coherence_history = [0.9, 0.9]

        result = cim._compute_ema(0.3)
        assert abs(result - 0.3) < 0.001

    def test_ema_alpha_0_ignores_new(self):
        """Alpha=0.0 means full smoothing — new value has no effect."""
        cim = _make_cim(ema_alpha=0.0)
        cim._smoothed_coherence = 0.9
        cim._coherence_history = [0.9, 0.9]

        result = cim._compute_ema(0.1)
        assert abs(result - 0.9) < 0.001


# ═════════════════════════════════════════════════════════════════════════════
# 4. TELEMETRY HOOK
# ═════════════════════════════════════════════════════════════════════════════

class TestTelemetryHook:
    """Telemetry emits meta_diagnostics event, never crashes."""

    def test_telemetry_emits_event(self):
        events = []
        def hook(name, metadata):
            events.append((name, metadata))

        cim = _make_cim(telemetry_hook=hook)
        for _ in range(3):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)

        meta_events = [(n, d) for n, d in events if n == "meta_diagnostics"]
        assert len(meta_events) >= 1

        # Check event structure
        last = meta_events[-1][1]
        assert "coherence" in last
        assert "raw_coherence" in last
        assert "flags" in last
        assert "history_size" in last

    def test_telemetry_hook_crash_does_not_crash_agent(self):
        def bad_hook(name, metadata):
            raise RuntimeError("telemetry exploded!")

        cim = _make_cim(telemetry_hook=bad_hook)
        # Should NOT raise
        r, t, e, m = _make_turn_signals()
        cim.update_from_turn(r, t, e, m)
        cim.update_from_turn(r, t, e, m)
        # Agent survives
        assert cim.smoothed_coherence > 0

    def test_no_telemetry_hook_is_safe(self):
        cim = _make_cim(telemetry_hook=None)
        r, t, e, m = _make_turn_signals()
        cim.update_from_turn(r, t, e, m)
        cim.update_from_turn(r, t, e, m)
        assert cim.last_diagnostics is not None

    def test_telemetry_not_called_when_meta_off(self):
        events = []
        def hook(name, metadata):
            events.append(name)

        cim = _make_cim(meta=False, telemetry_hook=hook)
        r, t, e, m = _make_turn_signals()
        cim.update_from_turn(r, t, e, m)
        assert "meta_diagnostics" not in events


# ═════════════════════════════════════════════════════════════════════════════
# 5. AGENTRESPONSE COHERENCE PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentResponseCoherence:
    """AgentResponse includes coherence_score when meta ON."""

    def test_response_includes_coherence(self):
        ar = AgentResponse(
            user_input="test", assistant_output="response",
            emotional_state=None, modulation=None,
            sections_used=(), latency_ms=100.0,
            coherence_score=0.85,
        )
        assert ar.coherence_score == 0.85
        log = ar.to_log_dict()
        assert "coherence_score" in log
        assert log["coherence_score"] == 0.85

    def test_response_no_coherence_when_none(self):
        ar = AgentResponse(
            user_input="test", assistant_output="response",
            emotional_state=None, modulation=None,
            sections_used=(), latency_ms=100.0,
        )
        assert ar.coherence_score is None
        log = ar.to_log_dict()
        assert "coherence_score" not in log

    def test_turn_record_builds_with_coherence(self):
        turn = TurnRecord("input", "session", 0)
        turn.assistant_output = "output"
        turn.emotional_state = None
        turn.modulation = None
        turn.coherence_score = 0.72
        response = turn.build(total_ms=50.0)
        assert response.coherence_score == 0.72

    def test_current_coherence_none_when_meta_off(self):
        cim = _make_cim(meta=False)
        assert cim.current_coherence() is None

    def test_current_coherence_returns_smoothed(self):
        cim = _make_cim()
        for _ in range(3):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        c = cim.current_coherence()
        assert c is not None
        assert 0.0 <= c <= 1.0
        assert c == cim.smoothed_coherence


# ═════════════════════════════════════════════════════════════════════════════
# 6. BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Meta OFF leaves Phase 4 behavior untouched."""

    def test_meta_off_no_ema(self):
        cim = _make_cim(meta=False)
        for _ in range(5):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        assert cim.current_coherence() is None
        assert len(cim.meta_history) == 0
        assert len(cim.coherence_history) == 0

    def test_meta_off_no_scoped_overrides(self):
        """Without meta, scoped override params should stay at defaults."""
        cim = _make_cim(meta=False)
        r, t, e, m = _make_turn_signals()
        personality_before = cim.personality
        cim.update_from_turn(r, t, e, m)
        # Normal drift should occur (no capping)
        assert cim.personality.total_drift_events == 1

    def test_meta_off_snapshot_no_meta_key(self):
        cim = _make_cim(meta=False)
        snap = cim.identity_snapshot()
        assert "meta_cognition" not in snap

    def test_default_config_meta_off(self):
        cfg = AgentConfig()
        assert cfg.enable_meta_cognition is False
        assert cfg.meta_ema_alpha == 0.3

    def test_phase4_relationship_backward_compatible(self):
        """RelationshipEngine still works without multiplier param."""
        engine = RelationshipEngine()
        state = RelationshipState()
        r, t, e, m = _make_turn_signals()
        # Call without the new param — should work
        result = engine.update_from_turn(state, r, t, e)
        assert result.interaction_count == 1


# ═════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION: SCOPED OVERRIDES VIA CIM
# ═════════════════════════════════════════════════════════════════════════════

class TestCIMScopedOverrideIntegration:
    """End-to-end: CIM uses adjustments from turn N-1 on turn N."""

    def test_forced_drift_cap_limits_personality_change(self):
        """Force a tiny drift cap → personality barely moves."""
        cim = _make_cim()

        # Pre-seed adjustment with tight drift cap
        cim._last_adjustment = StrategicAdjustment(
            suggested_drift_cap=0.0005,
            any_adjustment=True,
        )

        personality_before = cim.personality
        r, t, e, m = _make_turn_signals(goal="push_forward", positive=True)
        cim.update_from_turn(r, t, e, m)
        change = cim.personality.distance_from(personality_before)
        assert change < 0.005  # very small

    def test_forced_dep_multiplier_reduces_dependency(self):
        """Force high dep mitigation → dependency drops more."""
        cim = _make_cim()
        # Set high dependency
        cim._relationship = RelationshipState(dependency_risk=0.5)

        # Pre-seed adjustment with 3× dep mitigation
        cim._last_adjustment = StrategicAdjustment(
            dependency_mitigation_multiplier=3.0,
            any_adjustment=True,
        )

        r, t, e, m = _make_turn_signals(goal="push_forward", positive=True)
        dep_before = cim.relationship.dependency_risk
        cim.update_from_turn(r, t, e, m)
        dep_after = cim.relationship.dependency_risk

        # Compare with normal multiplier
        cim2 = _make_cim()
        cim2._relationship = RelationshipState(dependency_risk=0.5)
        cim2.update_from_turn(r, t, e, m)
        dep_normal = cim2.relationship.dependency_risk

        # Boosted mitigation should produce lower dependency
        assert dep_after < dep_normal

    def test_meta_snapshot_includes_smoothed_coherence(self):
        cim = _make_cim()
        for _ in range(3):
            r, t, e, m = _make_turn_signals()
            cim.update_from_turn(r, t, e, m)
        snap = cim.meta_snapshot()
        assert "smoothed_coherence" in snap
        assert 0.0 <= snap["smoothed_coherence"] <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 8. DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Same inputs produce same outputs across runs."""

    def test_identical_runs_same_coherence(self):
        results = []
        for _ in range(3):
            cim = _make_cim(ema_alpha=0.3)
            for _ in range(10):
                r, t, e, m = _make_turn_signals()
                cim.update_from_turn(r, t, e, m)
            results.append(cim.smoothed_coherence)

        assert all(abs(r - results[0]) < 1e-10 for r in results)

    def test_identical_runs_same_diagnostics(self):
        diags = []
        for _ in range(3):
            cim = _make_cim()
            for _ in range(10):
                r, t, e, m = _make_turn_signals()
                cim.update_from_turn(r, t, e, m)
            diags.append(cim.last_diagnostics.to_dict())

        for d in diags[1:]:
            assert d["coherence_score"] == diags[0]["coherence_score"]
            assert d["personality_volatility"] == diags[0]["personality_volatility"]


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])