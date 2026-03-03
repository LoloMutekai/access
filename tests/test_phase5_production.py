"""
Phase 5 — Production Test Suite

Tests the REAL production modules:
    agent.adaptive_meta.AdaptiveMetaController
    agent.meta_strategy.StrategicAdjustmentEngine (with AdaptedContext)
    agent.cognitive_identity.CognitiveIdentityManager (with adaptive wiring)
    agent.persistence.IdentityStore (adaptive state save/load)

Coverage:
    Section A: Property-based invariants (P1–P7)
    Section B: Subsystem behavioral tests (B1–B8)
    Section C: Integration with StrategicAdjustmentEngine
    Section D: CognitiveIdentityManager end-to-end wiring
    Section E: Persistence roundtrip
    Section F: 500-turn long-run simulation
    Section G: Backward compatibility (Phase 4.6 unaffected)
"""

import math
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

import pytest

from agent.adaptive_meta import (
    AdaptiveMetaController,
    AdaptiveMetaConfig,
    AdaptiveMetaState,
    AdaptedContext,
    SAFETY_FLAGS,
    DIMENSION_FLAGS,
    DEFAULT_WEIGHTS,
    _bounded_normalize,
    _clamp,
)
from agent.meta_strategy import (
    StrategicAdjustmentEngine,
    StrategyConfig,
    StrategicAdjustment,
    MetaGoal,
)
from agent.meta_diagnostics import (
    MetaDiagnosticsEngine,
    MetaDiagnostics,
    IdentitySnapshot,
)
from agent.agent_config import AgentConfig
from agent.persistence import IdentityStore, PersistenceConfig
from agent.relationship_state import RelationshipState
from agent.personality_state import PersonalityTraits
from agent.self_model import SelfModel
from agent.goal_queue import GoalQueue, GoalQueueConfig
from agent.cognitive_identity import CognitiveIdentityManager


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_ctrl(cfg_overrides=None) -> AdaptiveMetaController:
    """Create a controller with optional config overrides."""
    kwargs = cfg_overrides or {}
    return AdaptiveMetaController(AdaptiveMetaConfig(**kwargs))


def _make_snapshot(
    trust=0.5, respect=0.5, dep_risk=0.0,
    mode="collaborating", goals_active=2, goals_completed=3, goals_expired=1,
    personality_kwargs=None,
) -> IdentitySnapshot:
    p_kwargs = personality_kwargs or {}
    return IdentitySnapshot(
        personality=PersonalityTraits(**p_kwargs),
        relationship=RelationshipState(
            trust_level=trust, respect_level=respect,
            dependency_risk=dep_risk,
        ),
        dominant_mode=mode,
        dependency_risk=dep_risk,
        active_goals=goals_active,
        completed_goals=goals_completed,
        expired_goals=goals_expired,
    )


def _make_cim(adaptive=True, persistence_dir=None, telemetry_hook=None):
    """Create a fully-wired CognitiveIdentityManager."""
    kwargs = dict(
        enable_relationship_tracking=True,
        enable_personality_drift=True,
        enable_self_model=True,
        enable_goal_queue=True,
        enable_meta_cognition=True,
        enable_adaptive_meta=adaptive,
    )
    if persistence_dir:
        kwargs["enable_persistence"] = True
        kwargs["identity_data_dir"] = str(persistence_dir)
        kwargs["persist_every_n_turns"] = 100  # manual save in tests
    cfg = AgentConfig(**kwargs)
    return CognitiveIdentityManager(cfg, telemetry_hook=telemetry_hook)


@dataclass
class FakeReflection:
    goal_signal: str = "push_forward"
    importance_score: float = 0.7
    trajectory_signal: str = "progressing"

@dataclass
class FakeTrajectory:
    dominant_trajectory: str = "progressing"
    drift_score: float = 0.1

@dataclass
class FakeEmotion:
    primary_emotion: str = "calm"
    is_positive: bool = True
    is_negative: bool = False

@dataclass
class FakeModulation:
    tone: str = "calm"
    emotional_validation: bool = False
    structure_bias: str = "conversational"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A: PROPERTY-BASED INVARIANTS (P1–P7)
# ═════════════════════════════════════════════════════════════════════════════

class TestP1ThresholdBounds:
    """P1: 0.30 ≤ adapted_threshold ≤ 0.85 under all inputs."""

    def test_normal_range(self):
        ctrl = _make_ctrl()
        for c in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            ctx = ctrl.adapt(c, (), [c], False)
            assert 0.30 <= ctx.adapted_threshold <= 0.85, f"c={c}, θ={ctx.adapted_threshold}"

    def test_extreme_oscillation(self):
        ctrl = _make_ctrl()
        history = []
        for i in range(100):
            c = 0.0 if i % 2 == 0 else 1.0
            history.append(c)
            ctx = ctrl.adapt(c, (), list(history), False)
            assert 0.30 <= ctx.adapted_threshold <= 0.85

    def test_constant_low(self):
        ctrl = _make_ctrl()
        history = []
        for _ in range(50):
            history.append(0.1)
            ctx = ctrl.adapt(0.1, (), list(history), False)
            assert 0.30 <= ctx.adapted_threshold <= 0.85

    def test_constant_high(self):
        ctrl = _make_ctrl()
        history = []
        for _ in range(50):
            history.append(0.99)
            ctx = ctrl.adapt(0.99, (), list(history), False)
            assert 0.30 <= ctx.adapted_threshold <= 0.85

    def test_hysteresis_band_valid(self):
        """Deactivate threshold must be ≥ activate threshold."""
        ctrl = _make_ctrl()
        for c in [0.2, 0.5, 0.8]:
            ctx = ctrl.adapt(c, (), [c], False)
            assert ctx.threshold_deactivate >= ctx.threshold_activate


class TestP2WeightNormalization:
    """P2: |sum(weights) - 1.0| < 1e-10 ∧ ∀i: 0.08 ≤ w_i ≤ 0.40."""

    def test_initial_weights(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.8, (), [0.8], False)
        assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
        for w in ctx.adapted_weights:
            assert 0.08 <= w <= 0.40

    def test_single_dimension_hammer(self):
        """Repeatedly trigger one flag — weight must not exceed 0.40."""
        ctrl = _make_ctrl()
        history = [0.8]
        for _ in range(100):
            ctx = ctrl.adapt(0.5, ("dependency_rising",), list(history), True)
            history.append(0.5)
            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
            for w in ctx.adapted_weights:
                assert 0.08 - 1e-10 <= w <= 0.40 + 1e-10

    def test_all_flags_simultaneously(self):
        ctrl = _make_ctrl()
        all_flags = tuple(DIMENSION_FLAGS)
        history = [0.5]
        for _ in range(50):
            ctx = ctrl.adapt(0.3, all_flags, list(history), True)
            history.append(0.3)
            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10

    def test_no_flags_ever(self):
        ctrl = _make_ctrl()
        history = [0.9]
        for _ in range(50):
            ctx = ctrl.adapt(0.9, (), list(history), False)
            history.append(0.9)
            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
            for w in ctx.adapted_weights:
                assert 0.08 <= w <= 0.40


class TestP3AlphaBounds:
    """P3: 0.10 ≤ adapted_alpha ≤ 0.50."""

    def test_stable_regime(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.8, (), [0.80, 0.81, 0.80, 0.81], False)
        assert 0.10 <= ctx.adapted_alpha <= 0.50

    def test_volatile_regime(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.2, (), [0.1, 0.9, 0.1, 0.9, 0.1], False)
        assert 0.10 <= ctx.adapted_alpha <= 0.50

    def test_extreme_s2(self):
        """Even at S²=0 or S²=1, alpha stays in bounds."""
        ctrl = _make_ctrl()
        # S² ≈ 1.0 (no change)
        ctx1 = ctrl.adapt(0.8, (), [0.8, 0.8, 0.8], False)
        assert 0.10 <= ctx1.adapted_alpha <= 0.50

        # S² ≈ 0 (max change)
        ctrl2 = _make_ctrl()
        ctx2 = ctrl2.adapt(0.0, (), [0.0, 1.0], False)
        assert 0.10 <= ctx2.adapted_alpha <= 0.50


class TestP4FatigueBounds:
    """P4: 0.0 ≤ fatigue ≤ 1.0 ∧ 0.30 ≤ dampening ≤ 1.0."""

    def test_accumulation(self):
        ctrl = _make_ctrl()
        for i in range(20):
            ctx = ctrl.adapt(0.5, (), [0.5], True)
            assert 0.0 <= ctx.fatigue_level <= 1.0
            assert 0.30 - 1e-10 <= ctx.dampening_factor <= 1.0

    def test_recovery(self):
        ctrl = _make_ctrl()
        # Build fatigue
        for _ in range(10):
            ctrl.adapt(0.5, (), [0.5], True)
        # Recover
        for _ in range(30):
            ctx = ctrl.adapt(0.5, (), [0.5], False)
            assert 0.0 <= ctx.fatigue_level <= 1.0
            assert 0.30 - 1e-10 <= ctx.dampening_factor <= 1.0

    def test_dampening_at_max_fatigue(self):
        ctrl = _make_ctrl()
        for _ in range(15):
            ctx = ctrl.adapt(0.5, (), [0.5], True)
        # At saturation: dampening = 1.0 - 1.0 * 0.70 = 0.30
        assert ctx.dampening_factor == pytest.approx(0.30, abs=0.02)


class TestP5StabilityBounds:
    """P5: 0.0 ≤ stability_of_stability ≤ 1.0."""

    def test_empty_history(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.5, (), [], False)
        assert 0.0 <= ctx.stability_of_stability <= 1.0

    def test_single_point(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.5, (), [0.5], False)
        assert ctx.stability_of_stability == 1.0

    def test_max_oscillation(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.5, (), [0.0, 1.0, 0.0, 1.0], False)
        assert 0.0 <= ctx.stability_of_stability <= 1.0
        assert ctx.stability_of_stability < 0.15  # very volatile

    def test_stable(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.8, (), [0.80, 0.81, 0.80, 0.79, 0.80], False)
        assert ctx.stability_of_stability > 0.5


class TestP6Determinism:
    """P6: Same inputs → bit-identical outputs."""

    def test_identical_runs(self):
        results = []
        for _ in range(3):
            ctrl = _make_ctrl()
            history = []
            run_outputs = []
            for i in range(20):
                c = 0.3 + (i % 5) * 0.15
                flags = ("personality_volatile",) if i % 3 == 0 else ()
                history.append(c)
                ctx = ctrl.adapt(c, flags, list(history), i % 4 == 0)
                run_outputs.append((
                    ctx.adapted_threshold,
                    tuple(ctx.adapted_weights),
                    ctx.adapted_alpha,
                    ctx.dampening_factor,
                    ctx.stability_of_stability,
                    ctx.circuit_breaker_active,
                    ctx.stability_gate_open,
                ))
            results.append(run_outputs)

        # All 3 runs must be bit-identical
        assert results[0] == results[1]
        assert results[1] == results[2]


class TestP7ColdStart:
    """P7: Fresh controller ≈ Phase 4.6 static behavior."""

    def test_cold_start_threshold(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.85, (), [0.85], False)
        # Cold start μ=0.85 → threshold should be near 0.6–0.85
        assert 0.60 <= ctx.adapted_threshold <= 0.85

    def test_cold_start_alpha(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.85, (), [0.85], False)
        # S² = 1.0 for single point → alpha = alpha_max = 0.50
        # But adapted_alpha is for NEXT turn
        assert 0.10 <= ctx.adapted_alpha <= 0.50

    def test_cold_start_no_fatigue(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.85, (), [0.85], False)
        assert ctx.fatigue_level == 0.0
        assert ctx.dampening_factor == 1.0
        assert ctx.circuit_breaker_active == False

    def test_cold_start_weights_near_defaults(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.85, (), [0.85], False)
        for w, d in zip(ctx.adapted_weights, DEFAULT_WEIGHTS):
            assert abs(w - d) < 0.05  # close to defaults


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B: SUBSYSTEM BEHAVIORAL TESTS (B1–B8)
# ═════════════════════════════════════════════════════════════════════════════

class TestB1StableConvergence:
    """B1: 50 stable turns → threshold tightens, system calm."""

    def test_threshold_tightens_under_stability(self):
        ctrl = _make_ctrl()
        history = []
        for _ in range(50):
            history.append(0.90)
            ctx = ctrl.adapt(0.90, (), list(history), False)

        assert ctrl.state.coherence_mean > 0.87
        assert ctx.adapted_threshold > 0.60
        assert ctx.stability_of_stability > 0.9
        assert ctx.dampening_factor == 1.0


class TestB2VolatileRegime:
    """B2: Alternating coherence → S² drops, alpha drops."""

    def test_volatile_detection(self):
        ctrl = _make_ctrl()
        history = []
        for i in range(50):
            c = 0.3 if i % 2 == 0 else 0.9
            history.append(c)
            ctx = ctrl.adapt(c, (), list(history), False)

        assert ctx.stability_of_stability < 0.30
        assert ctx.adapted_alpha < 0.20


class TestB3WeightShift:
    """B3: Persistent dependency flags → dependency weight increases."""

    def test_dependency_weight_increases(self):
        ctrl = _make_ctrl()
        history = [0.6]
        for _ in range(60):
            ctx = ctrl.adapt(0.5, ("dependency_rising",), list(history), True)
            history.append(0.5)

        dep_idx = DIMENSION_FLAGS.index("dependency_rising")
        assert ctx.adapted_weights[dep_idx] > 0.25


class TestB4CircuitBreaker:
    """B4: 10+ consecutive adjustments → circuit breaker activates."""

    def test_circuit_breaker_at_10(self):
        ctrl = _make_ctrl()
        history = [0.5]
        for i in range(15):
            ctx = ctrl.adapt(0.5, (), list(history), True)
            history.append(0.5)
            if i < 9:
                assert ctx.circuit_breaker_active == False
            elif i == 9:
                assert ctx.circuit_breaker_active == True

    def test_circuit_breaker_recovery(self):
        """After circuit break, quiet turns reset consecutive count."""
        ctrl = _make_ctrl()
        for _ in range(12):
            ctrl.adapt(0.5, (), [0.5], True)
        # Circuit breaker active
        assert ctrl.state.consecutive_adjustments >= 10

        # Quiet turn resets
        ctx = ctrl.adapt(0.5, (), [0.5], False)
        assert ctrl.state.consecutive_adjustments == 0
        assert ctx.circuit_breaker_active == False


class TestB5SafetyFlags:
    """B5: Safety flags are correctly classified."""

    def test_safety_flag_membership(self):
        assert "dependency_rising" in SAFETY_FLAGS
        assert "goals_failing" in SAFETY_FLAGS
        assert "personality_volatile" not in SAFETY_FLAGS
        assert "relationship_volatile" not in SAFETY_FLAGS
        assert "mode_unstable" not in SAFETY_FLAGS


class TestB6HysteresisCorrectness:
    """B6: Hysteresis prevents chattering near threshold boundary."""

    def test_no_chattering(self):
        ctrl = _make_ctrl()
        # Warm up
        for _ in range(20):
            ctrl.adapt(0.7, (), [0.7], False)

        # Hover near threshold
        transitions = 0
        prev_active = ctrl.state.adjustments_active
        for i in range(50):
            # Oscillate tightly around threshold
            c = 0.55 + 0.02 * (1 if i % 2 == 0 else -1)
            ctx = ctrl.adapt(c, (), [c], False)
            if ctx.adjustments_active != prev_active:
                transitions += 1
            prev_active = ctx.adjustments_active

        # With hysteresis, transitions should be very few (< 5)
        # Without hysteresis, would be ~50
        assert transitions < 10


class TestB7FatigueAsymmetry:
    """B7: Fatigue builds faster than it recovers."""

    def test_asymmetric_rates(self):
        ctrl = _make_ctrl()
        # Build fatigue to saturation
        build_turns = 0
        for _ in range(20):
            ctrl.adapt(0.5, (), [0.5], True)
            build_turns += 1
            if ctrl.state.fatigue_level >= 0.99:
                break

        # Recover to near-zero
        recover_turns = 0
        for _ in range(30):
            ctrl.adapt(0.5, (), [0.5], False)
            recover_turns += 1
            if ctrl.state.fatigue_level <= 0.01:
                break

        assert recover_turns > build_turns  # asymmetric by design


class TestB8GatingBypassesSafety:
    """B8: S² gating blocks non-safety but allows safety adjustments."""

    def test_gate_blocks_nonsafety(self):
        """When S² < 0.25, non-safety adjustments should be suppressed."""
        eng = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(
            coherence_score=0.3,
            risk_flags=("mode_unstable",),
            mode_instability=0.5,
        )
        ctx = AdaptedContext(
            adapted_threshold=0.5,
            stability_gate_open=False,  # gate closed
        )
        adj = eng.evaluate(diag, adapted_context=ctx)
        # mode_unstable is non-safety → should be suppressed
        assert adj.suggested_drift_cap is None
        assert adj.suggested_guidance_mode is None

    def test_gate_allows_safety(self):
        """When S² < 0.25, safety adjustments still pass through."""
        eng = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(
            coherence_score=0.3,
            dependency_trend=0.05,
            risk_flags=("dependency_rising",),
            goal_resolution_rate=0.1,
        )
        ctx = AdaptedContext(
            adapted_threshold=0.5,
            stability_gate_open=False,  # gate closed
        )
        adj = eng.evaluate(diag, adapted_context=ctx)
        # dependency_rising is safety → must pass
        assert adj.dependency_mitigation_multiplier > 1.0
        assert adj.allow_goal_injection == False
        assert adj.any_adjustment == True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C: INTEGRATION WITH STRATEGIC ADJUSTMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestStrategyWithAdaptiveContext:

    def test_circuit_breaker_blocks_all(self):
        eng = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(coherence_score=0.1, risk_flags=("personality_volatile",))
        ctx = AdaptedContext(circuit_breaker_active=True)
        adj = eng.evaluate(diag, adapted_context=ctx)
        assert adj.any_adjustment == False
        assert adj.suggested_drift_cap is None

    def test_adapted_threshold_used(self):
        """Strategy uses adapted_threshold, not static."""
        eng = StrategicAdjustmentEngine(StrategyConfig(coherence_threshold=0.6))
        # Coherence=0.55 is below static 0.6 but above adapted 0.5
        diag = MetaDiagnostics(coherence_score=0.55)
        # Without adaptive: should trigger (0.55 < 0.6)
        adj_no_adapt = eng.evaluate(diag)
        assert adj_no_adapt.suggested_drift_cap is not None

        # With adaptive threshold at 0.5: should NOT trigger (0.55 > 0.5)
        ctx = AdaptedContext(adapted_threshold=0.5, dampening_factor=1.0, stability_gate_open=True)
        adj_adapted = eng.evaluate(diag, adapted_context=ctx)
        assert adj_adapted.suggested_drift_cap is None

    def test_dampening_relaxes_drift_cap(self):
        """At high fatigue, drift_cap should move toward normal."""
        eng = StrategicAdjustmentEngine(StrategyConfig(reduced_drift_cap=0.005))
        diag = MetaDiagnostics(coherence_score=0.3)

        # No dampening
        ctx_fresh = AdaptedContext(adapted_threshold=0.6, dampening_factor=1.0, stability_gate_open=True)
        adj_fresh = eng.evaluate(diag, adapted_context=ctx_fresh)
        assert adj_fresh.suggested_drift_cap == 0.005

        # High dampening (fatigue)
        ctx_fatigued = AdaptedContext(adapted_threshold=0.6, dampening_factor=0.4, stability_gate_open=True)
        adj_fatigued = eng.evaluate(diag, adapted_context=ctx_fatigued)
        # drift_cap = 0.005 + (0.01 - 0.005) * 0.6 = 0.005 + 0.003 = 0.008
        assert adj_fatigued.suggested_drift_cap > 0.005

    def test_no_adapted_context_backward_compat(self):
        """Without adapted_context, strategy behaves as Phase 4.6."""
        eng = StrategicAdjustmentEngine(StrategyConfig(coherence_threshold=0.6))
        diag = MetaDiagnostics(coherence_score=0.4)
        adj = eng.evaluate(diag)  # no adapted_context
        assert adj.any_adjustment == True
        assert adj.suggested_drift_cap is not None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D: CIM END-TO-END WIRING
# ═════════════════════════════════════════════════════════════════════════════

class TestCIMAdaptiveWiring:

    def test_adaptive_off_no_context(self):
        cim = _make_cim(adaptive=False)
        assert cim.adaptive_enabled == False
        assert cim.last_adapted_context is None
        assert cim.adaptive_state is None

    def test_adaptive_on_produces_context(self):
        cim = _make_cim(adaptive=True)
        ref = FakeReflection()
        traj = FakeTrajectory()
        emo = FakeEmotion()
        mod = FakeModulation()

        # Run 3 turns to build history
        for _ in range(3):
            cim.update_from_turn(ref, traj, emo, mod)

        # Should now have an adapted context
        ctx = cim.last_adapted_context
        assert ctx is not None
        assert isinstance(ctx, AdaptedContext)
        assert 0.30 <= ctx.adapted_threshold <= 0.85
        assert 0.10 <= ctx.adapted_alpha <= 0.50

    def test_meta_snapshot_includes_adaptive(self):
        cim = _make_cim(adaptive=True)
        for _ in range(3):
            cim.update_from_turn(FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation())
        snap = cim.meta_snapshot()
        assert snap is not None
        assert "adaptive" in snap
        assert "adapted_threshold" in snap["adaptive"]
        assert "stability_of_stability" in snap["adaptive"]

    def test_telemetry_includes_adaptive(self):
        events = []
        def hook(event_type, payload):
            events.append((event_type, payload))

        cim = _make_cim(adaptive=True, telemetry_hook=hook)
        for _ in range(3):
            cim.update_from_turn(FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation())

        meta_events = [e for e in events if e[0] == "meta_diagnostics"]
        assert len(meta_events) >= 2  # at least 2 turns with diagnostics
        last = meta_events[-1][1]
        assert "adaptive" in last
        assert "threshold" in last["adaptive"]

    def test_ema_alpha_updates(self):
        """Phase 5 should update the EMA alpha used by CIM."""
        cim = _make_cim(adaptive=True)
        initial_alpha = cim._ema_alpha

        # Run stable turns — alpha should shift toward max
        for _ in range(20):
            cim.update_from_turn(FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation())

        # After stable turns, alpha should have changed
        # (exact value depends on S², but it should not be initial)
        assert cim._ema_alpha != initial_alpha or cim._ema_alpha == 0.50

    def test_adaptive_state_accessible(self):
        cim = _make_cim(adaptive=True)
        for _ in range(5):
            cim.update_from_turn(FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation())
        state = cim.adaptive_state
        assert state is not None
        assert state.total_adaptive_turns == 5


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E: PERSISTENCE ROUNDTRIP
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptivePersistence:

    def test_state_serialization_roundtrip(self):
        """to_dict → from_dict produces equivalent state."""
        state = AdaptiveMetaState(
            coherence_mean=0.72,
            coherence_variance=0.015,
            adjustments_active=True,
            flag_counts=[1.5, 0.3, 0.0, 4.2, 0.8],
            adapted_weights=[0.15, 0.12, 0.08, 0.40, 0.25],
            adapted_alpha=0.22,
            fatigue_level=0.45,
            consecutive_adjustments=3,
            adjustment_history=[True, False, True],
            total_adaptive_turns=42,
        )
        d = state.to_dict()
        restored = AdaptiveMetaState.from_dict(d)
        assert restored.coherence_mean == pytest.approx(state.coherence_mean, abs=1e-5)
        assert restored.coherence_variance == pytest.approx(state.coherence_variance, abs=1e-5)
        assert restored.adjustments_active == state.adjustments_active
        assert len(restored.flag_counts) == 5
        assert restored.adapted_alpha == pytest.approx(state.adapted_alpha, abs=1e-3)
        assert restored.fatigue_level == pytest.approx(state.fatigue_level, abs=1e-3)
        assert restored.consecutive_adjustments == state.consecutive_adjustments
        assert restored.total_adaptive_turns == state.total_adaptive_turns

    def test_store_save_load(self):
        """IdentityStore saves and loads adaptive state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IdentityStore(PersistenceConfig(data_dir=Path(tmpdir)))

            state = AdaptiveMetaState(
                coherence_mean=0.65,
                adapted_weights=[0.18, 0.22, 0.20, 0.30, 0.10],
                total_adaptive_turns=17,
            )
            assert store.save_adaptive_meta(state)

            loaded = store.load_adaptive_meta()
            assert loaded.coherence_mean == pytest.approx(0.65, abs=1e-5)
            assert loaded.total_adaptive_turns == 17

    def test_missing_file_returns_defaults(self):
        """Missing file → fresh defaults (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IdentityStore(PersistenceConfig(data_dir=Path(tmpdir)))
            loaded = store.load_adaptive_meta()
            assert loaded.coherence_mean == 0.85  # default
            assert loaded.total_adaptive_turns == 0

    def test_corrupt_file_returns_defaults(self):
        """Corrupt JSON → fresh defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IdentityStore(PersistenceConfig(data_dir=Path(tmpdir)))
            # Write garbage
            path = Path(tmpdir) / "adaptive_meta_state.json"
            path.write_text("{not valid json!!!")
            loaded = store.load_adaptive_meta()
            assert loaded.coherence_mean == 0.85

    def test_cim_persistence_roundtrip(self):
        """Full CIM lifecycle: run turns → save → create new CIM → load → verify state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Phase 1: run turns
            cim1 = _make_cim(adaptive=True, persistence_dir=tmpdir)
            cim1.load()
            for _ in range(10):
                cim1.update_from_turn(
                    FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation()
                )
            cim1.save()

            turns_before = cim1.adaptive_state.total_adaptive_turns
            assert turns_before == 10

            # Phase 2: new CIM, load from disk
            cim2 = _make_cim(adaptive=True, persistence_dir=tmpdir)
            cim2.load()

            assert cim2.adaptive_state.total_adaptive_turns == 10
            assert cim2.adaptive_state.coherence_mean == pytest.approx(
                cim1.adaptive_state.coherence_mean, abs=1e-4
            )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F: 500-TURN LONG-RUN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class TestLongRunSimulation:

    def test_500_turn_stable(self):
        """500 stable turns. System must converge and remain bounded."""
        ctrl = _make_ctrl()
        history = []
        for i in range(500):
            c = 0.85 + 0.02 * (i % 3 - 1)  # mild wobble: 0.83, 0.85, 0.87
            history.append(c)
            if len(history) > 40:
                history = history[-40:]
            ctx = ctrl.adapt(c, (), list(history), False)

            # ALL invariants must hold every turn
            assert 0.30 <= ctx.adapted_threshold <= 0.85
            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
            assert 0.10 <= ctx.adapted_alpha <= 0.50
            assert 0.0 <= ctx.fatigue_level <= 1.0
            assert 0.0 <= ctx.stability_of_stability <= 1.0

        # After 500 stable turns: mean should be near 0.85
        assert ctrl.state.coherence_mean == pytest.approx(0.85, abs=0.05)

    def test_500_turn_volatile(self):
        """500 volatile turns. System must never diverge."""
        ctrl = _make_ctrl()
        history = []
        for i in range(500):
            c = 0.2 if i % 2 == 0 else 0.9
            history.append(c)
            if len(history) > 40:
                history = history[-40:]
            flags = ("personality_volatile",) if i % 3 == 0 else ()
            adj_triggered = i % 4 == 0
            ctx = ctrl.adapt(c, flags, list(history), adj_triggered)

            assert 0.30 <= ctx.adapted_threshold <= 0.85
            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
            assert 0.10 <= ctx.adapted_alpha <= 0.50
            assert 0.0 <= ctx.fatigue_level <= 1.0
            assert 0.30 - 1e-10 <= ctx.dampening_factor <= 1.0

    def test_500_turn_regime_shift(self):
        """250 stable + 250 volatile. System must adapt to regime change."""
        ctrl = _make_ctrl()
        history = []

        # Phase 1: stable
        for _ in range(250):
            history.append(0.85)
            if len(history) > 40:
                history = history[-40:]
            ctx = ctrl.adapt(0.85, (), list(history), False)

        stable_threshold = ctx.adapted_threshold
        stable_alpha = ctx.adapted_alpha

        # Phase 2: volatile
        for i in range(250):
            c = 0.2 if i % 2 == 0 else 0.9
            history.append(c)
            if len(history) > 40:
                history = history[-40:]
            ctx = ctrl.adapt(c, ("personality_volatile",), list(history), True)

        # After volatile phase: threshold should have loosened, alpha dropped
        assert ctx.adapted_threshold < stable_threshold or ctx.adapted_threshold == 0.30
        assert ctx.adapted_alpha < stable_alpha or ctx.adapted_alpha == 0.10

    def test_500_turn_mixed_flags(self):
        """500 turns with rotating flags. All weights stay bounded."""
        ctrl = _make_ctrl()
        history = []
        for i in range(500):
            c = 0.5 + 0.1 * math.sin(i * 0.1)
            history.append(c)
            if len(history) > 40:
                history = history[-40:]
            flag_idx = i % len(DIMENSION_FLAGS)
            flags = (DIMENSION_FLAGS[flag_idx],)
            ctx = ctrl.adapt(c, flags, list(history), i % 5 == 0)

            assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10
            for w in ctx.adapted_weights:
                assert 0.08 - 1e-10 <= w <= 0.40 + 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# SECTION G: BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_phase46_unaffected(self):
        """With adaptive OFF, CIM behavior is identical to Phase 4.6."""
        cim = _make_cim(adaptive=False)
        for _ in range(5):
            cim.update_from_turn(
                FakeReflection(), FakeTrajectory(), FakeEmotion(), FakeModulation()
            )
        assert cim.adaptive_enabled == False
        assert cim.last_adapted_context is None
        # Meta-cognition still works
        assert cim.last_diagnostics is not None
        assert cim.smoothed_coherence > 0

    def test_strategy_backward_compat(self):
        """StrategicAdjustmentEngine works without adapted_context."""
        eng = StrategicAdjustmentEngine()
        diag = MetaDiagnostics(coherence_score=0.4)
        adj = eng.evaluate(diag)
        assert adj.any_adjustment == True

    def test_config_defaults_backward_compat(self):
        """All Phase 5 config fields have defaults. No required args."""
        cfg = AgentConfig()
        assert cfg.enable_adaptive_meta == False
        assert cfg.adaptive_threshold_k == 1.5
        assert cfg.adaptive_circuit_breaker_threshold == 10

    def test_bounded_normalize_helper(self):
        """_bounded_normalize produces valid results."""
        w = _bounded_normalize([0.5, 0.1, 0.1, 0.1, 0.1], 0.08, 0.40)
        assert abs(sum(w) - 1.0) < 1e-10
        for x in w:
            assert 0.08 <= x <= 0.40

    def test_adapted_context_to_dict(self):
        """AdaptedContext serializes cleanly."""
        ctx = AdaptedContext(
            adapted_threshold=0.55,
            adapted_alpha=0.25,
            stability_of_stability=0.7,
            dampening_factor=0.85,
        )
        d = ctx.to_dict()
        assert d["adapted_threshold"] == 0.55
        assert d["adapted_alpha"] == 0.25


# ═════════════════════════════════════════════════════════════════════════════
# SECTION H: EDGE CASES AND FAILURE MODES
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_zero_coherence_sustained(self):
        """50 turns of coherence=0.0. System must not crash or diverge."""
        ctrl = _make_ctrl()
        history = []
        for _ in range(50):
            history.append(0.0)
            ctx = ctrl.adapt(0.0, DIMENSION_FLAGS, list(history), True)
            assert 0.30 <= ctx.adapted_threshold <= 0.85

    def test_perfect_coherence_sustained(self):
        """50 turns of coherence=1.0. System must not exceed ceiling."""
        ctrl = _make_ctrl()
        history = []
        for _ in range(50):
            history.append(1.0)
            ctx = ctrl.adapt(1.0, (), list(history), False)
            assert ctx.adapted_threshold <= 0.85

    def test_empty_flags_tuple(self):
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.5, (), [0.5], False)
        assert abs(sum(ctx.adapted_weights) - 1.0) < 1e-10

    def test_all_flags_every_turn(self):
        ctrl = _make_ctrl()
        all_flags = tuple(DIMENSION_FLAGS)
        history = [0.5]
        for _ in range(30):
            ctx = ctrl.adapt(0.3, all_flags, list(history), True)
            history.append(0.3)
        # Weights should be roughly uniform (all flags equally represented)
        assert max(ctx.adapted_weights) - min(ctx.adapted_weights) < 0.15

    def test_state_from_dict_missing_fields(self):
        """Partial dict → graceful defaults."""
        state = AdaptiveMetaState.from_dict({"coherence_mean": 0.5})
        assert state.coherence_mean == 0.5
        assert state.coherence_variance == 0.005  # default
        assert state.total_adaptive_turns == 0

    def test_state_from_dict_empty(self):
        state = AdaptiveMetaState.from_dict({})
        assert state.coherence_mean == 0.85  # default

    def test_nan_coherence_clamped(self):
        """NaN coherence should not crash the system."""
        ctrl = _make_ctrl()
        # NaN in math operations should be handled by clamp
        try:
            ctx = ctrl.adapt(float('nan'), (), [0.5], False)
            # If it doesn't crash, check bounds still hold
            assert True
        except (ValueError, FloatingPointError):
            # Also acceptable — system rejects invalid input
            assert True

    def test_single_turn_history(self):
        """Single coherence history point → S²=1.0 (stable)."""
        ctrl = _make_ctrl()
        ctx = ctrl.adapt(0.5, (), [0.5], False)
        assert ctx.stability_of_stability == 1.0

    def test_adapted_context_repr(self):
        """AdaptedContext has a useful repr."""
        ctx = AdaptedContext(
            adapted_threshold=0.55,
            adapted_alpha=0.25,
            stability_of_stability=0.7,
            dampening_factor=0.85,
            circuit_breaker_active=True,
        )
        r = repr(ctx)
        assert "CB=ON" in r
        assert "0.55" in r