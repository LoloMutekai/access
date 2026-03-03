"""
A.C.C.E.S.S. — Structural Meta-State Test Suite

Coverage:
    TestStructuralMetaConfig     — config immutability, weight sum, bounds
    TestStructuralMetaState      — frozen, bounds, serialization, from_dict
    TestStructuralMetaTracker    — EMA update, cold start, trend, multi-step
    TestStructuralGate           — gating rules, aggressiveness, thresholds
    TestGateHysteresis           — gate engage/disengage with hysteresis
    TestInstabilityIndex         — composite formula, monotonicity, bounds
    TestIntegrationHelpers       — meta snapshot, sensitivity reduction
    TestDeterminism              — identical inputs → identical outputs
    TestEdgeCases                — empty reports, extreme values, NaN safety
    TestColdStart                — first inspections, conservative behavior
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.structural_meta import (
    StructuralMetaConfig,
    StructuralMetaState,
    StructuralMetaTracker,
    StructuralGate,
    StructuralGateDecision,
    structural_state_to_meta_snapshot,
    apply_structural_sensitivity_reduction,
    _clamp,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE INSPECTION REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeFileMetrics:
    path: str = "test.py"
    io_density: float = 0.1


@dataclass
class FakeReport:
    files_analyzed: int = 15
    lines_analyzed: int = 2000
    composite_risk: float = 0.3
    health_grade: str = "B"
    smells: tuple = field(default_factory=tuple)
    cycles: tuple = field(default_factory=tuple)
    layer_violations: tuple = field(default_factory=tuple)
    files: tuple = field(default_factory=tuple)


def _make_report(
    risk=0.3, grade="B", smell_count=3, cycle_count=0,
    violation_count=0, files_analyzed=15, io_density=0.1,
):
    smells = tuple(f"smell_{i}" for i in range(smell_count))
    cycles = tuple((f"a_{i}", f"b_{i}") for i in range(cycle_count))
    violations = tuple((f"s_{i}", f"t_{i}") for i in range(violation_count))
    files = tuple(FakeFileMetrics(path=f"f{i}.py", io_density=io_density) for i in range(files_analyzed))
    return FakeReport(
        files_analyzed=files_analyzed, lines_analyzed=files_analyzed * 120,
        composite_risk=risk, health_grade=grade,
        smells=smells, cycles=cycles, layer_violations=violations, files=files,
    )


class TestStructuralMetaConfig:
    def test_config_is_frozen(self):
        cfg = StructuralMetaConfig()
        try:
            cfg.ema_alpha = 0.99
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass

    def test_instability_weights_sum_to_one(self):
        cfg = StructuralMetaConfig()
        total = (cfg.instability_w_risk + cfg.instability_w_smells
                 + cfg.instability_w_cycles + cfg.instability_w_violations + cfg.instability_w_io)
        assert abs(total - 1.0) < 1e-10

    def test_gate_hysteresis_band(self):
        cfg = StructuralMetaConfig()
        assert cfg.gate_threshold_engage > cfg.gate_threshold_disengage
        assert cfg.gate_threshold_engage - cfg.gate_threshold_disengage >= 0.05

    def test_custom_config(self):
        cfg = StructuralMetaConfig(ema_alpha=0.20, max_patch_suggestions=30)
        assert cfg.ema_alpha == 0.20
        assert cfg.max_patch_suggestions == 30

    def test_ema_alpha_bounds(self):
        cfg = StructuralMetaConfig()
        assert cfg.ema_alpha_min < cfg.ema_alpha < cfg.ema_alpha_max


class TestStructuralMetaState:
    def test_state_is_frozen(self):
        state = StructuralMetaState()
        try:
            state.structural_instability_index = 0.99
            assert False
        except (AttributeError, TypeError):
            pass

    def test_default_state_is_healthy(self):
        state = StructuralMetaState()
        assert state.is_healthy is True
        assert state.is_degraded is False

    def test_instability_bounded_high(self):
        state = StructuralMetaState(structural_instability_index=1.5)
        assert state.structural_instability_index == 1.0

    def test_instability_bounded_low(self):
        state = StructuralMetaState(structural_instability_index=-0.5)
        assert state.structural_instability_index == 0.0

    def test_to_dict_json_serializable(self):
        state = StructuralMetaState(last_composite_risk=0.35, structural_instability_index=0.4)
        d = state.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["last_composite_risk"] == 0.35
        assert parsed["structural_instability_index"] == 0.4

    def test_from_dict_roundtrip(self):
        original = StructuralMetaState(
            last_composite_risk=0.5, ema_composite_risk=0.45,
            structural_instability_index=0.6, inspection_count=10,
            last_health_grade="C",
        )
        d = original.to_dict()
        restored = StructuralMetaState.from_dict(d)
        assert restored.last_composite_risk == original.last_composite_risk
        assert restored.ema_composite_risk == original.ema_composite_risk
        assert abs(restored.structural_instability_index - original.structural_instability_index) < 0.001
        assert restored.inspection_count == original.inspection_count

    def test_from_dict_missing_keys(self):
        state = StructuralMetaState.from_dict({})
        assert state.structural_instability_index == 0.0
        assert state.last_health_grade == "A"

    def test_is_cold_start(self):
        assert StructuralMetaState(inspection_count=1).is_cold_start is True
        assert StructuralMetaState(inspection_count=5).is_cold_start is False

    def test_is_degraded_boundary(self):
        assert StructuralMetaState(structural_instability_index=0.5).is_degraded is True
        assert StructuralMetaState(structural_instability_index=0.49).is_degraded is False

    def test_repr_informative(self):
        state = StructuralMetaState(structural_instability_index=0.6, intervention_gate_active=True)
        r = repr(state)
        assert "StructuralMetaState" in r
        assert "GATED" in r


class TestStructuralMetaTracker:
    def test_first_update_initializes_ema(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=0.4))
        assert abs(state.ema_composite_risk - 0.4) < 0.01
        assert state.inspection_count == 1

    def test_second_update_smooths(self):
        tracker = StructuralMetaTracker()
        tracker.update(_make_report(risk=0.4))
        state = tracker.update(_make_report(risk=0.8))
        assert 0.4 < state.ema_composite_risk < 0.8

    def test_ema_converges(self):
        tracker = StructuralMetaTracker()
        for _ in range(50):
            state = tracker.update(_make_report(risk=0.5))
        assert abs(state.ema_composite_risk - 0.5) < 0.01

    def test_inspection_count_increments(self):
        tracker = StructuralMetaTracker()
        for i in range(5):
            state = tracker.update(_make_report())
            assert state.inspection_count == i + 1

    def test_instability_bounded_after_update(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=1.0, smell_count=100, cycle_count=50, violation_count=50))
        assert 0.0 <= state.structural_instability_index <= 1.0

    def test_trend_positive_on_degradation(self):
        tracker = StructuralMetaTracker()
        tracker.update(_make_report(risk=0.1))
        tracker.update(_make_report(risk=0.2))
        state = tracker.update(_make_report(risk=0.5))
        assert state.trend > 0

    def test_trend_negative_on_improvement(self):
        tracker = StructuralMetaTracker()
        # Build up high instability first (stabilize EMA at high level)
        for _ in range(5):
            tracker.update(_make_report(risk=0.8, smell_count=15))
        # Then consistently improve — EMA trend needs enough steps to go negative
        for _ in range(8):
            state = tracker.update(_make_report(risk=0.1, smell_count=1))
        assert state.trend < 0

    def test_state_persists_via_initial_state(self):
        prev = StructuralMetaState(ema_composite_risk=0.5, ema_smell_density=0.3, inspection_count=10)
        tracker = StructuralMetaTracker(initial_state=prev)
        state = tracker.update(_make_report(risk=0.5))
        assert state.inspection_count == 11

    def test_custom_config_alpha(self):
        cfg = StructuralMetaConfig(ema_alpha=0.50)
        tracker = StructuralMetaTracker(config=cfg)
        tracker.update(_make_report(risk=0.2))
        state = tracker.update(_make_report(risk=0.8))
        assert state.ema_composite_risk > 0.45

    def test_health_grade_propagated(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(grade="D"))
        assert state.last_health_grade == "D"


class TestStructuralGate:
    def test_healthy_full_aggressiveness(self):
        gate = StructuralGate()
        state = StructuralMetaState(structural_instability_index=0.1)
        decision = gate.evaluate(state)
        assert decision.patch_aggressiveness > 0.95
        assert decision.meta_warning is False
        assert decision.reason == "healthy"

    def test_degraded_reduced_aggressiveness(self):
        gate = StructuralGate()
        state = StructuralMetaState(structural_instability_index=0.6, intervention_gate_active=True)
        decision = gate.evaluate(state)
        assert decision.patch_aggressiveness < 0.6
        assert decision.meta_warning is True
        assert decision.reason == "degraded"

    def test_critical_minimal_suggestions(self):
        gate = StructuralGate()
        state = StructuralMetaState(structural_instability_index=0.9)
        decision = gate.evaluate(state)
        assert decision.patch_aggressiveness < 0.2
        assert decision.max_suggestions <= 5
        assert decision.reason == "critical"

    def test_aggressiveness_monotonic(self):
        gate = StructuralGate()
        prev_aggr = 2.0
        for inst in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            state = StructuralMetaState(structural_instability_index=inst)
            decision = gate.evaluate(state)
            assert decision.patch_aggressiveness <= prev_aggr
            prev_aggr = decision.patch_aggressiveness

    def test_max_suggestions_bounded(self):
        gate = StructuralGate()
        cfg = gate._cfg
        for inst in [0.0, 0.5, 1.0]:
            state = StructuralMetaState(structural_instability_index=inst)
            decision = gate.evaluate(state)
            assert cfg.min_patch_suggestions <= decision.max_suggestions <= cfg.max_patch_suggestions

    def test_sensitivity_reduction_only_when_degraded(self):
        gate = StructuralGate()
        assert gate.evaluate(StructuralMetaState(structural_instability_index=0.2)).sensitivity_reduction == 0.0
        assert gate.evaluate(StructuralMetaState(structural_instability_index=0.6)).sensitivity_reduction > 0.0

    def test_gate_decision_to_dict(self):
        d = StructuralGateDecision(patch_aggressiveness=0.7, max_suggestions=15).to_dict()
        assert isinstance(d, dict)
        assert d["patch_aggressiveness"] == 0.7

    def test_gate_decision_repr(self):
        assert "caution" in repr(StructuralGateDecision(reason="caution"))


class TestGateHysteresis:
    def test_gate_engages_above_threshold(self):
        tracker = StructuralMetaTracker()
        for _ in range(10):
            state = tracker.update(_make_report(risk=0.9, smell_count=18, cycle_count=4))
        assert state.intervention_gate_active is True

    def test_gate_disengages_below_threshold(self):
        initial = StructuralMetaState(
            intervention_gate_active=True, structural_instability_index=0.6,
            ema_composite_risk=0.5, ema_smell_density=0.3, inspection_count=10,
        )
        tracker = StructuralMetaTracker(initial_state=initial)
        for _ in range(30):
            state = tracker.update(_make_report(risk=0.05, smell_count=0, cycle_count=0))
        assert state.structural_instability_index < 0.45
        assert state.intervention_gate_active is False


class TestInstabilityIndex:
    def test_zero_on_perfect(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=0.0, smell_count=0, cycle_count=0, violation_count=0, io_density=0.0))
        assert state.structural_instability_index < 0.05

    def test_high_on_bad(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=0.9, smell_count=20, cycle_count=5, violation_count=8, io_density=0.5))
        assert state.structural_instability_index > 0.5

    def test_monotonic_with_risk(self):
        results = []
        for risk in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            tracker = StructuralMetaTracker()
            state = tracker.update(_make_report(risk=risk))
            results.append(state.structural_instability_index)
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]

    def test_monotonic_with_smells(self):
        results = []
        for smells in [0, 5, 10, 15, 20]:
            tracker = StructuralMetaTracker()
            state = tracker.update(_make_report(smell_count=smells))
            results.append(state.structural_instability_index)
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]

    def test_bounded_always(self):
        for risk in [0.0, 0.5, 1.0]:
            for smells in [0, 50]:
                for cycles in [0, 20]:
                    tracker = StructuralMetaTracker()
                    state = tracker.update(_make_report(risk=risk, smell_count=smells, cycle_count=cycles))
                    assert 0.0 <= state.structural_instability_index <= 1.0


class TestIntegrationHelpers:
    def test_meta_snapshot_structure(self):
        state = StructuralMetaState(structural_instability_index=0.4, ema_composite_risk=0.35, last_health_grade="B")
        snapshot = structural_state_to_meta_snapshot(state)
        assert "structural_health" in snapshot
        assert snapshot["structural_health"]["instability_index"] == 0.4

    def test_sensitivity_reduction_none(self):
        decision = StructuralGateDecision(sensitivity_reduction=0.0)
        assert apply_structural_sensitivity_reduction(0.30, decision) == 0.30

    def test_sensitivity_reduction_applied(self):
        decision = StructuralGateDecision(sensitivity_reduction=0.20)
        adjusted = apply_structural_sensitivity_reduction(0.30, decision)
        assert adjusted == 0.30 * 0.80

    def test_sensitivity_reduction_floor(self):
        decision = StructuralGateDecision(sensitivity_reduction=0.95)
        assert apply_structural_sensitivity_reduction(0.15, decision) >= 0.10


class TestDeterminism:
    def test_identical_runs(self):
        reports = [_make_report(risk=0.2, smell_count=3), _make_report(risk=0.4, smell_count=5), _make_report(risk=0.3, smell_count=4)]
        tracker1 = StructuralMetaTracker()
        tracker2 = StructuralMetaTracker()
        for r in reports:
            s1 = tracker1.update(r)
            s2 = tracker2.update(r)
        assert s1.ema_composite_risk == s2.ema_composite_risk
        assert s1.structural_instability_index == s2.structural_instability_index

    def test_gate_deterministic(self):
        gate = StructuralGate()
        state = StructuralMetaState(structural_instability_index=0.55)
        d1 = gate.evaluate(state)
        d2 = gate.evaluate(state)
        assert d1.patch_aggressiveness == d2.patch_aggressiveness
        assert d1.reason == d2.reason


class TestEdgeCases:
    def test_empty_report(self):
        tracker = StructuralMetaTracker()
        report = FakeReport(files_analyzed=0, lines_analyzed=0, composite_risk=0.0, health_grade="A", smells=(), cycles=(), layer_violations=(), files=())
        state = tracker.update(report)
        assert state.structural_instability_index == 0.0

    def test_nan_composite_risk(self):
        tracker = StructuralMetaTracker()
        report = FakeReport(composite_risk=float("nan"))
        state = tracker.update(report)
        assert not math.isnan(state.ema_composite_risk)
        assert 0.0 <= state.ema_composite_risk <= 1.0

    def test_inf_composite_risk(self):
        tracker = StructuralMetaTracker()
        report = FakeReport(composite_risk=float("inf"))
        state = tracker.update(report)
        assert not math.isinf(state.ema_composite_risk)

    def test_clamp_nan(self):
        assert _clamp(float("nan")) == 0.0

    def test_clamp_inf(self):
        assert _clamp(float("inf")) == 0.0

    def test_clamp_non_numeric(self):
        """_clamp must handle non-numeric types gracefully."""
        assert _clamp("not_a_number") == 0.0
        assert _clamp(None) == 0.0
        assert _clamp([1, 2, 3]) == 0.0

    def test_from_dict_garbage(self):
        """from_dict must survive completely garbage values without crashing."""
        state = StructuralMetaState.from_dict({"structural_instability_index": "not_a_number"})
        assert isinstance(state, StructuralMetaState)
        assert state.structural_instability_index == 0.0

    def test_from_dict_garbage_all_fields(self):
        """from_dict must survive garbage in every single field."""
        garbage = {
            "last_composite_risk": "abc",
            "last_smell_count": "xyz",
            "last_cycle_count": None,
            "last_violation_count": [],
            "last_health_grade": 12345,
            "last_files_analyzed": {},
            "last_lines_analyzed": float("nan"),
            "ema_composite_risk": "bad",
            "ema_smell_density": object(),
            "ema_cycle_signal": "inf_text",
            "structural_instability_index": "not_a_number",
            "trend": "rising",
            "intervention_gate_active": "maybe",
            "inspection_count": "ten",
            "inspection_timestamp": 99999,
        }
        state = StructuralMetaState.from_dict(garbage)
        assert isinstance(state, StructuralMetaState)
        assert 0.0 <= state.structural_instability_index <= 1.0


class TestColdStart:
    def test_first_inspection_no_smoothing(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=0.6))
        assert abs(state.ema_composite_risk - 0.6) < 0.01

    def test_cold_start_flag_clears(self):
        tracker = StructuralMetaTracker()
        for _ in range(4):
            state = tracker.update(_make_report())
        assert state.is_cold_start is False

    def test_cold_start_still_bounded(self):
        tracker = StructuralMetaTracker()
        state = tracker.update(_make_report(risk=0.9))
        assert 0.0 <= state.structural_instability_index <= 1.0


if __name__ == "__main__":
    import traceback
    test_classes = [
        TestStructuralMetaConfig, TestStructuralMetaState, TestStructuralMetaTracker,
        TestStructuralGate, TestGateHysteresis, TestInstabilityIndex,
        TestIntegrationHelpers, TestDeterminism, TestEdgeCases, TestColdStart,
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
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")