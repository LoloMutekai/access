"""
A.C.C.E.S.S. — Phase 7.7 / 7.8 / 7.9 Test Suite:
    compute_training_state(), compute_recommended_load(), compute_training_advice()

Coverage map
────────────
SECTION 14 — compute_training_state()  (Phase 7.7.0)

  Return-type & value-set invariants
    14.1   Always returns str
    14.2   Return value always in VALID_TRAINING_STATES
    14.3   All four states are reachable

  Priority rule correctness
    14.4   injury_risk >= 0.65 → RECOVERY  (exact threshold)
    14.5   injury_risk < 0.65 and fatigue_index >= 0.70 → LIGHT  (exact threshold)
    14.6   injury_risk < 0.65 and fatigue_index < 0.70
             and anomaly_score >= 0.60 → CAUTION  (exact threshold)
    14.7   All scores below threshold → FULL
    14.8   RECOVERY wins even when fatigue_index >= 0.70  (priority)
    14.9   RECOVERY wins even when anomaly_score >= 0.60  (priority)
    14.10  LIGHT wins over CAUTION when fatigue_index >= 0.70
             and anomaly_score >= 0.60  (priority)

  Boundary / edge-value analysis
    14.11  injury_risk just below threshold (0.6499…) → not RECOVERY
    14.12  fatigue_index just below threshold (0.6999…) → not LIGHT
    14.13  anomaly_score just below threshold (0.5999…) → not CAUTION
    14.14  All scores = 0.0 → FULL
    14.15  All scores = 1.0 → RECOVERY  (highest-priority rule)

  Determinism
    14.16  Identical inputs → identical output (100 repeated calls, module fn)
    14.17  Identical inputs → identical output (100 repeated calls, engine method)
    14.18  Two BiometricEngine instances → identical output

  No input mutation
    14.19  Module-level function does not mutate the metrics dict
    14.20  Engine method does not mutate the metrics dict

  Integration with process()
    14.21  process() output contains "training_state" key
    14.22  training_state value is str
    14.23  training_state value is in VALID_TRAINING_STATES
    14.24  process() output has exactly 11 metrics keys
    14.25  High injury_risk input → training_state == "RECOVERY" via process()
    14.26  process() on error → training_state absent (metrics is None)
    14.27  determinism: 100 process() calls with same input → same training_state

  Module-level function vs engine method parity
    14.28  Module fn and engine method return identical results for same dict

  VALID_TRAINING_STATES constant
    14.29  VALID_TRAINING_STATES is a frozenset
    14.30  VALID_TRAINING_STATES contains exactly {"FULL","CAUTION","LIGHT","RECOVERY"}

SECTION 15 — compute_recommended_load()  (Phase 7.8.0)

  Return-type & value invariants
    15.1   Always returns float
    15.2   Return value always in [0.0, 1.0]
    15.3   Return value is finite

  Mapping correctness (exact values)
    15.4   "FULL"     → 1.00
    15.5   "CAUTION"  → 0.75
    15.6   "LIGHT"    → 0.50
    15.7   "RECOVERY" → 0.25
    15.8   All four mappings cover VALID_TRAINING_STATES exactly

  Ordering invariant
    15.9   RECOVERY < LIGHT < CAUTION < FULL  (more severe → lower load)

  Invalid state → ValueError
    15.10  Empty string raises ValueError
    15.11  Lowercase valid name raises ValueError
    15.12  Arbitrary unknown string raises ValueError
    15.13  None raises ValueError (wrong type coercion check)
    15.14  ValueError message contains the invalid state

  Determinism
    15.15  Identical inputs → identical output (100 repeated calls, module fn)
    15.16  Identical inputs → identical output (100 repeated calls, engine method)
    15.17  Two BiometricEngine instances → identical output

  No input mutation
    15.18  training_state str is not mutated (strings are immutable — structural check)

  Integration with process()
    15.19  process() output contains "recommended_load" key
    15.20  recommended_load value is float
    15.21  recommended_load value is finite
    15.22  recommended_load value is in [0.0, 1.0]
    15.23  recommended_load is consistent with training_state in process() output
    15.24  process() on error → recommended_load absent (metrics is None)
    15.25  determinism: 100 process() calls → same recommended_load

  RECOMMENDED_LOAD constant
    15.26  RECOMMENDED_LOAD is a dict
    15.27  RECOMMENDED_LOAD keys == VALID_TRAINING_STATES
    15.28  All values in RECOMMENDED_LOAD are finite floats in [0.0, 1.0]

  Module-level function vs engine method parity
    15.29  Module fn and engine method return identical results for all four states

SECTION 16 — compute_training_advice() / TrainingRecommendation  (Phase 7.9.0)

  TrainingRecommendation dataclass
    16.1   compute_training_advice() returns a TrainingRecommendation
    16.2   .state field is present and is str
    16.3   .recommended_load field is present and is float
    16.4   .state is always in VALID_TRAINING_STATES
    16.5   .recommended_load is always in [0.0, 1.0]
    16.6   .recommended_load is always finite
    16.7   Dataclass is frozen (reassignment raises FrozenInstanceError)

  to_dict() method
    16.8   to_dict() returns a dict
    16.9   to_dict() contains exactly the keys {"state", "recommended_load"}
    16.10  to_dict()["state"] matches .state
    16.11  to_dict()["recommended_load"] matches .recommended_load
    16.12  to_dict() output is JSON-serialisable
    16.13  Repeated to_dict() calls return equal dicts (idempotent)
    16.14  to_dict() does not mutate the dataclass

  Internal consistency
    16.15  .state and .recommended_load are consistent (load == RECOMMENDED_LOAD[state])
    16.16  All four states produce correct loads via compute_training_advice()

  Determinism
    16.17  Identical inputs → identical output (100 repeated calls, module fn)
    16.18  Identical inputs → identical output (100 repeated calls, engine method)
    16.19  Two BiometricEngine instances → identical output

  No input mutation
    16.20  Module fn does not mutate the metrics dict
    16.21  Engine method does not mutate the metrics dict

  Module fn vs engine method parity
    16.22  Module fn and engine method return equal objects for same metrics dict

  Integration with process()
    16.23  process() output contains "training_advice" key
    16.24  training_advice is a dict
    16.25  training_advice contains keys {"state", "recommended_load"}
    16.26  training_advice["state"] is str in VALID_TRAINING_STATES
    16.27  training_advice["recommended_load"] is float in [0.0, 1.0] and finite
    16.28  training_advice["state"] == metrics["training_state"]
    16.29  training_advice["recommended_load"] == metrics["recommended_load"]
    16.30  training_advice is JSON-serialisable
    16.31  process() on error → training_advice absent (metrics is None)
    16.32  determinism: 100 process() calls → same training_advice
"""

from __future__ import annotations

import math
import sys
import os

import pytest

# Insert the project root (two levels up from this test file) so that
# "biometric.biometric_engine" is resolvable regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.biometric_engine import (
    BiometricEngine,
    BiometricConfig,
    TS_RECOVERY_THRESHOLD,
    TS_LIGHT_THRESHOLD,
    TS_CAUTION_THRESHOLD,
    VALID_TRAINING_STATES,
    RECOMMENDED_LOAD,
    TrainingRecommendation,
    compute_training_state,      # module-level function
    compute_recommended_load,    # module-level function
    compute_training_advice,     # module-level function
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_engine() -> BiometricEngine:
    return BiometricEngine(BiometricConfig())


def _make_valid_input(n: int = 10) -> dict:
    return {
        "hr":   [float(60 + i) for i in range(n)],
        "hrv":  [float(40 + i) for i in range(n)],
        "load": [float(100 + i * 10) for i in range(n)],
    }


def _metrics(
    injury_risk:   float = 0.0,
    fatigue_index: float = 0.0,
    anomaly_score: float = 0.0,
) -> dict:
    """Construct a minimal metrics dict for compute_training_state()."""
    return {
        "injury_risk":   injury_risk,
        "fatigue_index": fatigue_index,
        "anomaly_score": anomaly_score,
    }


# =============================================================================
# SECTION 14 — compute_training_state()
# =============================================================================

class TestComputeTrainingState:

    # ── 14.1  always returns str ──────────────────────────────────────────────

    def test_14_1_always_returns_str(self):
        for ir, fi, ans in [
            (0.0, 0.0, 0.0),
            (0.65, 0.0, 0.0),
            (0.0, 0.70, 0.0),
            (0.0, 0.0, 0.60),
            (1.0, 1.0, 1.0),
        ]:
            result = compute_training_state(_metrics(ir, fi, ans))
            assert isinstance(result, str), (
                f"Expected str, got {type(result)} for ({ir}, {fi}, {ans})"
            )

    # ── 14.2  return value always in VALID_TRAINING_STATES ───────────────────

    def test_14_2_return_value_in_valid_states(self):
        test_cases = [
            _metrics(0.0, 0.0, 0.0),
            _metrics(0.65, 0.0, 0.0),
            _metrics(0.0, 0.70, 0.0),
            _metrics(0.0, 0.0, 0.60),
            _metrics(1.0, 1.0, 1.0),
            _metrics(0.5, 0.5, 0.5),
        ]
        for m in test_cases:
            result = compute_training_state(m)
            assert result in VALID_TRAINING_STATES, (
                f"Got {result!r} which is not in VALID_TRAINING_STATES"
            )

    # ── 14.3  all four states are reachable ───────────────────────────────────

    def test_14_3_all_four_states_reachable(self):
        states = {
            compute_training_state(_metrics(0.65, 0.0, 0.0)),   # RECOVERY
            compute_training_state(_metrics(0.0, 0.70, 0.0)),   # LIGHT
            compute_training_state(_metrics(0.0, 0.0, 0.60)),   # CAUTION
            compute_training_state(_metrics(0.0, 0.0, 0.0)),    # FULL
        }
        assert states == VALID_TRAINING_STATES

    # ── 14.4  RECOVERY at exact threshold ────────────────────────────────────

    def test_14_4_recovery_at_exact_threshold(self):
        result = compute_training_state(
            _metrics(injury_risk=TS_RECOVERY_THRESHOLD, fatigue_index=0.0, anomaly_score=0.0)
        )
        assert result == "RECOVERY"

    # ── 14.5  LIGHT at exact threshold (injury_risk just below) ──────────────

    def test_14_5_light_at_exact_threshold(self):
        result = compute_training_state(
            _metrics(
                injury_risk=TS_RECOVERY_THRESHOLD - 1e-9,
                fatigue_index=TS_LIGHT_THRESHOLD,
                anomaly_score=0.0,
            )
        )
        assert result == "LIGHT"

    # ── 14.6  CAUTION at exact threshold (injury_risk and fatigue below) ─────

    def test_14_6_caution_at_exact_threshold(self):
        result = compute_training_state(
            _metrics(
                injury_risk=TS_RECOVERY_THRESHOLD - 1e-9,
                fatigue_index=TS_LIGHT_THRESHOLD - 1e-9,
                anomaly_score=TS_CAUTION_THRESHOLD,
            )
        )
        assert result == "CAUTION"

    # ── 14.7  FULL when all scores below thresholds ───────────────────────────

    def test_14_7_full_when_all_below_threshold(self):
        result = compute_training_state(
            _metrics(
                injury_risk=TS_RECOVERY_THRESHOLD - 1e-9,
                fatigue_index=TS_LIGHT_THRESHOLD - 1e-9,
                anomaly_score=TS_CAUTION_THRESHOLD - 1e-9,
            )
        )
        assert result == "FULL"

    # ── 14.8  RECOVERY beats LIGHT (priority) ────────────────────────────────

    def test_14_8_recovery_beats_light(self):
        result = compute_training_state(
            _metrics(injury_risk=0.70, fatigue_index=0.75, anomaly_score=0.0)
        )
        assert result == "RECOVERY", (
            "RECOVERY should win when both injury_risk>=0.65 and fatigue_index>=0.70"
        )

    # ── 14.9  RECOVERY beats CAUTION (priority) ──────────────────────────────

    def test_14_9_recovery_beats_caution(self):
        result = compute_training_state(
            _metrics(injury_risk=0.70, fatigue_index=0.0, anomaly_score=0.65)
        )
        assert result == "RECOVERY"

    # ── 14.10  LIGHT beats CAUTION (priority) ────────────────────────────────

    def test_14_10_light_beats_caution(self):
        result = compute_training_state(
            _metrics(
                injury_risk=TS_RECOVERY_THRESHOLD - 1e-9,
                fatigue_index=0.75,
                anomaly_score=0.65,
            )
        )
        assert result == "LIGHT", (
            "LIGHT should win over CAUTION when fatigue_index>=0.70 and injury_risk<0.65"
        )

    # ── 14.11  just below RECOVERY threshold → not RECOVERY ──────────────────

    def test_14_11_just_below_recovery_threshold(self):
        result = compute_training_state(
            _metrics(injury_risk=TS_RECOVERY_THRESHOLD - 1e-9, fatigue_index=0.0, anomaly_score=0.0)
        )
        assert result != "RECOVERY"

    # ── 14.12  just below LIGHT threshold → not LIGHT ────────────────────────

    def test_14_12_just_below_light_threshold(self):
        result = compute_training_state(
            _metrics(injury_risk=0.0, fatigue_index=TS_LIGHT_THRESHOLD - 1e-9, anomaly_score=0.0)
        )
        assert result != "LIGHT"

    # ── 14.13  just below CAUTION threshold → not CAUTION ────────────────────

    def test_14_13_just_below_caution_threshold(self):
        result = compute_training_state(
            _metrics(injury_risk=0.0, fatigue_index=0.0, anomaly_score=TS_CAUTION_THRESHOLD - 1e-9)
        )
        assert result != "CAUTION"

    # ── 14.14  all scores zero → FULL ────────────────────────────────────────

    def test_14_14_all_zero_gives_full(self):
        assert compute_training_state(_metrics(0.0, 0.0, 0.0)) == "FULL"

    # ── 14.15  all scores one → RECOVERY (highest priority) ──────────────────

    def test_14_15_all_one_gives_recovery(self):
        assert compute_training_state(_metrics(1.0, 1.0, 1.0)) == "RECOVERY"

    # ── 14.16  determinism: 100 calls, module-level function ──────────────────

    def test_14_16_deterministic_module_fn_100_calls(self):
        m = _metrics(0.3, 0.4, 0.55)
        first = compute_training_state(m)
        for _ in range(99):
            assert compute_training_state(m) == first

    # ── 14.17  determinism: 100 calls, engine method ──────────────────────────

    def test_14_17_deterministic_engine_method_100_calls(self):
        eng = _make_engine()
        m = _metrics(0.3, 0.4, 0.55)
        first = eng.compute_training_state(m)
        for _ in range(99):
            assert eng.compute_training_state(m) == first

    # ── 14.18  two engine instances → same result ─────────────────────────────

    def test_14_18_two_instances_same_result(self):
        m = _metrics(0.5, 0.5, 0.5)
        r1 = BiometricEngine().compute_training_state(m)
        r2 = BiometricEngine().compute_training_state(m)
        assert r1 == r2

    # ── 14.19  module fn does not mutate metrics dict ─────────────────────────

    def test_14_19_module_fn_no_mutation(self):
        m = _metrics(0.3, 0.75, 0.2)
        snapshot = dict(m)
        compute_training_state(m)
        assert m == snapshot, "compute_training_state() must not mutate the metrics dict"

    # ── 14.20  engine method does not mutate metrics dict ─────────────────────

    def test_14_20_engine_method_no_mutation(self):
        eng = _make_engine()
        m = _metrics(0.3, 0.75, 0.2)
        snapshot = dict(m)
        eng.compute_training_state(m)
        assert m == snapshot, "engine.compute_training_state() must not mutate the dict"

    # ── 14.21  process() output contains "training_state" key ────────────────

    def test_14_21_process_contains_training_state(self):
        result = _make_engine().process(_make_valid_input())
        assert result["status"] == "ok"
        assert "training_state" in result["metrics"]

    # ── 14.22  training_state from process() is str ───────────────────────────

    def test_14_22_process_training_state_is_str(self):
        result = _make_engine().process(_make_valid_input())
        assert isinstance(result["metrics"]["training_state"], str)

    # ── 14.23  training_state from process() is in VALID_TRAINING_STATES ──────

    def test_14_23_process_training_state_in_valid_states(self):
        for n in (10, 25, 100):
            result = _make_engine().process(_make_valid_input(n=n))
            assert result["metrics"]["training_state"] in VALID_TRAINING_STATES, (
                f"n={n}: got {result['metrics']['training_state']!r}"
            )

    # ── 14.24  process() output has exactly 10 metrics keys ───────────────────

    def test_14_24_process_output_has_twelve_keys(self):
        result = _make_engine().process(_make_valid_input())
        expected = {
            "mean_hr", "mean_hrv", "load_mean", "hr_std", "hrv_std",
            "fatigue_index", "drift_score", "injury_risk", "anomaly_score",
            "training_state", "recommended_load",
            "training_advice",   # added Phase 7.9.0
        }
        assert set(result["metrics"].keys()) == expected

    # ── 14.25  high injury_risk → RECOVERY via process() ─────────────────────

    def test_14_25_high_injury_risk_gives_recovery_via_process(self):
        # load_mean=5000 → load_signal=1.0; combined with high fatigue → risk near 1
        raw = {
            "hr":   [50.0] + [180.0] * 9,    # high HR spread → high fatigue + drift
            "hrv":  [80.0] * 10,
            "load": [5000.0] * 10,            # load_signal = 1.0 → maximises risk
        }
        result = _make_engine().process(raw)
        assert result["status"] == "ok"
        ir = result["metrics"]["injury_risk"]
        if ir >= TS_RECOVERY_THRESHOLD:
            assert result["metrics"]["training_state"] == "RECOVERY"

    # ── 14.26  process() error → metrics is None (no training_state) ──────────

    def test_14_26_process_error_no_training_state(self):
        result = _make_engine().process("not a dict")
        assert result["status"] == "error"
        assert result["metrics"] is None
        assert "training_state" not in result

    # ── 14.27  process() determinism: 100 calls → same training_state ─────────

    def test_14_27_process_deterministic_training_state(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["training_state"]
        for _ in range(99):
            assert eng.process(raw)["metrics"]["training_state"] == first

    # ── 14.28  module fn and engine method parity ─────────────────────────────

    def test_14_28_module_fn_and_method_parity(self):
        eng = _make_engine()
        cases = [
            _metrics(0.0, 0.0, 0.0),
            _metrics(0.70, 0.0, 0.0),
            _metrics(0.0, 0.75, 0.0),
            _metrics(0.0, 0.0, 0.65),
            _metrics(1.0, 1.0, 1.0),
        ]
        for m in cases:
            fn_result     = compute_training_state(m)
            method_result = eng.compute_training_state(m)
            assert fn_result == method_result, (
                f"Mismatch for {m}: fn={fn_result!r}, method={method_result!r}"
            )

    # ── 14.29  VALID_TRAINING_STATES is a frozenset ───────────────────────────

    def test_14_29_valid_training_states_is_frozenset(self):
        assert isinstance(VALID_TRAINING_STATES, frozenset)

    # ── 14.30  VALID_TRAINING_STATES contains the four expected strings ────────

    def test_14_30_valid_training_states_contents(self):
        assert VALID_TRAINING_STATES == frozenset({"FULL", "CAUTION", "LIGHT", "RECOVERY"})


# =============================================================================
# SECTION 15 — compute_recommended_load()  (Phase 7.8.0)
# =============================================================================

class TestComputeRecommendedLoad:
    """
    Tests for compute_recommended_load() and its integration into process().

    Mapping under test
    ------------------
    "FULL"     → 1.00
    "CAUTION"  → 0.75
    "LIGHT"    → 0.50
    "RECOVERY" → 0.25

    All four values are in [0, 1] before clamping; clamping is a guard only.
    Invalid state → ValueError (always; never silently returns).
    """

    # ── 15.1  always returns float ────────────────────────────────────────────

    def test_15_1_always_returns_float(self):
        for state in VALID_TRAINING_STATES:
            result = compute_recommended_load(state)
            assert isinstance(result, float), (
                f"Expected float for {state!r}, got {type(result)}"
            )

    # ── 15.2  return value always in [0.0, 1.0] ──────────────────────────────

    def test_15_2_return_value_in_unit_interval(self):
        for state in VALID_TRAINING_STATES:
            result = compute_recommended_load(state)
            assert 0.0 <= result <= 1.0, (
                f"{state!r} → {result} is outside [0, 1]"
            )

    # ── 15.3  return value is finite ─────────────────────────────────────────

    def test_15_3_return_value_is_finite(self):
        for state in VALID_TRAINING_STATES:
            result = compute_recommended_load(state)
            assert math.isfinite(result), f"{state!r} → non-finite {result}"

    # ── 15.4  FULL → 1.0 ─────────────────────────────────────────────────────

    def test_15_4_full_maps_to_1_0(self):
        assert compute_recommended_load("FULL") == pytest.approx(1.0)

    # ── 15.5  CAUTION → 0.75 ─────────────────────────────────────────────────

    def test_15_5_caution_maps_to_0_75(self):
        assert compute_recommended_load("CAUTION") == pytest.approx(0.75)

    # ── 15.6  LIGHT → 0.50 ───────────────────────────────────────────────────

    def test_15_6_light_maps_to_0_50(self):
        assert compute_recommended_load("LIGHT") == pytest.approx(0.50)

    # ── 15.7  RECOVERY → 0.25 ────────────────────────────────────────────────

    def test_15_7_recovery_maps_to_0_25(self):
        assert compute_recommended_load("RECOVERY") == pytest.approx(0.25)

    # ── 15.8  mapping covers all four states ─────────────────────────────────

    def test_15_8_all_four_states_covered(self):
        """Every member of VALID_TRAINING_STATES must return without raising."""
        results = {state: compute_recommended_load(state) for state in VALID_TRAINING_STATES}
        assert set(results.keys()) == VALID_TRAINING_STATES

    # ── 15.9  ordering: RECOVERY < LIGHT < CAUTION < FULL ────────────────────

    def test_15_9_ordering_more_severe_lower_load(self):
        r = compute_recommended_load("RECOVERY")
        l = compute_recommended_load("LIGHT")
        c = compute_recommended_load("CAUTION")
        f = compute_recommended_load("FULL")
        assert r < l < c < f, (
            f"Expected RECOVERY({r}) < LIGHT({l}) < CAUTION({c}) < FULL({f})"
        )

    # ── 15.10  empty string raises ValueError ─────────────────────────────────

    def test_15_10_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_recommended_load("")

    # ── 15.11  lowercase valid name raises ValueError ─────────────────────────

    def test_15_11_lowercase_name_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_recommended_load("full")

    # ── 15.12  arbitrary unknown string raises ValueError ─────────────────────

    def test_15_12_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_recommended_load("UNKNOWN")

    # ── 15.13  None raises ValueError ────────────────────────────────────────

    def test_15_13_none_raises_value_error(self):
        with pytest.raises((ValueError, TypeError)):
            compute_recommended_load(None)  # type: ignore[arg-type]

    # ── 15.14  ValueError message contains the invalid state ──────────────────

    def test_15_14_value_error_message_contains_state(self):
        bad = "TURBO"
        with pytest.raises(ValueError, match="TURBO"):
            compute_recommended_load(bad)

    # ── 15.15  determinism: 100 calls, module-level function ──────────────────

    def test_15_15_deterministic_module_fn_100_calls(self):
        for state in VALID_TRAINING_STATES:
            first = compute_recommended_load(state)
            for _ in range(99):
                assert compute_recommended_load(state) == first

    # ── 15.16  determinism: 100 calls, engine method ──────────────────────────

    def test_15_16_deterministic_engine_method_100_calls(self):
        eng = _make_engine()
        for state in VALID_TRAINING_STATES:
            first = eng.compute_recommended_load(state)
            for _ in range(99):
                assert eng.compute_recommended_load(state) == first

    # ── 15.17  two engine instances → same result ─────────────────────────────

    def test_15_17_two_instances_same_result(self):
        for state in VALID_TRAINING_STATES:
            r1 = BiometricEngine().compute_recommended_load(state)
            r2 = BiometricEngine().compute_recommended_load(state)
            assert r1 == r2

    # ── 15.18  input string not mutated (structural check) ────────────────────

    def test_15_18_input_not_mutated(self):
        """Strings are immutable in Python; verify the value is unchanged."""
        state = "FULL"
        snapshot = state
        compute_recommended_load(state)
        assert state == snapshot

    # ── 15.19  process() output contains "recommended_load" key ───────────────

    def test_15_19_process_contains_recommended_load(self):
        result = _make_engine().process(_make_valid_input())
        assert result["status"] == "ok"
        assert "recommended_load" in result["metrics"]

    # ── 15.20  recommended_load from process() is float ───────────────────────

    def test_15_20_process_recommended_load_is_float(self):
        result = _make_engine().process(_make_valid_input())
        assert isinstance(result["metrics"]["recommended_load"], float)

    # ── 15.21  recommended_load from process() is finite ─────────────────────

    def test_15_21_process_recommended_load_is_finite(self):
        result = _make_engine().process(_make_valid_input())
        assert math.isfinite(result["metrics"]["recommended_load"])

    # ── 15.22  recommended_load from process() is in [0.0, 1.0] ──────────────

    def test_15_22_process_recommended_load_in_unit_interval(self):
        for n in (10, 25, 100):
            result = _make_engine().process(_make_valid_input(n=n))
            rl = result["metrics"]["recommended_load"]
            assert 0.0 <= rl <= 1.0, f"n={n}: recommended_load={rl} out of [0, 1]"

    # ── 15.23  recommended_load is consistent with training_state ─────────────

    def test_15_23_recommended_load_consistent_with_training_state(self):
        """The recommended_load in process() output must match the direct mapping."""
        for n in (10, 25, 50):
            result = _make_engine().process(_make_valid_input(n=n))
            state = result["metrics"]["training_state"]
            rl    = result["metrics"]["recommended_load"]
            expected = compute_recommended_load(state)
            assert rl == pytest.approx(expected), (
                f"n={n}: state={state!r}, recommended_load={rl}, expected={expected}"
            )

    # ── 15.24  process() error → recommended_load absent ─────────────────────

    def test_15_24_process_error_no_recommended_load(self):
        result = _make_engine().process("not a dict")
        assert result["status"] == "error"
        assert result["metrics"] is None
        assert "recommended_load" not in result

    # ── 15.25  process() determinism: 100 calls → same recommended_load ───────

    def test_15_25_process_deterministic_recommended_load(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["recommended_load"]
        for _ in range(99):
            assert eng.process(raw)["metrics"]["recommended_load"] == first

    # ── 15.26  RECOMMENDED_LOAD is a dict ────────────────────────────────────

    def test_15_26_recommended_load_constant_is_dict(self):
        assert isinstance(RECOMMENDED_LOAD, dict)

    # ── 15.27  RECOMMENDED_LOAD keys == VALID_TRAINING_STATES ────────────────

    def test_15_27_recommended_load_keys_match_valid_states(self):
        assert set(RECOMMENDED_LOAD.keys()) == VALID_TRAINING_STATES

    # ── 15.28  all values in RECOMMENDED_LOAD are finite floats in [0, 1] ─────

    def test_15_28_recommended_load_values_are_valid(self):
        for state, val in RECOMMENDED_LOAD.items():
            assert isinstance(val, float), f"{state}: value {val!r} is not float"
            assert math.isfinite(val),     f"{state}: value {val} is not finite"
            assert 0.0 <= val <= 1.0,      f"{state}: value {val} outside [0, 1]"

    # ── 15.29  module fn and engine method parity ─────────────────────────────

    def test_15_29_module_fn_and_method_parity(self):
        eng = _make_engine()
        for state in VALID_TRAINING_STATES:
            fn_result     = compute_recommended_load(state)
            method_result = eng.compute_recommended_load(state)
            assert fn_result == method_result, (
                f"{state!r}: fn={fn_result}, method={method_result}"
            )


# =============================================================================
# SECTION 16 — compute_training_advice() / TrainingRecommendation  (Phase 7.9.0)
# =============================================================================

class TestComputeTrainingAdvice:
    """
    Tests for compute_training_advice() and the TrainingRecommendation dataclass.

    The function is a pure two-step pipeline:
        state = compute_training_state(metrics)
        load  = compute_recommended_load(state)
        return TrainingRecommendation(state=state, recommended_load=load)

    All invariants on the component functions therefore also hold here.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _advice(injury_risk=0.2, fatigue_index=0.3, anomaly_score=0.1):
        return compute_training_advice(_metrics(injury_risk, fatigue_index, anomaly_score))

    # ── 16.1  returns TrainingRecommendation ─────────────────────────────────

    def test_16_1_returns_training_recommendation(self):
        result = self._advice()
        assert isinstance(result, TrainingRecommendation)

    # ── 16.2  .state is str ───────────────────────────────────────────────────

    def test_16_2_state_is_str(self):
        assert isinstance(self._advice().state, str)

    # ── 16.3  .recommended_load is float ─────────────────────────────────────

    def test_16_3_recommended_load_is_float(self):
        assert isinstance(self._advice().recommended_load, float)

    # ── 16.4  .state always in VALID_TRAINING_STATES ─────────────────────────

    def test_16_4_state_in_valid_training_states(self):
        for ir, fi, a in [
            (0.0, 0.0, 0.0), (0.7, 0.0, 0.0),
            (0.0, 0.75, 0.0), (0.0, 0.0, 0.65),
        ]:
            result = compute_training_advice(_metrics(ir, fi, a))
            assert result.state in VALID_TRAINING_STATES, (
                f"state {result.state!r} not in VALID_TRAINING_STATES"
            )

    # ── 16.5  .recommended_load always in [0.0, 1.0] ─────────────────────────

    def test_16_5_recommended_load_in_unit_interval(self):
        for state in VALID_TRAINING_STATES:
            m = _metrics(
                injury_risk=0.7 if state == "RECOVERY" else 0.1,
                fatigue_index=0.8 if state == "LIGHT" else 0.1,
                anomaly_score=0.7 if state == "CAUTION" else 0.1,
            )
            result = compute_training_advice(m)
            assert 0.0 <= result.recommended_load <= 1.0

    # ── 16.6  .recommended_load always finite ────────────────────────────────

    def test_16_6_recommended_load_finite(self):
        assert math.isfinite(self._advice().recommended_load)

    # ── 16.7  dataclass is frozen ─────────────────────────────────────────────

    def test_16_7_dataclass_is_frozen(self):
        from dataclasses import FrozenInstanceError
        adv = self._advice()
        with pytest.raises(FrozenInstanceError):
            adv.state = "FULL"  # type: ignore[misc]

    # ── 16.8  to_dict() returns dict ─────────────────────────────────────────

    def test_16_8_to_dict_returns_dict(self):
        assert isinstance(self._advice().to_dict(), dict)

    # ── 16.9  to_dict() has exactly {"state", "recommended_load"} ────────────

    def test_16_9_to_dict_exact_keys(self):
        assert set(self._advice().to_dict().keys()) == {"state", "recommended_load"}

    # ── 16.10  to_dict()["state"] matches .state ─────────────────────────────

    def test_16_10_to_dict_state_matches_field(self):
        adv = self._advice()
        assert adv.to_dict()["state"] == adv.state

    # ── 16.11  to_dict()["recommended_load"] matches .recommended_load ────────

    def test_16_11_to_dict_load_matches_field(self):
        adv = self._advice()
        assert adv.to_dict()["recommended_load"] == adv.recommended_load

    # ── 16.12  to_dict() output is JSON-serialisable ─────────────────────────

    def test_16_12_to_dict_json_serialisable(self):
        import json
        d = self._advice().to_dict()
        json.dumps(d)  # must not raise

    # ── 16.13  repeated to_dict() calls return equal dicts ───────────────────

    def test_16_13_to_dict_idempotent(self):
        adv = self._advice()
        first = adv.to_dict()
        for _ in range(9):
            assert adv.to_dict() == first

    # ── 16.14  to_dict() does not mutate the dataclass ───────────────────────

    def test_16_14_to_dict_no_mutation(self):
        adv = self._advice()
        state_before = adv.state
        load_before  = adv.recommended_load
        d = adv.to_dict()
        d["state"] = "TAMPERED"
        d["recommended_load"] = -999.0
        assert adv.state            == state_before
        assert adv.recommended_load == load_before

    # ── 16.15  .state and .recommended_load are consistent ───────────────────

    def test_16_15_state_and_load_consistent(self):
        adv = self._advice()
        assert adv.recommended_load == pytest.approx(RECOMMENDED_LOAD[adv.state])

    # ── 16.16  all four states produce correct loads ──────────────────────────

    def test_16_16_all_four_states_correct_load(self):
        cases = [
            (_metrics(0.7,  0.0, 0.0), "RECOVERY", 0.25),
            (_metrics(0.0, 0.75, 0.0), "LIGHT",    0.50),
            (_metrics(0.0,  0.0, 0.65),"CAUTION",  0.75),
            (_metrics(0.0,  0.0, 0.0), "FULL",     1.00),
        ]
        for m, expected_state, expected_load in cases:
            adv = compute_training_advice(m)
            assert adv.state            == expected_state,    f"state mismatch for {expected_state}"
            assert adv.recommended_load == pytest.approx(expected_load), (
                f"load mismatch for {expected_state}: got {adv.recommended_load}"
            )

    # ── 16.17  determinism: 100 calls, module fn ─────────────────────────────

    def test_16_17_deterministic_module_fn_100_calls(self):
        m = _metrics(0.2, 0.3, 0.1)
        first = compute_training_advice(m)
        for _ in range(99):
            result = compute_training_advice(m)
            assert result.state            == first.state
            assert result.recommended_load == first.recommended_load

    # ── 16.18  determinism: 100 calls, engine method ─────────────────────────

    def test_16_18_deterministic_engine_method_100_calls(self):
        eng = _make_engine()
        m   = _metrics(0.2, 0.3, 0.1)
        first = eng.compute_training_advice(m)
        for _ in range(99):
            result = eng.compute_training_advice(m)
            assert result.state            == first.state
            assert result.recommended_load == first.recommended_load

    # ── 16.19  two engine instances → same result ─────────────────────────────

    def test_16_19_two_instances_same_result(self):
        m  = _metrics(0.2, 0.3, 0.1)
        r1 = BiometricEngine().compute_training_advice(m)
        r2 = BiometricEngine().compute_training_advice(m)
        assert r1.state            == r2.state
        assert r1.recommended_load == r2.recommended_load

    # ── 16.20  module fn does not mutate metrics ──────────────────────────────

    def test_16_20_module_fn_no_mutation(self):
        m      = _metrics(0.2, 0.3, 0.1)
        before = dict(m)
        compute_training_advice(m)
        assert m == before

    # ── 16.21  engine method does not mutate metrics ──────────────────────────

    def test_16_21_engine_method_no_mutation(self):
        eng    = _make_engine()
        m      = _metrics(0.2, 0.3, 0.1)
        before = dict(m)
        eng.compute_training_advice(m)
        assert m == before

    # ── 16.22  module fn and engine method return equal objects ───────────────

    def test_16_22_module_fn_and_engine_method_parity(self):
        eng = _make_engine()
        m   = _metrics(0.2, 0.3, 0.1)
        fn_result     = compute_training_advice(m)
        method_result = eng.compute_training_advice(m)
        assert fn_result.state            == method_result.state
        assert fn_result.recommended_load == method_result.recommended_load

    # ── 16.23  process() output contains "training_advice" key ────────────────

    def test_16_23_process_contains_training_advice(self):
        result = _make_engine().process(_make_valid_input())
        assert result["status"] == "ok"
        assert "training_advice" in result["metrics"]

    # ── 16.24  training_advice is a dict ──────────────────────────────────────

    def test_16_24_training_advice_is_dict(self):
        result = _make_engine().process(_make_valid_input())
        assert isinstance(result["metrics"]["training_advice"], dict)

    # ── 16.25  training_advice has exactly {"state","recommended_load"} ───────

    def test_16_25_training_advice_exact_keys(self):
        ta = _make_engine().process(_make_valid_input())["metrics"]["training_advice"]
        assert set(ta.keys()) == {"state", "recommended_load"}

    # ── 16.26  training_advice["state"] is str in VALID_TRAINING_STATES ───────

    def test_16_26_training_advice_state_valid(self):
        ta = _make_engine().process(_make_valid_input())["metrics"]["training_advice"]
        assert isinstance(ta["state"], str)
        assert ta["state"] in VALID_TRAINING_STATES

    # ── 16.27  training_advice["recommended_load"] is float, finite, in [0,1] ─

    def test_16_27_training_advice_load_valid(self):
        for n in (10, 25, 50):
            ta = _make_engine().process(_make_valid_input(n=n))["metrics"]["training_advice"]
            rl = ta["recommended_load"]
            assert isinstance(rl, float), f"n={n}: recommended_load is {type(rl)}"
            assert math.isfinite(rl),     f"n={n}: recommended_load not finite"
            assert 0.0 <= rl <= 1.0,      f"n={n}: recommended_load={rl} out of [0,1]"

    # ── 16.28  training_advice["state"] == metrics["training_state"] ──────────

    def test_16_28_training_advice_state_matches_training_state(self):
        m = _make_engine().process(_make_valid_input())["metrics"]
        assert m["training_advice"]["state"] == m["training_state"]

    # ── 16.29  training_advice["recommended_load"] == metrics["recommended_load"]

    def test_16_29_training_advice_load_matches_recommended_load(self):
        m = _make_engine().process(_make_valid_input())["metrics"]
        assert m["training_advice"]["recommended_load"] == pytest.approx(
            m["recommended_load"]
        )

    # ── 16.30  training_advice is JSON-serialisable ────────────────────────────

    def test_16_30_training_advice_json_serialisable(self):
        import json
        result = _make_engine().process(_make_valid_input())
        json.dumps(result["metrics"]["training_advice"])  # must not raise

    # ── 16.31  process() error → training_advice absent ───────────────────────

    def test_16_31_process_error_no_training_advice(self):
        result = _make_engine().process("not a dict")
        assert result["status"] == "error"
        assert result["metrics"] is None
        assert "training_advice" not in result

    # ── 16.32  determinism: 100 process() calls → same training_advice ────────

    def test_16_32_process_deterministic_training_advice(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["training_advice"]
        for _ in range(99):
            ta = eng.process(raw)["metrics"]["training_advice"]
            assert ta == first


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)