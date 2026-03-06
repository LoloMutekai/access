"""
A.C.C.E.S.S. — Recovery Prediction Test Suite (Phase 7.9)
tests/test_recovery_prediction.py

Full coverage of biometric/recovery_prediction.py

Coverage map
────────────
SECTION 1  — Constants
  1.1   MIN_RECOVERY_HOURS == 4.0
  1.2   MAX_RECOVERY_HOURS == 72.0
  1.3   W_FATIGUE == 0.35
  1.4   W_INJURY == 0.30
  1.5   W_ANOMALY == 0.15
  1.6   W_BASELINE_DEV == 0.10
  1.7   W_LOAD == 0.10
  1.8   All weights are floats
  1.9   Weights sum to 1.0 (within floating-point tolerance)
  1.10  MIN_RECOVERY_HOURS < MAX_RECOVERY_HOURS

SECTION 2  — RecoveryPrediction dataclass
  2.1   Instantiates with valid hours and score
  2.2   Dataclass is frozen (FrozenInstanceError on reassignment)
  2.3   hours field stored exactly
  2.4   score field stored exactly
  2.5   is_valid() True for well-formed instance
  2.6   is_valid() False when hours is non-finite
  2.7   is_valid() False when hours < MIN_RECOVERY_HOURS
  2.8   is_valid() False when hours > MAX_RECOVERY_HOURS
  2.9   is_valid() False when score is non-finite
  2.10  is_valid() False when score < 0.0
  2.11  is_valid() False when score > 1.0

SECTION 3  — RecoveryPrediction.to_dict()
  3.1   Returns a dict
  3.2   Contains exactly the keys {"recovery_hours", "recovery_score"}
  3.3   recovery_hours matches .hours
  3.4   recovery_score matches .score
  3.5   Returns a fresh dict (mutation does not affect instance)
  3.6   Output is JSON-serialisable
  3.7   All values are native Python floats
  3.8   Repeated calls return equal dicts (idempotent)

SECTION 4  — compute_recovery_prediction() — return type & bounds
  4.1   Returns RecoveryPrediction
  4.2   score ∈ [0.0, 1.0] — all-zero signals
  4.3   score ∈ [0.0, 1.0] — typical signals
  4.4   score ∈ [0.0, 1.0] — all-max signals
  4.5   hours ∈ [MIN, MAX] — all-zero signals
  4.6   hours ∈ [MIN, MAX] — typical signals
  4.7   hours ∈ [MIN, MAX] — all-max signals
  4.8   score is finite
  4.9   hours is finite
  4.10  Returned instance passes is_valid()

SECTION 5  — compute_recovery_prediction() — formula correctness
  5.1   All-zero inputs, load=1.0 → score ≈ 0.0, hours ≈ MIN
  5.2   All-one inputs, load=0.0 → score ≈ 1.0, hours ≈ MAX
  5.3   W_FATIGUE single-channel contribution verifies weight
  5.4   W_INJURY single-channel contribution verifies weight
  5.5   W_ANOMALY single-channel contribution verifies weight
  5.6   W_BASELINE_DEV single-channel contribution verifies weight
  5.7   W_LOAD single-channel (via inverted recommended_load) verifies weight
  5.8   Combined five-channel formula matches manual calculation
  5.9   recommended_load is inverted: high load → low burden
  5.10  recommended_load=0 → full load weight applied
  5.11  recommended_load=1 → zero load weight applied

SECTION 6  — compute_recovery_prediction() — hours model
  6.1   score=0.0 → hours == MIN_RECOVERY_HOURS exactly
  6.2   score=1.0 → hours == MAX_RECOVERY_HOURS (approx, due to IEEE 754)
  6.3   hours is monotone-increasing with score
  6.4   Linear mapping: hours == MIN + score * (MAX − MIN)
  6.5   Midpoint: score≈0.5 → hours near 38.0

SECTION 7  — compute_recovery_prediction() — determinism
  7.1   100 identical calls → identical result
  7.2   Two independent call sequences → identical results
  7.3   Call-order independence (no internal state)

SECTION 8  — compute_recovery_prediction() — monotonicity
  8.1   Higher fatigue_index → higher score and hours
  8.2   Higher injury_risk → higher score and hours
  8.3   Higher anomaly_score → higher score and hours
  8.4   Higher baseline_deviation → higher score and hours
  8.5   Lower recommended_load → higher score and hours (inverted signal)

SECTION 9  — compute_recovery_prediction() — non-finite input guards
  9.1   NaN fatigue_index treated as 0.0
  9.2   Inf injury_risk treated as 0.0
  9.3   NaN anomaly_score treated as 0.0
  9.4   Inf baseline_deviation treated as 0.0
  9.5   NaN recommended_load treated as 0.0 (worst-case inversion)
  9.6   All-NaN inputs → valid prediction returned
  9.7   Mixed NaN/valid → prediction still valid

SECTION 10 — compute_recovery_prediction() — clamp behaviour
  10.1  Raw score exactly 0.0 → clamped score == 0.0
  10.2  Raw score above 1.0 → clamped score == 1.0
  10.3  Hours always ≥ MIN_RECOVERY_HOURS
  10.4  Hours always ≤ MAX_RECOVERY_HOURS

SECTION 11 — Input immutability
  11.1  Passing float arguments — no mutation possible (value semantics)
  11.2  Calling with same argument objects twice → same result (idempotent)

SECTION 12 — JSON serialisability
  12.1  to_dict() passes json.dumps
  12.2  Full round-trip: to_dict → json.dumps → json.loads preserves values
  12.3  recovery_hours is native float (not numpy etc.)
  12.4  recovery_score is native float

SECTION 13 — _clamp() helper
  13.1  Value within range returned unchanged
  13.2  Value below lo → lo
  13.3  Value above hi → hi
  13.4  NaN → lo
  13.5  +Inf → lo
  13.6  Boundary values lo and hi returned exactly

SECTION 14 — _finite_or_zero() helper
  14.1  Finite value returned unchanged
  14.2  NaN → 0.0
  14.3  +Inf → 0.0
  14.4  -Inf → 0.0
  14.5  0.0 returned as 0.0
  14.6  Negative finite value returned unchanged
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.recovery_prediction import (
    MAX_RECOVERY_HOURS,
    MIN_RECOVERY_HOURS,
    W_ANOMALY,
    W_BASELINE_DEV,
    W_FATIGUE,
    W_INJURY,
    W_LOAD,
    RecoveryPrediction,
    _clamp,
    _finite_or_zero,
    compute_recovery_prediction,
)

# Derived constant — used in formula tests.
_HOUR_RANGE = MAX_RECOVERY_HOURS - MIN_RECOVERY_HOURS


# =============================================================================
# HELPERS
# =============================================================================

def _pred(
    fi:  float = 0.0,
    ir:  float = 0.0,
    ans: float = 0.0,
    bd:  float = 0.0,
    rl:  float = 1.0,   # default: full load recommended → zero load burden
) -> RecoveryPrediction:
    """One-liner shorthand for compute_recovery_prediction()."""
    return compute_recovery_prediction(
        fatigue_index=fi,
        injury_risk=ir,
        anomaly_score=ans,
        baseline_deviation=bd,
        recommended_load=rl,
    )


def _manual_score(fi: float, ir: float, ans: float, bd: float, rl: float) -> float:
    """Direct formula for cross-checking."""
    raw = W_FATIGUE * fi + W_INJURY * ir + W_ANOMALY * ans + W_BASELINE_DEV * bd + W_LOAD * (1.0 - rl)
    return max(0.0, min(1.0, raw))


def _manual_hours(score: float) -> float:
    return MIN_RECOVERY_HOURS + score * _HOUR_RANGE


# =============================================================================
# SECTION 1 — Constants
# =============================================================================

class TestConstants:

    def test_1_1_min_recovery_hours(self):
        assert MIN_RECOVERY_HOURS == 4.0

    def test_1_2_max_recovery_hours(self):
        assert MAX_RECOVERY_HOURS == 72.0

    def test_1_3_w_fatigue(self):
        assert W_FATIGUE == pytest.approx(0.35)

    def test_1_4_w_injury(self):
        assert W_INJURY == pytest.approx(0.30)

    def test_1_5_w_anomaly(self):
        assert W_ANOMALY == pytest.approx(0.15)

    def test_1_6_w_baseline_dev(self):
        assert W_BASELINE_DEV == pytest.approx(0.10)

    def test_1_7_w_load(self):
        assert W_LOAD == pytest.approx(0.10)

    def test_1_8_all_weights_are_floats(self):
        for w in (W_FATIGUE, W_INJURY, W_ANOMALY, W_BASELINE_DEV, W_LOAD):
            assert isinstance(w, float)

    def test_1_9_weights_sum_to_one(self):
        total = W_FATIGUE + W_INJURY + W_ANOMALY + W_BASELINE_DEV + W_LOAD
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_1_10_min_less_than_max(self):
        assert MIN_RECOVERY_HOURS < MAX_RECOVERY_HOURS


# =============================================================================
# SECTION 2 — RecoveryPrediction dataclass
# =============================================================================

class TestRecoveryPredictionDataclass:

    def test_2_1_instantiation(self):
        rp = RecoveryPrediction(hours=24.0, score=0.5)
        assert isinstance(rp, RecoveryPrediction)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        rp = RecoveryPrediction(hours=24.0, score=0.5)
        with pytest.raises(FrozenInstanceError):
            rp.hours = 99.0  # type: ignore[misc]

    def test_2_3_hours_stored_exactly(self):
        rp = RecoveryPrediction(hours=36.5, score=0.5)
        assert rp.hours == 36.5

    def test_2_4_score_stored_exactly(self):
        rp = RecoveryPrediction(hours=24.0, score=0.478)
        assert rp.score == 0.478

    def test_2_5_is_valid_true(self):
        rp = RecoveryPrediction(hours=24.0, score=0.5)
        assert rp.is_valid() is True

    def test_2_6_is_valid_false_nonfinite_hours(self):
        rp = RecoveryPrediction(hours=float("inf"), score=0.5)
        assert rp.is_valid() is False

    def test_2_7_is_valid_false_hours_below_min(self):
        rp = RecoveryPrediction(hours=3.9, score=0.0)
        assert rp.is_valid() is False

    def test_2_8_is_valid_false_hours_above_max(self):
        rp = RecoveryPrediction(hours=72.1, score=1.0)
        assert rp.is_valid() is False

    def test_2_9_is_valid_false_nonfinite_score(self):
        rp = RecoveryPrediction(hours=24.0, score=float("nan"))
        assert rp.is_valid() is False

    def test_2_10_is_valid_false_score_negative(self):
        rp = RecoveryPrediction(hours=4.0, score=-0.001)
        assert rp.is_valid() is False

    def test_2_11_is_valid_false_score_above_one(self):
        rp = RecoveryPrediction(hours=72.0, score=1.001)
        assert rp.is_valid() is False


# =============================================================================
# SECTION 3 — RecoveryPrediction.to_dict()
# =============================================================================

class TestToDict:

    def test_3_1_returns_dict(self):
        rp = _pred(fi=0.5, rl=0.5)
        assert isinstance(rp.to_dict(), dict)

    def test_3_2_exact_keys(self):
        rp = _pred()
        assert set(rp.to_dict().keys()) == {"recovery_hours", "recovery_score"}

    def test_3_3_recovery_hours_matches_field(self):
        rp = _pred(fi=0.4, ir=0.3)
        assert rp.to_dict()["recovery_hours"] == rp.hours

    def test_3_4_recovery_score_matches_field(self):
        rp = _pred(fi=0.4, ir=0.3)
        assert rp.to_dict()["recovery_score"] == rp.score

    def test_3_5_fresh_dict_no_leakback(self):
        rp = _pred(fi=0.5)
        d  = rp.to_dict()
        d["recovery_hours"] = 999.0
        assert rp.hours != 999.0

    def test_3_6_json_serialisable(self):
        json.dumps(_pred(fi=0.5).to_dict())   # must not raise

    def test_3_7_all_values_native_float(self):
        d = _pred(fi=0.5).to_dict()
        for val in d.values():
            assert isinstance(val, float), f"got {type(val)}"

    def test_3_8_idempotent(self):
        rp = _pred(fi=0.6, ir=0.4)
        assert rp.to_dict() == rp.to_dict()


# =============================================================================
# SECTION 4 — compute_recovery_prediction() — return type & bounds
# =============================================================================

class TestReturnTypeBounds:

    def test_4_1_returns_recovery_prediction(self):
        assert isinstance(_pred(), RecoveryPrediction)

    def test_4_2_score_bounded_all_zero(self):
        s = _pred().score
        assert 0.0 <= s <= 1.0

    def test_4_3_score_bounded_typical(self):
        s = _pred(fi=0.5, ir=0.3, ans=0.2, bd=0.1, rl=0.6).score
        assert 0.0 <= s <= 1.0

    def test_4_4_score_bounded_all_max(self):
        s = _pred(fi=1.0, ir=1.0, ans=1.0, bd=1.0, rl=0.0).score
        assert 0.0 <= s <= 1.0

    def test_4_5_hours_bounded_all_zero(self):
        h = _pred().hours
        assert MIN_RECOVERY_HOURS <= h <= MAX_RECOVERY_HOURS

    def test_4_6_hours_bounded_typical(self):
        h = _pred(fi=0.5, ir=0.3, ans=0.2, bd=0.1, rl=0.6).hours
        assert MIN_RECOVERY_HOURS <= h <= MAX_RECOVERY_HOURS

    def test_4_7_hours_bounded_all_max(self):
        h = _pred(fi=1.0, ir=1.0, ans=1.0, bd=1.0, rl=0.0).hours
        assert MIN_RECOVERY_HOURS <= h <= MAX_RECOVERY_HOURS

    def test_4_8_score_is_finite(self):
        assert math.isfinite(_pred(fi=0.5).score)

    def test_4_9_hours_is_finite(self):
        assert math.isfinite(_pred(fi=0.5).hours)

    def test_4_10_returned_instance_is_valid(self):
        for fi in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert _pred(fi=fi).is_valid(), f"is_valid failed for fi={fi}"


# =============================================================================
# SECTION 5 — compute_recovery_prediction() — formula correctness
# =============================================================================

class TestFormulaCorrectness:

    def test_5_1_all_zero_inputs_min_hours(self):
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=1.0)
        assert p.score == pytest.approx(0.0, abs=1e-12)
        assert p.hours == pytest.approx(MIN_RECOVERY_HOURS, abs=1e-9)

    def test_5_2_all_max_inputs_max_hours(self):
        p = _pred(fi=1.0, ir=1.0, ans=1.0, bd=1.0, rl=0.0)
        assert p.score == pytest.approx(1.0, abs=1e-9)
        assert p.hours == pytest.approx(MAX_RECOVERY_HOURS, abs=1e-6)

    def test_5_3_w_fatigue_single_channel(self):
        """Only fatigue_index=0.8, all others at neutral → contribution=W_FATIGUE*0.8."""
        p = _pred(fi=0.8, ir=0.0, ans=0.0, bd=0.0, rl=1.0)
        expected_score = W_FATIGUE * 0.8
        assert p.score == pytest.approx(expected_score, abs=1e-12)

    def test_5_4_w_injury_single_channel(self):
        p = _pred(fi=0.0, ir=0.6, ans=0.0, bd=0.0, rl=1.0)
        assert p.score == pytest.approx(W_INJURY * 0.6, abs=1e-12)

    def test_5_5_w_anomaly_single_channel(self):
        p = _pred(fi=0.0, ir=0.0, ans=0.7, bd=0.0, rl=1.0)
        assert p.score == pytest.approx(W_ANOMALY * 0.7, abs=1e-12)

    def test_5_6_w_baseline_dev_single_channel(self):
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.5, rl=1.0)
        assert p.score == pytest.approx(W_BASELINE_DEV * 0.5, abs=1e-12)

    def test_5_7_w_load_single_channel(self):
        """recommended_load=0 → load_burden = W_LOAD * (1 - 0) = W_LOAD."""
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=0.0)
        assert p.score == pytest.approx(W_LOAD * 1.0, abs=1e-12)

    def test_5_8_combined_five_channels_manual(self):
        fi, ir, ans, bd, rl = 0.6, 0.4, 0.3, 0.2, 0.5
        p = _pred(fi=fi, ir=ir, ans=ans, bd=bd, rl=rl)
        expected_score = _manual_score(fi, ir, ans, bd, rl)
        expected_hours = _manual_hours(expected_score)
        assert p.score == pytest.approx(expected_score, abs=1e-12)
        assert p.hours == pytest.approx(expected_hours, abs=1e-9)

    def test_5_9_recommended_load_is_inverted(self):
        """High recommended_load → low burden (athlete is fine to train)."""
        p_high_load = _pred(fi=0.3, ir=0.2, ans=0.1, bd=0.1, rl=0.9)
        p_low_load  = _pred(fi=0.3, ir=0.2, ans=0.1, bd=0.1, rl=0.1)
        assert p_high_load.score < p_low_load.score
        assert p_high_load.hours < p_low_load.hours

    def test_5_10_recommended_load_zero_full_burden(self):
        """rl=0 → 1−rl=1, full W_LOAD weight applied."""
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=0.0)
        assert p.score == pytest.approx(W_LOAD, abs=1e-12)

    def test_5_11_recommended_load_one_zero_burden(self):
        """rl=1 → 1−rl=0, zero W_LOAD contribution."""
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=1.0)
        assert p.score == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# SECTION 6 — compute_recovery_prediction() — hours model
# =============================================================================

class TestHoursModel:

    def test_6_1_score_zero_gives_min_hours(self):
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=1.0)
        assert p.hours == pytest.approx(MIN_RECOVERY_HOURS, abs=1e-9)

    def test_6_2_score_one_gives_max_hours(self):
        p = _pred(fi=1.0, ir=1.0, ans=1.0, bd=1.0, rl=0.0)
        assert p.hours == pytest.approx(MAX_RECOVERY_HOURS, abs=1e-6)

    def test_6_3_hours_monotone_with_score(self):
        """Increasing fatigue alone must increase hours."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        hours  = [_pred(fi=s, rl=1.0).hours for s in scores]
        assert all(hours[i] < hours[i+1] for i in range(len(hours)-1))

    def test_6_4_linear_mapping_exact(self):
        fi, ir, ans, bd, rl = 0.4, 0.3, 0.2, 0.1, 0.5
        p = _pred(fi=fi, ir=ir, ans=ans, bd=bd, rl=rl)
        expected = MIN_RECOVERY_HOURS + p.score * _HOUR_RANGE
        assert p.hours == pytest.approx(expected, abs=1e-9)

    def test_6_5_midpoint_score_near_38_hours(self):
        """score=0.5 → hours = 4 + 0.5*68 = 38.0"""
        # Build inputs whose manual score ≈ 0.5
        # 0.35*fi = 0.5 → fi = 0.5/0.35 ≈ 1.43 → clamp to 1 → overshoots
        # Use fi=0.0 and let other channels carry the load:
        # W_INJURY*0.5 + W_ANOMALY*0.5 + W_BASELINE_DEV*0.5 + W_LOAD*0.5
        #   = 0.5*(0.30+0.15+0.10+0.10) = 0.5*0.65 = 0.325  (not 0.5)
        # Simple: fi=0.5, others neutral
        #   score = 0.35*0.5 = 0.175 → hours ≈ 15.9  (not 38)
        # For score exactly 0.5 we need fi≈1.43 → use all channels:
        # fi=0.5, ir=0.5, ans=0.5, bd=0.5, rl=0.5
        # score = 0.35*0.5+0.30*0.5+0.15*0.5+0.10*0.5+0.10*0.5 = 0.5*1.0 = 0.5
        p = _pred(fi=0.5, ir=0.5, ans=0.5, bd=0.5, rl=0.5)
        assert p.hours == pytest.approx(38.0, abs=1e-6)


# =============================================================================
# SECTION 7 — Determinism
# =============================================================================

class TestDeterminism:

    def test_7_1_100_identical_calls(self):
        first = _pred(fi=0.5, ir=0.3, ans=0.2, bd=0.1, rl=0.6)
        for _ in range(99):
            assert _pred(fi=0.5, ir=0.3, ans=0.2, bd=0.1, rl=0.6) == first

    def test_7_2_two_independent_sequences(self):
        def _run() -> list[RecoveryPrediction]:
            return [
                _pred(fi=0.1, ir=0.0, ans=0.0, bd=0.0, rl=0.9),
                _pred(fi=0.5, ir=0.4, ans=0.3, bd=0.2, rl=0.5),
                _pred(fi=0.9, ir=0.8, ans=0.7, bd=0.6, rl=0.1),
            ]
        seq1 = _run()
        seq2 = _run()
        for p1, p2 in zip(seq1, seq2):
            assert p1.score == p2.score
            assert p1.hours == p2.hours

    def test_7_3_call_order_independence(self):
        """Result for a given input must not depend on what was called before."""
        target  = _pred(fi=0.4, ir=0.3, ans=0.2, bd=0.1, rl=0.5)
        # call with different args first
        _pred(fi=0.9, ir=0.8, ans=0.7, bd=0.6, rl=0.1)
        after   = _pred(fi=0.4, ir=0.3, ans=0.2, bd=0.1, rl=0.5)
        assert target.score == after.score
        assert target.hours == after.hours


# =============================================================================
# SECTION 8 — Monotonicity
# =============================================================================

class TestMonotonicity:

    def test_8_1_higher_fatigue_raises_score(self):
        assert _pred(fi=0.3).score < _pred(fi=0.7).score

    def test_8_2_higher_injury_raises_score(self):
        assert _pred(ir=0.2).score < _pred(ir=0.6).score

    def test_8_3_higher_anomaly_raises_score(self):
        assert _pred(ans=0.1).score < _pred(ans=0.5).score

    def test_8_4_higher_baseline_dev_raises_score(self):
        assert _pred(bd=0.1).score < _pred(bd=0.6).score

    def test_8_5_lower_load_raises_score(self):
        """recommended_load is inverted — lower load → higher burden."""
        assert _pred(rl=0.9).score < _pred(rl=0.1).score


# =============================================================================
# SECTION 9 — Non-finite input guards
# =============================================================================

class TestNonFiniteInputGuards:

    def test_9_1_nan_fatigue_treated_as_zero(self):
        p_nan  = _pred(fi=float("nan"))
        p_zero = _pred(fi=0.0)
        assert p_nan.score == pytest.approx(p_zero.score, abs=1e-12)

    def test_9_2_inf_injury_treated_as_zero(self):
        p_inf  = _pred(ir=float("inf"))
        p_zero = _pred(ir=0.0)
        assert p_inf.score == pytest.approx(p_zero.score, abs=1e-12)

    def test_9_3_nan_anomaly_treated_as_zero(self):
        p_nan  = _pred(ans=float("nan"))
        p_zero = _pred(ans=0.0)
        assert p_nan.score == pytest.approx(p_zero.score, abs=1e-12)

    def test_9_4_inf_baseline_dev_treated_as_zero(self):
        p_inf  = _pred(bd=float("inf"))
        p_zero = _pred(bd=0.0)
        assert p_inf.score == pytest.approx(p_zero.score, abs=1e-12)

    def test_9_5_nan_recommended_load_treated_as_zero(self):
        """NaN load → treated as 0 → (1−0) = 1 applied."""
        p_nan  = _pred(rl=float("nan"))
        p_zero = _pred(rl=0.0)
        assert p_nan.score == pytest.approx(p_zero.score, abs=1e-12)

    def test_9_6_all_nan_returns_valid_prediction(self):
        p = compute_recovery_prediction(
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"),
        )
        assert p.is_valid()

    def test_9_7_mixed_nan_valid_stays_valid(self):
        p = compute_recovery_prediction(
            0.5, float("nan"), 0.3, float("inf"), 0.6,
        )
        assert p.is_valid()


# =============================================================================
# SECTION 10 — Clamp behaviour
# =============================================================================

class TestClampBehaviour:

    def test_10_1_zero_score_stays_zero(self):
        p = _pred(fi=0.0, ir=0.0, ans=0.0, bd=0.0, rl=1.0)
        assert p.score == 0.0

    def test_10_2_raw_score_above_one_clamped(self):
        """All inputs at maximum → raw score ≈ 1.0; clamp keeps it ≤ 1.0."""
        p = _pred(fi=1.0, ir=1.0, ans=1.0, bd=1.0, rl=0.0)
        assert p.score <= 1.0

    def test_10_3_hours_always_gte_min(self):
        for fi in (0.0, 0.5, 1.0):
            assert _pred(fi=fi).hours >= MIN_RECOVERY_HOURS

    def test_10_4_hours_always_lte_max(self):
        for fi in (0.0, 0.5, 1.0):
            assert _pred(fi=fi).hours <= MAX_RECOVERY_HOURS


# =============================================================================
# SECTION 11 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_11_1_float_arguments_value_semantics(self):
        """Python floats are immutable by definition; confirm no side-effects."""
        fi, ir, ans, bd, rl = 0.4, 0.3, 0.2, 0.1, 0.6
        snapshot = (fi, ir, ans, bd, rl)
        compute_recovery_prediction(fi, ir, ans, bd, rl)
        assert (fi, ir, ans, bd, rl) == snapshot

    def test_11_2_same_args_twice_identical(self):
        """Calling twice with the same arguments must give identical results."""
        args = (0.5, 0.4, 0.3, 0.2, 0.5)
        p1 = compute_recovery_prediction(*args)
        p2 = compute_recovery_prediction(*args)
        assert p1.score == p2.score
        assert p1.hours == p2.hours


# =============================================================================
# SECTION 12 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_12_1_to_dict_json_serialisable(self):
        json.dumps(_pred(fi=0.5).to_dict())   # must not raise

    def test_12_2_full_round_trip(self):
        p    = _pred(fi=0.6, ir=0.4, ans=0.3, bd=0.2, rl=0.5)
        d    = p.to_dict()
        back = json.loads(json.dumps(d))
        assert abs(back["recovery_hours"] - p.hours) < 1e-9
        assert abs(back["recovery_score"] - p.score) < 1e-9

    def test_12_3_recovery_hours_native_float(self):
        assert isinstance(_pred().to_dict()["recovery_hours"], float)

    def test_12_4_recovery_score_native_float(self):
        assert isinstance(_pred().to_dict()["recovery_score"], float)


# =============================================================================
# SECTION 13 — _clamp() helper
# =============================================================================

class TestClampHelper:

    def test_13_1_within_range_unchanged(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_13_2_below_lo_returns_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_13_3_above_hi_returns_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_13_4_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_13_5_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_13_6_boundary_values_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# =============================================================================
# SECTION 14 — _finite_or_zero() helper
# =============================================================================

class TestFiniteOrZero:

    def test_14_1_finite_value_unchanged(self):
        assert _finite_or_zero(0.42) == pytest.approx(0.42)

    def test_14_2_nan_returns_zero(self):
        assert _finite_or_zero(float("nan")) == 0.0

    def test_14_3_pos_inf_returns_zero(self):
        assert _finite_or_zero(float("inf")) == 0.0

    def test_14_4_neg_inf_returns_zero(self):
        assert _finite_or_zero(float("-inf")) == 0.0

    def test_14_5_zero_returns_zero(self):
        assert _finite_or_zero(0.0) == 0.0

    def test_14_6_negative_finite_unchanged(self):
        assert _finite_or_zero(-0.5) == pytest.approx(-0.5)


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