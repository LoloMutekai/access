"""
A.C.C.E.S.S. — Baseline Deviation Test Suite (Phase 7.8B)
tests/test_baseline_deviation.py

Full coverage of biometric/baseline_deviation.py

Coverage map
────────────
SECTION 1  — Constants
  1.1   WEIGHT_HR  == 0.35
  1.2   WEIGHT_HRV == 0.25
  1.3   WEIGHT_LOAD == 0.20
  1.4   WEIGHT_FATIGUE == 0.20
  1.5   Weights sum to exactly 1.0
  1.6   All weight constants are floats

SECTION 2  — compute_baseline_deviation() — return type & bounds
  2.1   Returns float
  2.2   Score ∈ [0.0, 1.0] — identical inputs
  2.3   Score ∈ [0.0, 1.0] — small deviations
  2.4   Score ∈ [0.0, 1.0] — large deviations
  2.5   Score ∈ [0.0, 1.0] — extreme deviations (ensures clamp)
  2.6   Score is finite
  2.7   Score ≥ 0.0 always
  2.8   Score ≤ 1.0 always

SECTION 3  — compute_baseline_deviation() — formula correctness
  3.1   Identical metrics + fatigue → score == 0.0
  3.2   dev_hr formula: single-channel change verifies weight contribution
  3.3   dev_hrv formula: single-channel change verifies weight contribution
  3.4   dev_load formula: single-channel change verifies weight contribution
  3.5   dev_fatigue formula: single-channel change verifies weight contribution
  3.6   Combined formula matches manual calculation (all four channels differ)
  3.7   Denominator guard: baseline channel == 0 uses max(0, 1) = 1
  3.8   Denominator guard: baseline channel < 1 uses max(val, 1) = 1
  3.9   dev_fatigue has no denominator (direct absolute difference)

SECTION 4  — compute_baseline_deviation() — determinism
  4.1   100 calls with identical inputs → identical score
  4.2   Two fresh call sequences with same arguments → identical results
  4.3   Result independent of call history (no internal state)

SECTION 5  — compute_baseline_deviation() — monotonicity
  5.1   Larger HR gap → higher score (other channels fixed)
  5.2   Larger HRV gap → higher score (other channels fixed)
  5.3   Larger load gap → higher score (other channels fixed)
  5.4   Larger fatigue gap → higher score (other channels fixed)

SECTION 6  — compute_baseline_deviation() — constant-signal near-zero
  6.1   Metrics exactly at baseline → score == 0.0
  6.2   Metrics within 1% of baseline → score < 0.01
  6.3   Many small perturbations average near zero

SECTION 7  — compute_baseline_deviation() — clamp behaviour
  7.1   Raw score > 1.0 is clamped to 1.0
  7.2   Extreme HR deviation alone can push score to 1.0
  7.3   Clamped result is still a valid float

SECTION 8  — compute_baseline_deviation() — input immutability
  8.1   CoreMetrics object not mutated
  8.2   FatigueResult object not mutated
  8.3   BaselineState object not mutated

SECTION 9  — compute_baseline_deviation() — JSON serialisability
  9.1   Return value passes json.dumps
  9.2   Return value has native Python float type (not numpy etc.)

SECTION 10 — deviation_components() — structure
  10.1  Returns a dict
  10.2  Contains exactly the five expected keys
  10.3  "score" matches compute_baseline_deviation() exactly
  10.4  All values are finite floats
  10.5  All values are JSON-serialisable
  10.6  Component scores are non-negative
  10.7  dev_fatigue ∈ [0, 1] (both sides are bounded)

SECTION 11 — deviation_components() — formula parity
  11.1  dev_hr   equals _safe_abs_norm(metrics.mean_hr,   baseline.mean_hr)
  11.2  dev_hrv  equals _safe_abs_norm(metrics.mean_hrv,  baseline.mean_hrv)
  11.3  dev_load equals _safe_abs_norm(metrics.load_mean, baseline.mean_load)
  11.4  dev_fatigue equals _safe_abs_diff(fatigue.value,  baseline.baseline_fatigue)

SECTION 12 — _safe_abs_norm() helper
  12.1  Normal inputs: result == |current − baseline| / baseline (when baseline ≥ 1)
  12.2  baseline == 0 → denominator becomes 1, result == |current|
  12.3  baseline == 0.5 → denominator becomes 1 (max guard)
  12.4  Non-finite current → returns 0.0
  12.5  Non-finite baseline → returns 0.0
  12.6  Result is always ≥ 0.0

SECTION 13 — _safe_abs_diff() helper
  13.1  Result == |a − b| for normal inputs
  13.2  Non-finite a → returns 0.0
  13.3  Non-finite b → returns 0.0
  13.4  Result is always ≥ 0.0
  13.5  Symmetric: _safe_abs_diff(a, b) == _safe_abs_diff(b, a)

SECTION 14 — _clamp() helper
  14.1  Value within range returned unchanged
  14.2  Value below lo → lo
  14.3  Value above hi → hi
  14.4  Non-finite value → lo
  14.5  Boundary values lo and hi returned exactly
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.baseline_deviation import (
    WEIGHT_FATIGUE,
    WEIGHT_HR,
    WEIGHT_HRV,
    WEIGHT_LOAD,
    _clamp,
    _safe_abs_diff,
    _safe_abs_norm,
    compute_baseline_deviation,
    deviation_components,
)
from biometric.baseline_engine import BaselineState
from biometric.biometric_engine import CoreMetrics, FatigueResult


# =============================================================================
# HELPERS
# =============================================================================

def _baseline(
    hr:      float = 70.0,
    hrv:     float = 40.0,
    load:    float = 100.0,
    fatigue: float = 0.30,
    n:       int   = 10,
) -> BaselineState:
    return BaselineState(
        mean_hr=hr, mean_hrv=hrv, mean_load=load,
        baseline_fatigue=fatigue, sample_count=n,
    )


def _metrics(
    hr:   float = 70.0,
    hrv:  float = 40.0,
    load: float = 100.0,
) -> CoreMetrics:
    return CoreMetrics(
        mean_hr=hr, mean_hrv=hrv, load_mean=load, hr_std=5.0, hrv_std=3.0,
    )


def _fatigue(value: float = 0.30) -> FatigueResult:
    return FatigueResult(value=value, raw=value * 2.0)


def _score(
    m_hr:  float = 70.0,  m_hrv:  float = 40.0,  m_load: float = 100.0,
    fat:   float = 0.30,
    b_hr:  float = 70.0,  b_hrv:  float = 40.0,  b_load: float = 100.0,
    b_fat: float = 0.30,
) -> float:
    """One-liner shorthand for the full call."""
    return compute_baseline_deviation(
        _metrics(m_hr, m_hrv, m_load),
        _fatigue(fat),
        _baseline(b_hr, b_hrv, b_load, b_fat),
    )


def _manual(
    m_hr:  float, m_hrv:  float, m_load: float, fat:   float,
    b_hr:  float, b_hrv:  float, b_load: float, b_fat: float,
) -> float:
    """Direct formula evaluation for cross-checking."""
    dev_hr      = abs(m_hr   - b_hr)   / max(b_hr,   1.0)
    dev_hrv     = abs(m_hrv  - b_hrv)  / max(b_hrv,  1.0)
    dev_load    = abs(m_load - b_load) / max(b_load,  1.0)
    dev_fatigue = abs(fat - b_fat)
    raw = 0.35 * dev_hr + 0.25 * dev_hrv + 0.20 * dev_load + 0.20 * dev_fatigue
    return max(0.0, min(1.0, raw))


# =============================================================================
# SECTION 1 — Constants
# =============================================================================

class TestConstants:

    def test_1_1_weight_hr(self):
        assert WEIGHT_HR == pytest.approx(0.35)

    def test_1_2_weight_hrv(self):
        assert WEIGHT_HRV == pytest.approx(0.25)

    def test_1_3_weight_load(self):
        assert WEIGHT_LOAD == pytest.approx(0.20)

    def test_1_4_weight_fatigue(self):
        assert WEIGHT_FATIGUE == pytest.approx(0.20)

    def test_1_5_weights_sum_to_one(self):
        total = WEIGHT_HR + WEIGHT_HRV + WEIGHT_LOAD + WEIGHT_FATIGUE
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_1_6_all_weights_are_floats(self):
        for w in (WEIGHT_HR, WEIGHT_HRV, WEIGHT_LOAD, WEIGHT_FATIGUE):
            assert isinstance(w, float)


# =============================================================================
# SECTION 2 — Return type & bounds
# =============================================================================

class TestReturnTypeBounds:

    def test_2_1_returns_float(self):
        assert isinstance(_score(), float)

    def test_2_2_bounds_identical_inputs(self):
        s = _score()
        assert 0.0 <= s <= 1.0

    def test_2_3_bounds_small_deviations(self):
        s = _score(m_hr=72.0, m_hrv=41.0, m_load=105.0, fat=0.32)
        assert 0.0 <= s <= 1.0

    def test_2_4_bounds_large_deviations(self):
        s = _score(m_hr=100.0, m_hrv=70.0, m_load=300.0, fat=0.9)
        assert 0.0 <= s <= 1.0

    def test_2_5_bounds_extreme_deviations(self):
        """Extreme values may push raw score above 1 — clamp must fire."""
        s = _score(m_hr=500.0, m_hrv=500.0, m_load=9000.0, fat=1.0,
                   b_hr=70.0,  b_hrv=40.0,  b_load=100.0,  b_fat=0.0)
        assert 0.0 <= s <= 1.0

    def test_2_6_score_is_finite(self):
        assert math.isfinite(_score())

    def test_2_7_score_gte_zero(self):
        for m_hr in (20.0, 70.0, 180.0):
            assert _score(m_hr=m_hr) >= 0.0

    def test_2_8_score_lte_one(self):
        for m_hr in (20.0, 300.0, 500.0):
            assert _score(m_hr=m_hr) <= 1.0


# =============================================================================
# SECTION 3 — Formula correctness
# =============================================================================

class TestFormulaCorrectness:

    def test_3_1_identical_inputs_zero(self):
        assert _score() == 0.0

    def test_3_2_dev_hr_single_channel(self):
        """Only HR differs — contribution is WEIGHT_HR * dev_hr."""
        m_hr, b_hr = 80.0, 70.0
        expected = WEIGHT_HR * abs(m_hr - b_hr) / max(b_hr, 1.0)
        assert _score(m_hr=m_hr) == pytest.approx(expected, abs=1e-12)

    def test_3_3_dev_hrv_single_channel(self):
        m_hrv, b_hrv = 55.0, 40.0
        expected = WEIGHT_HRV * abs(m_hrv - b_hrv) / max(b_hrv, 1.0)
        assert _score(m_hrv=m_hrv) == pytest.approx(expected, abs=1e-12)

    def test_3_4_dev_load_single_channel(self):
        m_load, b_load = 150.0, 100.0
        expected = WEIGHT_LOAD * abs(m_load - b_load) / max(b_load, 1.0)
        assert _score(m_load=m_load) == pytest.approx(expected, abs=1e-12)

    def test_3_5_dev_fatigue_single_channel(self):
        fat, b_fat = 0.7, 0.3
        expected = WEIGHT_FATIGUE * abs(fat - b_fat)
        assert _score(fat=fat, b_fat=b_fat) == pytest.approx(expected, abs=1e-12)

    def test_3_6_combined_all_channels(self):
        kw = dict(m_hr=80.0, m_hrv=50.0, m_load=150.0, fat=0.6,
                  b_hr=70.0, b_hrv=40.0, b_load=100.0, b_fat=0.3)
        assert _score(**kw) == pytest.approx(_manual(**kw), abs=1e-12)

    def test_3_7_denominator_guard_baseline_zero(self):
        """baseline_hr == 0 → denominator becomes max(0, 1) = 1.

        We use hr=1.2 so the raw score (0.35 × 1.2 = 0.42) stays below 1.0
        and the clamp does not obscure the denominator-guard behaviour.
        """
        obs_hr   = 1.2
        base_hr  = 0.0
        expected = WEIGHT_HR * abs(obs_hr - base_hr) / max(base_hr, 1.0)
        got = compute_baseline_deviation(
            _metrics(hr=obs_hr),
            _fatigue(0.0),
            _baseline(hr=base_hr, fatigue=0.0),
        )
        assert got == pytest.approx(expected, abs=1e-12)

    def test_3_8_denominator_guard_baseline_below_one(self):
        """baseline_hr == 0.5 → denominator becomes max(0.5, 1) = 1."""
        expected = WEIGHT_HR * abs(2.0 - 0.5) / 1.0
        got = compute_baseline_deviation(
            _metrics(hr=2.0),
            _fatigue(0.0),
            _baseline(hr=0.5, fatigue=0.0),
        )
        assert got == pytest.approx(expected, abs=1e-12)

    def test_3_9_dev_fatigue_no_denominator(self):
        """dev_fatigue = |fatigue.value − baseline.baseline_fatigue| exactly."""
        fat, b_fat = 0.8, 0.1
        expected = WEIGHT_FATIGUE * abs(fat - b_fat)
        got = compute_baseline_deviation(
            _metrics(),      # all channels identical to baseline
            _fatigue(fat),
            _baseline(fatigue=b_fat),
        )
        assert got == pytest.approx(expected, abs=1e-12)


# =============================================================================
# SECTION 4 — Determinism
# =============================================================================

class TestDeterminism:

    def test_4_1_100_calls_identical(self):
        m, f, b = _metrics(hr=75.0), _fatigue(0.4), _baseline(hr=70.0)
        first = compute_baseline_deviation(m, f, b)
        for _ in range(99):
            assert compute_baseline_deviation(m, f, b) == first

    def test_4_2_two_fresh_sequences_identical(self):
        def _run() -> list[float]:
            pairs = [
                (_metrics(70.0, 40.0, 100.0), _fatigue(0.3), _baseline()),
                (_metrics(80.0, 50.0, 120.0), _fatigue(0.5), _baseline(hr=72.0)),
                (_metrics(65.0, 38.0,  90.0), _fatigue(0.2), _baseline(hrv=42.0)),
            ]
            return [compute_baseline_deviation(m, f, b) for m, f, b in pairs]
        assert _run() == _run()

    def test_4_3_no_internal_state(self):
        """Call order must not affect individual results."""
        m1, f1, b1 = _metrics(80.0), _fatigue(0.5), _baseline()
        m2, f2, b2 = _metrics(65.0), _fatigue(0.2), _baseline(hrv=50.0)
        s1_first  = compute_baseline_deviation(m1, f1, b1)
        compute_baseline_deviation(m2, f2, b2)
        s1_second = compute_baseline_deviation(m1, f1, b1)
        assert s1_first == s1_second


# =============================================================================
# SECTION 5 — Monotonicity
# =============================================================================

class TestMonotonicity:

    def test_5_1_larger_hr_gap_raises_score(self):
        s1 = _score(m_hr=75.0)
        s2 = _score(m_hr=85.0)
        assert s2 > s1

    def test_5_2_larger_hrv_gap_raises_score(self):
        s1 = _score(m_hrv=45.0)
        s2 = _score(m_hrv=55.0)
        assert s2 > s1

    def test_5_3_larger_load_gap_raises_score(self):
        s1 = _score(m_load=120.0)
        s2 = _score(m_load=150.0)
        assert s2 > s1

    def test_5_4_larger_fatigue_gap_raises_score(self):
        s1 = _score(fat=0.4)
        s2 = _score(fat=0.6)
        assert s2 > s1


# =============================================================================
# SECTION 6 — Constant-signal near-zero
# =============================================================================

class TestConstantSignalNearZero:

    def test_6_1_exact_match_zero(self):
        assert compute_baseline_deviation(
            _metrics(70.0, 40.0, 100.0),
            _fatigue(0.3),
            _baseline(70.0, 40.0, 100.0, 0.3),
        ) == 0.0

    def test_6_2_one_percent_deviation_below_threshold(self):
        """1 % perturbation across all channels → score well below 0.01."""
        got = _score(
            m_hr=70.7, m_hrv=40.4, m_load=101.0, fat=0.303,
            b_hr=70.0, b_hrv=40.0, b_load=100.0, b_fat=0.300,
        )
        assert got < 0.01

    def test_6_3_repeated_identical_signals_stay_zero(self):
        m, f, b = _metrics(), _fatigue(), _baseline()
        for _ in range(50):
            assert compute_baseline_deviation(m, f, b) == 0.0


# =============================================================================
# SECTION 7 — Clamp behaviour
# =============================================================================

class TestClampBehaviour:

    def test_7_1_raw_above_one_clamped(self):
        """Extreme HR (10× baseline) should saturate the score at 1.0."""
        s = compute_baseline_deviation(
            _metrics(hr=700.0, hrv=400.0, load=1000.0),
            _fatigue(1.0),
            _baseline(hr=70.0, hrv=40.0, load=100.0, fatigue=0.0),
        )
        assert s == 1.0

    def test_7_2_extreme_hr_alone_can_reach_one(self):
        """HR weight = 0.35; dev_hr = (10000 - 70)/70 ≈ 142, so 0.35×142 >> 1."""
        s = compute_baseline_deviation(
            _metrics(hr=10_000.0),
            _fatigue(0.3),
            _baseline(),
        )
        assert s == 1.0

    def test_7_3_clamped_result_is_finite(self):
        s = compute_baseline_deviation(
            _metrics(hr=99_999.0),
            _fatigue(1.0),
            _baseline(),
        )
        assert math.isfinite(s)


# =============================================================================
# SECTION 8 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_8_1_core_metrics_not_mutated(self):
        import copy
        m = _metrics(hr=75.0, hrv=42.0, load=110.0)
        snap = copy.deepcopy(m)
        compute_baseline_deviation(m, _fatigue(), _baseline())
        assert m.mean_hr   == snap.mean_hr
        assert m.mean_hrv  == snap.mean_hrv
        assert m.load_mean == snap.load_mean

    def test_8_2_fatigue_result_not_mutated(self):
        import copy
        f    = _fatigue(0.55)
        snap = copy.deepcopy(f)
        compute_baseline_deviation(_metrics(), f, _baseline())
        assert f.value == snap.value
        assert f.raw   == snap.raw

    def test_8_3_baseline_state_not_mutated(self):
        import copy
        b    = _baseline(hr=72.0, hrv=38.0, load=95.0, fatigue=0.25)
        snap = copy.deepcopy(b)
        compute_baseline_deviation(_metrics(), _fatigue(), b)
        assert b.mean_hr          == snap.mean_hr
        assert b.mean_hrv         == snap.mean_hrv
        assert b.mean_load        == snap.mean_load
        assert b.baseline_fatigue == snap.baseline_fatigue
        assert b.sample_count     == snap.sample_count


# =============================================================================
# SECTION 9 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_9_1_score_json_serialisable(self):
        s = compute_baseline_deviation(_metrics(), _fatigue(), _baseline())
        json.dumps({"score": s})   # must not raise

    def test_9_2_native_python_float(self):
        s = compute_baseline_deviation(_metrics(), _fatigue(), _baseline())
        assert isinstance(s, float)


# =============================================================================
# SECTION 10 — deviation_components() structure
# =============================================================================

class TestDeviationComponentsStructure:

    def test_10_1_returns_dict(self):
        m, f, b = _metrics(), _fatigue(), _baseline()
        assert isinstance(deviation_components(m, f, b), dict)

    def test_10_2_exact_five_keys(self):
        m, f, b = _metrics(), _fatigue(), _baseline()
        assert set(deviation_components(m, f, b).keys()) == {
            "dev_hr", "dev_hrv", "dev_load", "dev_fatigue", "score",
        }

    def test_10_3_score_matches_compute(self):
        m, f, b = _metrics(hr=80.0), _fatigue(0.5), _baseline()
        comps = deviation_components(m, f, b)
        direct = compute_baseline_deviation(m, f, b)
        assert comps["score"] == direct

    def test_10_4_all_values_finite(self):
        m, f, b = _metrics(hr=75.0), _fatigue(0.4), _baseline(hr=70.0)
        for k, v in deviation_components(m, f, b).items():
            assert math.isfinite(v), f"{k} is not finite: {v}"

    def test_10_5_all_values_json_serialisable(self):
        m, f, b = _metrics(), _fatigue(), _baseline()
        json.dumps(deviation_components(m, f, b))

    def test_10_6_component_deviations_non_negative(self):
        m, f, b = _metrics(hr=65.0, hrv=35.0, load=80.0), _fatigue(0.1), _baseline()
        comps = deviation_components(m, f, b)
        for key in ("dev_hr", "dev_hrv", "dev_load", "dev_fatigue"):
            assert comps[key] >= 0.0, f"{key} is negative: {comps[key]}"

    def test_10_7_dev_fatigue_in_unit_interval(self):
        """Both fatigue sides in [0,1] so dev_fatigue must be in [0,1]."""
        m, f, b = _metrics(), _fatigue(1.0), _baseline(fatigue=0.0)
        comps = deviation_components(m, f, b)
        assert 0.0 <= comps["dev_fatigue"] <= 1.0


# =============================================================================
# SECTION 11 — deviation_components() formula parity
# =============================================================================

class TestDeviationComponentsParity:

    def test_11_1_dev_hr_matches_helper(self):
        m, f, b = _metrics(hr=80.0), _fatigue(), _baseline(hr=70.0)
        comps = deviation_components(m, f, b)
        assert comps["dev_hr"] == pytest.approx(
            _safe_abs_norm(m.mean_hr, b.mean_hr), abs=1e-12
        )

    def test_11_2_dev_hrv_matches_helper(self):
        m, f, b = _metrics(hrv=55.0), _fatigue(), _baseline(hrv=40.0)
        comps = deviation_components(m, f, b)
        assert comps["dev_hrv"] == pytest.approx(
            _safe_abs_norm(m.mean_hrv, b.mean_hrv), abs=1e-12
        )

    def test_11_3_dev_load_matches_helper(self):
        m, f, b = _metrics(load=180.0), _fatigue(), _baseline(load=100.0)
        comps = deviation_components(m, f, b)
        assert comps["dev_load"] == pytest.approx(
            _safe_abs_norm(m.load_mean, b.mean_load), abs=1e-12
        )

    def test_11_4_dev_fatigue_matches_helper(self):
        m, f, b = _metrics(), _fatigue(0.7), _baseline(fatigue=0.2)
        comps = deviation_components(m, f, b)
        assert comps["dev_fatigue"] == pytest.approx(
            _safe_abs_diff(f.value, b.baseline_fatigue), abs=1e-12
        )


# =============================================================================
# SECTION 12 — _safe_abs_norm() helper
# =============================================================================

class TestSafeAbsNorm:

    def test_12_1_normal_inputs(self):
        """current=80, baseline=70 → |80-70|/70 ≈ 0.142857"""
        assert _safe_abs_norm(80.0, 70.0) == pytest.approx(10.0 / 70.0, abs=1e-12)

    def test_12_2_baseline_zero_denominator_one(self):
        """baseline=0 → max(0, 1) = 1; result = |current - 0| / 1"""
        assert _safe_abs_norm(5.0, 0.0) == pytest.approx(5.0, abs=1e-12)

    def test_12_3_baseline_below_one_denominator_one(self):
        """baseline=0.5 → max(0.5, 1) = 1; result = |current - 0.5| / 1"""
        assert _safe_abs_norm(2.0, 0.5) == pytest.approx(1.5, abs=1e-12)

    def test_12_4_non_finite_current_returns_zero(self):
        assert _safe_abs_norm(float("inf"), 70.0) == 0.0
        assert _safe_abs_norm(float("nan"), 70.0) == 0.0

    def test_12_5_non_finite_baseline_returns_zero(self):
        assert _safe_abs_norm(70.0, float("inf")) == 0.0
        assert _safe_abs_norm(70.0, float("nan")) == 0.0

    def test_12_6_result_non_negative(self):
        for cur, base in [(50.0, 70.0), (90.0, 70.0), (70.0, 70.0)]:
            assert _safe_abs_norm(cur, base) >= 0.0


# =============================================================================
# SECTION 13 — _safe_abs_diff() helper
# =============================================================================

class TestSafeAbsDiff:

    def test_13_1_normal_inputs(self):
        assert _safe_abs_diff(0.7, 0.3) == pytest.approx(0.4, abs=1e-12)

    def test_13_2_non_finite_a_returns_zero(self):
        assert _safe_abs_diff(float("nan"), 0.3) == 0.0
        assert _safe_abs_diff(float("inf"), 0.3) == 0.0

    def test_13_3_non_finite_b_returns_zero(self):
        assert _safe_abs_diff(0.7, float("nan")) == 0.0
        assert _safe_abs_diff(0.7, float("inf")) == 0.0

    def test_13_4_result_non_negative(self):
        for a, b in [(0.2, 0.8), (0.8, 0.2), (0.5, 0.5)]:
            assert _safe_abs_diff(a, b) >= 0.0

    def test_13_5_symmetric(self):
        for a, b in [(0.1, 0.9), (0.6, 0.2), (0.0, 1.0)]:
            assert _safe_abs_diff(a, b) == pytest.approx(
                _safe_abs_diff(b, a), abs=1e-15
            )


# =============================================================================
# SECTION 14 — _clamp() helper
# =============================================================================

class TestClamp:

    def test_14_1_within_range_unchanged(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_14_2_below_lo_returns_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_14_3_above_hi_returns_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_14_4_non_finite_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_14_5_boundary_values_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


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