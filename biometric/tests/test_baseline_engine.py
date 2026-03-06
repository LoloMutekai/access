"""
A.C.C.E.S.S. — BaselineEngine Test Suite (Phase 7.8A)
tests/test_baseline_engine.py

Full coverage of biometric/baseline_engine.py

Coverage map
────────────
SECTION 1 — BaselineState dataclass
  1.1   Instantiation with valid values
  1.2   Dataclass is frozen (reassignment raises FrozenInstanceError)
  1.3   All float fields stored exactly
  1.4   sample_count stored as int
  1.5   is_valid() returns True for well-formed state
  1.6   is_valid() returns False when a float is non-finite
  1.7   is_valid() returns False when baseline_fatigue > 1.0
  1.8   is_valid() returns False when baseline_fatigue < 0.0

SECTION 2 — BaselineState.to_dict()
  2.1   Returns a dict
  2.2   Contains exactly the five expected keys
  2.3   Values match fields
  2.4   Returns a fresh dict on each call (mutations don't affect instance)
  2.5   Output is JSON-serialisable

SECTION 3 — BaselineState.from_dict()
  3.1   Reconstructs identical state from to_dict() output
  3.2   Round-trip through json.dumps / json.loads preserves all values
  3.3   Tolerates int sample_count stored as JSON number

SECTION 4 — BaselineEngine construction
  4.1   Default alpha == EMA_ALPHA == 0.05
  4.2   Custom alpha within [0.01, 0.05] accepted
  4.3   alpha = 0.01 accepted (lower bound)
  4.4   alpha = 0.05 accepted (upper bound)
  4.5   alpha < 0.01 raises ValueError
  4.6   alpha > 0.05 raises ValueError
  4.7   alpha = 0.0 raises ValueError
  4.8   alpha = 1.0 raises ValueError
  4.9   Initial sample_count == 0
  4.10  Initial current_state is None

SECTION 5 — BaselineEngine.update() — cold start
  5.1   First call returns a BaselineState
  5.2   Cold-start mean_hr equals clamped observation
  5.3   Cold-start mean_hrv equals clamped observation
  5.4   Cold-start mean_load equals clamped observation
  5.5   Cold-start baseline_fatigue equals clamped fatigue.value
  5.6   Cold-start sample_count == 1
  5.7   Cold-start state passes is_valid()
  5.8   engine.sample_count == 1 after first update
  5.9   engine.current_state is the returned state

SECTION 6 — BaselineEngine.update() — EMA smoothing
  6.1   mean_hr follows EMA formula exactly on second call
  6.2   mean_hrv follows EMA formula exactly on second call
  6.3   mean_load follows EMA formula exactly on second call
  6.4   baseline_fatigue follows EMA formula exactly on second call
  6.5   sample_count == 2 after second call
  6.6   EMA converges toward constant input over many calls
  6.7   Two-step manual EMA matches engine output at step 10
  6.8   alpha=0.01 converges more slowly than alpha=0.05

SECTION 7 — BaselineEngine.update() — sample_count
  7.1   sample_count increments by exactly 1 per call (10 calls)
  7.2   sample_count is always an int
  7.3   sample_count after reset() returns to 0
  7.4   engine.sample_count property mirrors state.sample_count

SECTION 8 — BaselineEngine.update() — value bounds
  8.1   baseline_fatigue always in [0.0, 1.0] (random-like sequence of inputs)
  8.2   mean_hr always finite after 50 updates with varied HR inputs
  8.3   All BaselineState floats finite after 50 mixed updates
  8.4   Clamp guard on HR: observation above _HR_MAX is clamped before EMA

SECTION 9 — Input immutability
  9.1   CoreMetrics object not mutated by update()
  9.2   FatigueResult object not mutated by update()

SECTION 10 — Determinism
  10.1  Identical call sequences on fresh engines → identical states
  10.2  100 repeated calls with same input → identical final state
  10.3  deterministic_check() returns True
  10.4  Order of updates matters (non-commutative)

SECTION 11 — JSON serialisation
  11.1  BaselineState.to_dict() is JSON-serialisable
  11.2  Full round-trip: to_dict → json.dumps → json.loads → from_dict
  11.3  All values in to_dict() have native Python types (float or int)
  11.4  Serialised state can restore engine via set_state()

SECTION 12 — get_state() / set_state()
  12.1  get_state() returns a dict
  12.2  get_state() is JSON-serialisable
  12.3  get_state()["alpha"] matches engine alpha
  12.4  get_state()["baseline"] is None before first update
  12.5  get_state()["baseline"] matches current_state.to_dict() after update
  12.6  set_state() restores baseline values exactly
  12.7  set_state() restores sample_count exactly
  12.8  set_state() restores alpha exactly
  12.9  set_state({"baseline": None}) resets to cold start
  12.10 set_state() with invalid alpha raises ValueError

SECTION 13 — reset()
  13.1  reset() sets sample_count to 0
  13.2  reset() sets current_state to None
  13.3  First update after reset() behaves as cold start

SECTION 14 — self_test()
  14.1  Returns a dict
  14.2  Dict contains "engine", "version", "checks", "passed"
  14.3  "checks" is a list of five items
  14.4  Each check dict has "name", "passed", "detail"
  14.5  All five checks pass on a freshly constructed engine
  14.6  "engine" == ENGINE_NAME
  14.7  "version" == ENGINE_VERSION
  14.8  Does not mutate the engine's own baseline state

SECTION 15 — deterministic_check()
  15.1  Returns True for a freshly constructed engine
  15.2  Returns True regardless of how many prior updates have been applied
  15.3  Does not mutate the engine's own baseline state

SECTION 16 — EMA_ALPHA constant
  16.1  EMA_ALPHA == 0.05
  16.2  EMA_ALPHA ∈ [0.01, 0.05]
  16.3  EMA_ALPHA is a float
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

# Make the project root importable regardless of test runner cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.baseline_engine import (
    BaselineEngine,
    BaselineState,
    EMA_ALPHA,
    ENGINE_NAME,
    ENGINE_VERSION,
    _canonical_pair,
    _clamp,
    _ema,
    _FATIGUE_MAX,
    _FATIGUE_MIN,
    _HR_MAX,
    _HR_MIN,
    _HRV_MAX,
    _HRV_MIN,
    _LOAD_MAX,
    _LOAD_MIN,
)
from biometric.biometric_engine import CoreMetrics, FatigueResult


# =============================================================================
# HELPERS
# =============================================================================

def _make_engine(alpha: float = EMA_ALPHA) -> BaselineEngine:
    return BaselineEngine(alpha=alpha)


def _make_metrics(
    mean_hr:   float = 70.0,
    mean_hrv:  float = 40.0,
    load_mean: float = 100.0,
    hr_std:    float = 5.0,
    hrv_std:   float = 3.0,
) -> CoreMetrics:
    return CoreMetrics(
        mean_hr=mean_hr,
        mean_hrv=mean_hrv,
        load_mean=load_mean,
        hr_std=hr_std,
        hrv_std=hrv_std,
    )


def _make_fatigue(value: float = 0.3, raw: float = 0.6) -> FatigueResult:
    return FatigueResult(value=value, raw=raw)


def _obs(
    hr:      float = 70.0,
    hrv:     float = 40.0,
    load:    float = 100.0,
    fatigue: float = 0.3,
) -> tuple[CoreMetrics, FatigueResult]:
    """Shorthand for building a (CoreMetrics, FatigueResult) pair."""
    return _make_metrics(mean_hr=hr, mean_hrv=hrv, load_mean=load), _make_fatigue(fatigue)


# =============================================================================
# SECTION 1 — BaselineState dataclass
# =============================================================================

class TestBaselineStateDataclass:

    def test_1_1_instantiation(self):
        s = BaselineState(
            mean_hr=70.0, mean_hrv=40.0, mean_load=100.0,
            baseline_fatigue=0.3, sample_count=1,
        )
        assert isinstance(s, BaselineState)

    def test_1_2_frozen(self):
        from dataclasses import FrozenInstanceError
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 1)
        with pytest.raises(FrozenInstanceError):
            s.mean_hr = 999.0  # type: ignore[misc]

    def test_1_3_float_fields_stored_exactly(self):
        s = BaselineState(70.123, 40.456, 100.789, 0.321, 5)
        assert s.mean_hr          == 70.123
        assert s.mean_hrv         == 40.456
        assert s.mean_load        == 100.789
        assert s.baseline_fatigue == 0.321
        assert s.sample_count     == 5

    def test_1_4_sample_count_is_int(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 3)
        assert isinstance(s.sample_count, int)

    def test_1_5_is_valid_true(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.5, 1)
        assert s.is_valid() is True

    def test_1_6_is_valid_false_nonfinite(self):
        s = BaselineState(float("inf"), 40.0, 100.0, 0.5, 1)
        assert s.is_valid() is False

    def test_1_7_is_valid_false_fatigue_too_high(self):
        s = BaselineState(70.0, 40.0, 100.0, 1.001, 1)
        assert s.is_valid() is False

    def test_1_8_is_valid_false_fatigue_negative(self):
        s = BaselineState(70.0, 40.0, 100.0, -0.001, 1)
        assert s.is_valid() is False


# =============================================================================
# SECTION 2 — BaselineState.to_dict()
# =============================================================================

class TestBaselineStateToDict:

    def test_2_1_returns_dict(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 1)
        assert isinstance(s.to_dict(), dict)

    def test_2_2_exact_keys(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 1)
        assert set(s.to_dict().keys()) == {
            "mean_hr", "mean_hrv", "mean_load",
            "baseline_fatigue", "sample_count",
        }

    def test_2_3_values_match_fields(self):
        s = BaselineState(71.5, 42.0, 110.0, 0.45, 7)
        d = s.to_dict()
        assert d["mean_hr"]          == 71.5
        assert d["mean_hrv"]         == 42.0
        assert d["mean_load"]        == 110.0
        assert d["baseline_fatigue"] == 0.45
        assert d["sample_count"]     == 7

    def test_2_4_fresh_dict_no_leakback(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 1)
        d = s.to_dict()
        d["mean_hr"] = 999.0
        assert s.mean_hr == 70.0

    def test_2_5_json_serialisable(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 1)
        json.dumps(s.to_dict())   # must not raise


# =============================================================================
# SECTION 3 — BaselineState.from_dict()
# =============================================================================

class TestBaselineStateFromDict:

    def test_3_1_roundtrip_identical(self):
        s = BaselineState(70.0, 40.0, 100.0, 0.3, 5)
        assert BaselineState.from_dict(s.to_dict()) == s

    def test_3_2_json_roundtrip(self):
        s = BaselineState(71.23456789, 41.5, 150.0, 0.42, 10)
        restored = BaselineState.from_dict(
            json.loads(json.dumps(s.to_dict()))
        )
        assert abs(restored.mean_hr          - s.mean_hr)          < 1e-9
        assert abs(restored.mean_hrv         - s.mean_hrv)         < 1e-9
        assert abs(restored.mean_load        - s.mean_load)        < 1e-9
        assert abs(restored.baseline_fatigue - s.baseline_fatigue) < 1e-9
        assert restored.sample_count == s.sample_count

    def test_3_3_int_sample_count_from_json_number(self):
        d = {"mean_hr": 70.0, "mean_hrv": 40.0, "mean_load": 100.0,
             "baseline_fatigue": 0.3, "sample_count": 3}
        s = BaselineState.from_dict(d)
        assert isinstance(s.sample_count, int)
        assert s.sample_count == 3


# =============================================================================
# SECTION 4 — BaselineEngine construction
# =============================================================================

class TestBaselineEngineConstruction:

    def test_4_1_default_alpha(self):
        assert _make_engine().alpha == EMA_ALPHA

    def test_4_2_custom_alpha_accepted(self):
        e = BaselineEngine(alpha=0.03)
        assert e.alpha == pytest.approx(0.03)

    def test_4_3_alpha_lower_bound(self):
        e = BaselineEngine(alpha=0.01)
        assert e.alpha == pytest.approx(0.01)

    def test_4_4_alpha_upper_bound(self):
        e = BaselineEngine(alpha=0.05)
        assert e.alpha == pytest.approx(0.05)

    def test_4_5_alpha_too_small_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            BaselineEngine(alpha=0.009)

    def test_4_6_alpha_too_large_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            BaselineEngine(alpha=0.051)

    def test_4_7_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            BaselineEngine(alpha=0.0)

    def test_4_8_alpha_one_raises(self):
        with pytest.raises(ValueError):
            BaselineEngine(alpha=1.0)

    def test_4_9_initial_sample_count_zero(self):
        assert _make_engine().sample_count == 0

    def test_4_10_initial_current_state_none(self):
        assert _make_engine().current_state is None


# =============================================================================
# SECTION 5 — update() — cold start
# =============================================================================

class TestUpdateColdStart:

    def test_5_1_returns_baseline_state(self):
        m, f = _obs()
        assert isinstance(_make_engine().update(m, f), BaselineState)

    def test_5_2_cold_start_mean_hr(self):
        m, f = _obs(hr=75.0)
        s = _make_engine().update(m, f)
        assert s.mean_hr == pytest.approx(_clamp(75.0, _HR_MIN, _HR_MAX))

    def test_5_3_cold_start_mean_hrv(self):
        m, f = _obs(hrv=45.0)
        s = _make_engine().update(m, f)
        assert s.mean_hrv == pytest.approx(_clamp(45.0, _HRV_MIN, _HRV_MAX))

    def test_5_4_cold_start_mean_load(self):
        m, f = _obs(load=120.0)
        s = _make_engine().update(m, f)
        assert s.mean_load == pytest.approx(_clamp(120.0, _LOAD_MIN, _LOAD_MAX))

    def test_5_5_cold_start_baseline_fatigue(self):
        m, f = _obs(fatigue=0.45)
        s = _make_engine().update(m, f)
        assert s.baseline_fatigue == pytest.approx(
            _clamp(0.45, _FATIGUE_MIN, _FATIGUE_MAX)
        )

    def test_5_6_cold_start_sample_count_one(self):
        m, f = _obs()
        s = _make_engine().update(m, f)
        assert s.sample_count == 1

    def test_5_7_cold_start_is_valid(self):
        m, f = _obs()
        s = _make_engine().update(m, f)
        assert s.is_valid() is True

    def test_5_8_engine_sample_count_after_first_update(self):
        eng = _make_engine()
        m, f = _obs()
        eng.update(m, f)
        assert eng.sample_count == 1

    def test_5_9_current_state_is_returned_state(self):
        eng = _make_engine()
        m, f = _obs()
        s = eng.update(m, f)
        assert eng.current_state is s


# =============================================================================
# SECTION 6 — update() — EMA smoothing
# =============================================================================

class TestUpdateEMA:

    def test_6_1_mean_hr_ema_exact(self):
        α = EMA_ALPHA
        eng = _make_engine()
        m1, f1 = _obs(hr=70.0)
        m2, f2 = _obs(hr=80.0)
        eng.update(m1, f1)
        s2 = eng.update(m2, f2)
        expected = (1 - α) * 70.0 + α * 80.0
        assert s2.mean_hr == pytest.approx(expected, abs=1e-9)

    def test_6_2_mean_hrv_ema_exact(self):
        α = EMA_ALPHA
        eng = _make_engine()
        eng.update(*_obs(hrv=40.0))
        s2 = eng.update(*_obs(hrv=55.0))
        expected = (1 - α) * 40.0 + α * 55.0
        assert s2.mean_hrv == pytest.approx(expected, abs=1e-9)

    def test_6_3_mean_load_ema_exact(self):
        α = EMA_ALPHA
        eng = _make_engine()
        eng.update(*_obs(load=100.0))
        s2 = eng.update(*_obs(load=200.0))
        expected = (1 - α) * 100.0 + α * 200.0
        assert s2.mean_load == pytest.approx(expected, abs=1e-9)

    def test_6_4_baseline_fatigue_ema_exact(self):
        α = EMA_ALPHA
        eng = _make_engine()
        eng.update(*_obs(fatigue=0.3))
        s2 = eng.update(*_obs(fatigue=0.7))
        expected = (1 - α) * 0.3 + α * 0.7
        assert s2.baseline_fatigue == pytest.approx(expected, abs=1e-9)

    def test_6_5_sample_count_two_after_second_update(self):
        eng = _make_engine()
        eng.update(*_obs())
        s2 = eng.update(*_obs(hr=80.0))
        assert s2.sample_count == 2

    def test_6_6_ema_converges_toward_constant_input(self):
        """After many updates with the same input, baseline should approach it."""
        target_hr = 90.0
        eng = _make_engine()
        # Seed with a distant value
        eng.update(*_obs(hr=60.0))
        for _ in range(200):
            eng.update(*_obs(hr=target_hr))
        assert abs(eng.current_state.mean_hr - target_hr) < 0.5

    def test_6_7_manual_ema_matches_engine_at_step_10(self):
        α = EMA_ALPHA
        eng = _make_engine()
        m_hr_series = [70.0, 72.0, 68.0, 75.0, 71.0, 73.0, 69.0, 74.0, 70.5, 72.5]
        manual_hr = m_hr_series[0]
        eng.update(*_obs(hr=m_hr_series[0]))
        for hr in m_hr_series[1:]:
            manual_hr = (1 - α) * manual_hr + α * hr
            s = eng.update(*_obs(hr=hr))
        assert abs(s.mean_hr - manual_hr) < 1e-9

    def test_6_8_slow_alpha_converges_more_slowly(self):
        eng_fast = BaselineEngine(alpha=0.05)
        eng_slow = BaselineEngine(alpha=0.01)
        # Seed both with 60 bpm
        eng_fast.update(*_obs(hr=60.0))
        eng_slow.update(*_obs(hr=60.0))
        # Drive both toward 90 bpm for 50 steps
        for _ in range(50):
            eng_fast.update(*_obs(hr=90.0))
            eng_slow.update(*_obs(hr=90.0))
        # Fast engine should be closer to 90
        diff_fast = abs(eng_fast.current_state.mean_hr - 90.0)
        diff_slow = abs(eng_slow.current_state.mean_hr - 90.0)
        assert diff_fast < diff_slow


# =============================================================================
# SECTION 7 — update() — sample_count
# =============================================================================

class TestSampleCount:

    def test_7_1_increments_by_one_per_call(self):
        eng = _make_engine()
        for i in range(1, 11):
            s = eng.update(*_obs())
            assert s.sample_count == i, f"step {i}: got {s.sample_count}"

    def test_7_2_sample_count_is_int(self):
        eng = _make_engine()
        s = eng.update(*_obs())
        assert isinstance(s.sample_count, int)

    def test_7_3_sample_count_zero_after_reset(self):
        eng = _make_engine()
        for _ in range(5):
            eng.update(*_obs())
        eng.reset()
        assert eng.sample_count == 0

    def test_7_4_property_mirrors_state(self):
        eng = _make_engine()
        for _ in range(7):
            s = eng.update(*_obs())
        assert eng.sample_count == s.sample_count == 7


# =============================================================================
# SECTION 8 — update() — value bounds
# =============================================================================

class TestValueBounds:

    def test_8_1_baseline_fatigue_always_in_unit_interval(self):
        """Alternating extreme fatigue values must stay in [0, 1]."""
        eng = _make_engine()
        for i in range(50):
            fatigue_val = 0.0 if i % 2 == 0 else 1.0
            s = eng.update(*_obs(fatigue=fatigue_val))
            assert 0.0 <= s.baseline_fatigue <= 1.0, (
                f"step {i+1}: baseline_fatigue={s.baseline_fatigue}"
            )

    def test_8_2_mean_hr_always_finite(self):
        eng = _make_engine()
        for hr in [60.0, 80.0, 75.0, 90.0, 65.0] * 10:
            s = eng.update(*_obs(hr=hr))
            assert math.isfinite(s.mean_hr)

    def test_8_3_all_floats_finite_after_many_updates(self):
        eng = _make_engine()
        for _ in range(50):
            s = eng.update(*_obs(
                hr=70.0, hrv=40.0, load=100.0, fatigue=0.3,
            ))
        assert s.is_valid()
        assert math.isfinite(s.mean_hr)
        assert math.isfinite(s.mean_hrv)
        assert math.isfinite(s.mean_load)
        assert math.isfinite(s.baseline_fatigue)

    def test_8_4_clamp_guard_on_hr_above_max(self):
        """An extreme HR observation is clamped before the EMA step."""
        eng = _make_engine()
        eng.update(*_obs(hr=70.0))
        # Pass an astronomically large HR — must be clamped to _HR_MAX
        s = eng.update(*_obs(hr=1_000_000.0))
        # EMA with clamped value: (1-α)*70 + α*_HR_MAX
        expected = (1 - EMA_ALPHA) * 70.0 + EMA_ALPHA * _HR_MAX
        assert s.mean_hr == pytest.approx(expected, abs=1e-6)
        assert math.isfinite(s.mean_hr)


# =============================================================================
# SECTION 9 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_9_1_core_metrics_not_mutated(self):
        eng = _make_engine()
        m   = _make_metrics(mean_hr=70.0, mean_hrv=40.0, load_mean=100.0)
        original_hr   = m.mean_hr
        original_hrv  = m.mean_hrv
        original_load = m.load_mean
        eng.update(m, _make_fatigue())
        assert m.mean_hr   == original_hr
        assert m.mean_hrv  == original_hrv
        assert m.load_mean == original_load

    def test_9_2_fatigue_result_not_mutated(self):
        eng = _make_engine()
        f   = _make_fatigue(value=0.4, raw=0.8)
        original_val = f.value
        original_raw = f.raw
        eng.update(_make_metrics(), f)
        assert f.value == original_val
        assert f.raw   == original_raw


# =============================================================================
# SECTION 10 — Determinism
# =============================================================================

class TestDeterminism:

    def test_10_1_identical_sequences_produce_identical_states(self):
        def _run() -> list[dict]:
            eng = _make_engine()
            inputs = [
                _obs(hr=70.0, hrv=40.0, load=100.0, fatigue=0.3),
                _obs(hr=75.0, hrv=42.0, load=120.0, fatigue=0.35),
                _obs(hr=68.0, hrv=38.0, load=90.0,  fatigue=0.28),
            ]
            return [eng.update(m, f).to_dict() for m, f in inputs]

        assert _run() == _run()

    def test_10_2_100_identical_inputs_same_final_state(self):
        eng1 = _make_engine()
        eng2 = _make_engine()
        for _ in range(100):
            eng1.update(*_obs())
            eng2.update(*_obs())
        assert eng1.current_state.to_dict() == eng2.current_state.to_dict()

    def test_10_3_deterministic_check_returns_true(self):
        assert _make_engine().deterministic_check() is True

    def test_10_4_order_matters(self):
        """Non-commutative: swapping first two observations gives different state."""
        eng1 = _make_engine()
        eng1.update(*_obs(hr=60.0))
        eng1.update(*_obs(hr=90.0))

        eng2 = _make_engine()
        eng2.update(*_obs(hr=90.0))
        eng2.update(*_obs(hr=60.0))

        assert eng1.current_state.mean_hr != eng2.current_state.mean_hr


# =============================================================================
# SECTION 11 — JSON serialisation
# =============================================================================

class TestJsonSerialisation:

    def test_11_1_to_dict_json_serialisable(self):
        eng = _make_engine()
        s = eng.update(*_obs())
        json.dumps(s.to_dict())   # must not raise

    def test_11_2_full_roundtrip(self):
        eng = _make_engine()
        eng.update(*_obs(hr=70.0))
        s = eng.update(*_obs(hr=72.0))
        snap = json.loads(json.dumps(s.to_dict()))
        restored = BaselineState.from_dict(snap)
        assert abs(restored.mean_hr          - s.mean_hr)          < 1e-9
        assert abs(restored.mean_hrv         - s.mean_hrv)         < 1e-9
        assert abs(restored.mean_load        - s.mean_load)        < 1e-9
        assert abs(restored.baseline_fatigue - s.baseline_fatigue) < 1e-9
        assert restored.sample_count == s.sample_count

    def test_11_3_native_python_types(self):
        eng = _make_engine()
        s = eng.update(*_obs())
        d = s.to_dict()
        for key, val in d.items():
            if key == "sample_count":
                assert isinstance(val, int), f"{key}: {type(val)}"
            else:
                assert isinstance(val, float), f"{key}: {type(val)}"

    def test_11_4_serialised_state_restores_engine(self):
        eng = _make_engine()
        for _ in range(5):
            eng.update(*_obs())
        snap = eng.get_state()
        raw  = json.dumps(snap)

        eng2 = _make_engine()
        eng2.set_state(json.loads(raw))
        assert eng2.current_state.to_dict() == eng.current_state.to_dict()


# =============================================================================
# SECTION 12 — get_state() / set_state()
# =============================================================================

class TestGetSetState:

    def test_12_1_get_state_returns_dict(self):
        assert isinstance(_make_engine().get_state(), dict)

    def test_12_2_get_state_json_serialisable(self):
        eng = _make_engine()
        eng.update(*_obs())
        json.dumps(eng.get_state())

    def test_12_3_get_state_alpha_matches(self):
        eng = BaselineEngine(alpha=0.03)
        assert eng.get_state()["alpha"] == pytest.approx(0.03)

    def test_12_4_get_state_baseline_none_before_first_update(self):
        assert _make_engine().get_state()["baseline"] is None

    def test_12_5_get_state_baseline_matches_current_state(self):
        eng = _make_engine()
        s = eng.update(*_obs())
        assert eng.get_state()["baseline"] == s.to_dict()

    def test_12_6_set_state_restores_baseline_values(self):
        eng1 = _make_engine()
        for _ in range(5):
            eng1.update(*_obs(hr=70.0, hrv=40.0, load=100.0, fatigue=0.3))
        snap = eng1.get_state()

        eng2 = _make_engine()
        eng2.set_state(snap)
        assert eng2.current_state.mean_hr          == pytest.approx(eng1.current_state.mean_hr)
        assert eng2.current_state.mean_hrv         == pytest.approx(eng1.current_state.mean_hrv)
        assert eng2.current_state.mean_load        == pytest.approx(eng1.current_state.mean_load)
        assert eng2.current_state.baseline_fatigue == pytest.approx(eng1.current_state.baseline_fatigue)

    def test_12_7_set_state_restores_sample_count(self):
        eng1 = _make_engine()
        for _ in range(7):
            eng1.update(*_obs())
        eng2 = _make_engine()
        eng2.set_state(eng1.get_state())
        assert eng2.sample_count == 7

    def test_12_8_set_state_restores_alpha(self):
        eng1 = BaselineEngine(alpha=0.02)
        snap = eng1.get_state()
        eng2 = BaselineEngine(alpha=0.05)
        eng2.set_state(snap)
        assert eng2.alpha == pytest.approx(0.02)

    def test_12_9_set_state_none_baseline_resets_to_cold_start(self):
        eng = _make_engine()
        for _ in range(3):
            eng.update(*_obs())
        eng.set_state({"alpha": EMA_ALPHA, "baseline": None})
        assert eng.current_state is None
        assert eng.sample_count == 0

    def test_12_10_set_state_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            _make_engine().set_state({"alpha": 0.5, "baseline": None})


# =============================================================================
# SECTION 13 — reset()
# =============================================================================

class TestReset:

    def test_13_1_reset_sets_sample_count_zero(self):
        eng = _make_engine()
        for _ in range(5):
            eng.update(*_obs())
        eng.reset()
        assert eng.sample_count == 0

    def test_13_2_reset_sets_current_state_none(self):
        eng = _make_engine()
        eng.update(*_obs())
        eng.reset()
        assert eng.current_state is None

    def test_13_3_first_update_after_reset_is_cold_start(self):
        eng = _make_engine()
        # Warm the engine first
        for _ in range(5):
            eng.update(*_obs(hr=90.0))
        eng.reset()
        # Cold start: baseline should equal the new observation
        s = eng.update(*_obs(hr=65.0))
        assert s.mean_hr     == pytest.approx(65.0)
        assert s.sample_count == 1


# =============================================================================
# SECTION 14 — self_test()
# =============================================================================

class TestSelfTest:

    def test_14_1_returns_dict(self):
        assert isinstance(_make_engine().self_test(), dict)

    def test_14_2_required_top_level_keys(self):
        result = _make_engine().self_test()
        assert set(result.keys()) >= {"engine", "version", "checks", "passed"}

    def test_14_3_checks_has_five_items(self):
        result = _make_engine().self_test()
        assert len(result["checks"]) == 5

    def test_14_4_each_check_has_required_keys(self):
        for check in _make_engine().self_test()["checks"]:
            assert "name"   in check
            assert "passed" in check
            assert "detail" in check

    def test_14_5_all_five_checks_pass(self):
        result = _make_engine().self_test()
        assert result["passed"] is True
        failures = [c for c in result["checks"] if not c["passed"]]
        assert failures == [], f"Failed checks: {failures}"

    def test_14_6_engine_name_correct(self):
        assert _make_engine().self_test()["engine"] == ENGINE_NAME

    def test_14_7_version_correct(self):
        assert _make_engine().self_test()["version"] == ENGINE_VERSION

    def test_14_8_does_not_mutate_live_state(self):
        """self_test() uses fresh probe instances — live state unchanged."""
        eng = _make_engine()
        eng.update(*_obs(hr=75.0))
        eng.update(*_obs(hr=80.0))
        state_before = eng.current_state.to_dict()
        eng.self_test()
        assert eng.current_state.to_dict() == state_before


# =============================================================================
# SECTION 15 — deterministic_check()
# =============================================================================

class TestDeterministicCheck:

    def test_15_1_returns_true_for_fresh_engine(self):
        assert _make_engine().deterministic_check() is True

    def test_15_2_returns_true_after_prior_updates(self):
        eng = _make_engine()
        for _ in range(20):
            eng.update(*_obs(hr=70.0))
        assert eng.deterministic_check() is True

    def test_15_3_does_not_mutate_live_state(self):
        eng = _make_engine()
        eng.update(*_obs(hr=70.0))
        eng.update(*_obs(hr=75.0))
        state_before = eng.current_state.to_dict()
        eng.deterministic_check()
        assert eng.current_state.to_dict() == state_before


# =============================================================================
# SECTION 16 — EMA_ALPHA constant
# =============================================================================

class TestEMAAlphaConstant:

    def test_16_1_value_is_0_05(self):
        assert EMA_ALPHA == 0.05

    def test_16_2_within_valid_range(self):
        assert 0.01 <= EMA_ALPHA <= 0.05

    def test_16_3_is_float(self):
        assert isinstance(EMA_ALPHA, float)


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