"""
A.C.C.E.S.S. — Performance Forecast Engine Test Suite (Phase 7.13)
tests/test_performance_forecast_engine.py

Full coverage of biometric/performance_forecast_engine.py

Coverage map
────────────
SECTION 1  — Module constants (12 tests)
  1.1   ALPHA_LOAD == 0.35
  1.2   BETA_RECOVERY == 0.45
  1.3   LOAD_REFERENCE == 8000.0
  1.4   W_FATIGUE == 0.6
  1.5   W_RECOVERY_DEFICIT == 0.4
  1.6   W_FATIGUE + W_RECOVERY_DEFICIT == 1.0
  1.7   READINESS_INJURY_WEIGHT == 0.5
  1.8   DEFAULT_FATIGUE == 0.0
  1.9   DEFAULT_RECOVERY == 24.0
  1.10  DEFAULT_LOAD == 0.0
  1.11  FORECAST_HORIZONS == (24, 48, 72)
  1.12  ENGINE_NAME and ENGINE_VERSION are correct strings

SECTION 2  — ForecastResult dataclass (12 tests)
  2.1   Instantiates with valid fields
  2.2   Frozen (FrozenInstanceError on reassignment)
  2.3   All six fields stored exactly
  2.4   is_valid() True for well-formed result
  2.5   is_valid() False when any field is non-finite
  2.6   is_valid() False when any field < 0
  2.7   is_valid() False when any field > 1
  2.8   is_valid() checks all six fields independently
  2.9   is_valid() boundary: 0.0 and 1.0 are valid
  2.10  to_dict() returns dict with exactly six expected keys
  2.11  to_dict() values match fields
  2.12  to_dict() returns fresh dict (mutation does not affect instance)

SECTION 3  — ForecastResult.to_dict() (6 tests)
  3.1   Returns a dict
  3.2   Exactly six keys
  3.3   All values are native Python floats
  3.4   Output is JSON-serialisable
  3.5   Full json.dumps/loads round-trip preserves values
  3.6   Repeated calls are idempotent

SECTION 4  — _is_valid_event() and _collect_valid_events() (8 tests)
  4.1   Valid event accepted
  4.2   Event with missing field rejected
  4.3   Event with non-finite fatigue_index rejected
  4.4   Event with non-numeric type rejected
  4.5   Non-dict entry rejected
  4.6   None rejected
  4.7   Mix of valid and invalid: only valid counted
  4.8   MAX_EVENTS cap enforced

SECTION 5  — _compute_event_summary() (10 tests)
  5.1   Empty list → uses all defaults
  5.2   Single event → means equal that event's values
  5.3   Two events → means are arithmetic means
  5.4   mean_fatigue clamped to [0, 1]
  5.5   norm_recovery clamped to [0, 1]
  5.6   norm_load unbounded above (large load)
  5.7   event_count matches valid event count
  5.8   No mutation of input list
  5.9   Returns _EventSummary with all finite fields
  5.10  Default norm_recovery == 1.0 (DEFAULT_RECOVERY/24)

SECTION 6  — _step_fatigue() (7 tests)
  6.1   Positive delta raises fatigue
  6.2   Negative delta lowers fatigue
  6.3   Zero delta is identity
  6.4   Large positive delta → clamped to 1.0
  6.5   Large negative delta → clamped to 0.0
  6.6   Non-finite input → lo returned (0.0)
  6.7   Result always finite

SECTION 7  — _project_fatigue() (9 tests)
  7.1   Returns a 3-tuple
  7.2   All three values ∈ [0, 1]
  7.3   All three values finite
  7.4   Monotone increasing with positive delta (high load, no recovery)
  7.5   Monotone decreasing with negative delta (no load, full recovery)
  7.6   Stepwise: f48 derived from f24, f72 derived from f48
  7.7   Zero load, full recovery, zero fatigue → all three stay 0
  7.8   Manual formula cross-check with known inputs
  7.9   Default summary → all three horizons == 0.0

SECTION 8  — forecast_fatigue() (12 tests)
  8.1   Returns a dict
  8.2   Exactly the three fatigue keys
  8.3   All values ∈ [0, 1]
  8.4   All values finite
  8.5   Empty events → all fatigue == 0.0
  8.6   All-invalid events → same as empty
  8.7   High load, zero recovery → fatigue_24h ≥ input fatigue
  8.8   Zero load, full recovery, high starting fatigue → fatigue decreases
  8.9   fatigue_48h ≥ fatigue_24h when delta > 0
  8.10  fatigue_48h ≤ fatigue_24h when delta < 0
  8.11  JSON-serialisable output
  8.12  Does not mutate event list

SECTION 9  — forecast_recovery() (9 tests)
  9.1   Returns float
  9.2   Result ∈ [0, 1]
  9.3   Result finite
  9.4   Empty events → recovery == 1.0 (default: fully rested)
  9.5   High fatigue events → lower recovery forecast
  9.6   Low fatigue events → higher recovery forecast
  9.7   recovery_forecast == 1 − fatigue_24h (manual cross-check)
  9.8   Consistent with forecast()["recovery_forecast"]
  9.9   JSON-serialisable (plain float)

SECTION 10 — forecast_injury_risk() (10 tests)
  10.1  Returns float
  10.2  Result ∈ [0, 1]
  10.3  Result finite
  10.4  Empty events → injury_risk == W_RECOVERY_DEFICIT (default: full recovery, zero fatigue)
  10.5  Manual formula cross-check: W_FATIGUE*f24 + W_RECOVERY_DEFICIT*(1−norm_rec)
  10.6  High fatigue + no recovery → high injury risk
  10.7  Zero fatigue + full recovery → min injury risk
  10.8  Result clamped: cannot exceed 1.0
  10.9  Consistent with forecast()["injury_risk_forecast"]
  10.10 Does not mutate event list

SECTION 11 — compute_readiness() (9 tests)
  11.1  Returns float
  11.2  Result ∈ [0, 1]
  11.3  Result finite
  11.4  Empty events → readiness == 1.0 (default: fully rested, zero risk)
  11.5  Manual formula: 1 − fatigue_24h − injury_risk * 0.5
  11.6  High fatigue, high risk → readiness ≈ 0
  11.7  Low fatigue, low risk → readiness ≈ 1
  11.8  Consistent with forecast()["readiness_score"]
  11.9  Result always ≥ 0 (clamped)

SECTION 12 — forecast() output structure (10 tests)
  12.1  Returns a dict
  12.2  Exactly six required keys
  12.3  All values ∈ [0, 1]
  12.4  All values finite
  12.5  All values native Python floats
  12.6  JSON-serialisable
  12.7  Empty events → known default outputs
  12.8  All-invalid events → same as empty
  12.9  Extra fields in events are silently ignored
  12.10 Single valid event produces valid output

SECTION 13 — forecast() formula correctness (9 tests)
  13.1  fatigue_24h == f0 + ALPHA_LOAD*norm_load − BETA_RECOVERY*norm_rec (no clamp needed)
  13.2  fatigue_48h == step(fatigue_24h, delta)
  13.3  fatigue_72h == step(fatigue_48h, delta)
  13.4  recovery_forecast == 1 − fatigue_24h (manual)
  13.5  injury_risk == W_FATIGUE * f24 + W_RECOVERY_DEFICIT * (1 − norm_rec) (manual)
  13.6  readiness == 1 − f24 − ir * 0.5 (manual)
  13.7  Multi-event means used as inputs: two-event average
  13.8  Clamp prevents fatigue > 1.0 on extreme load
  13.9  Clamp prevents fatigue < 0.0 on high recovery

SECTION 14 — Determinism (5 tests)
  14.1  100 identical calls → identical forecast
  14.2  Two independent engines produce identical results
  14.3  deterministic_check() returns True
  14.4  Event order affects score (non-trivially non-commutative via mean)
  14.5  Call-order independence (stateless engine)

SECTION 15 — Input immutability (4 tests)
  15.1  forecast() does not mutate event list
  15.2  forecast() does not mutate individual event dicts
  15.3  forecast_fatigue() does not mutate event list
  15.4  forecast_recovery() does not mutate event list

SECTION 16 — JSON serialisability (5 tests)
  16.1  forecast() output passes json.dumps
  16.2  Full round-trip: json.dumps → json.loads preserves all values
  16.3  ForecastResult.to_dict() passes json.dumps
  16.4  All values are native Python floats (not int or numpy)
  16.5  forecast_fatigue() output is JSON-serialisable

SECTION 17 — self_test() (8 tests)
  17.1  Returns a dict
  17.2  Contains "engine", "version", "checks", "passed"
  17.3  "checks" has exactly six items
  17.4  Each check has "name", "passed", "detail"
  17.5  All six checks pass
  17.6  "engine" == ENGINE_NAME
  17.7  "version" == ENGINE_VERSION
  17.8  Stateless: can be called multiple times identically

SECTION 18 — _clamp() helper (6 tests)
  18.1  Value within range returned unchanged
  18.2  Value below lo → lo
  18.3  Value above hi → hi
  18.4  NaN → lo
  18.5  +Inf → lo
  18.6  Boundary values exact

SECTION 19 — _finite_or_zero() helper (4 tests)
  19.1  Finite value returned unchanged
  19.2  NaN → 0.0
  19.3  +Inf → 0.0
  19.4  -Inf → 0.0
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.performance_forecast_engine import (
    ALPHA_LOAD,
    BETA_RECOVERY,
    DEFAULT_FATIGUE,
    DEFAULT_LOAD,
    DEFAULT_RECOVERY,
    ENGINE_NAME,
    ENGINE_VERSION,
    FORECAST_HORIZONS,
    LOAD_REFERENCE,
    MAX_EVENTS,
    READINESS_INJURY_WEIGHT,
    W_FATIGUE,
    W_RECOVERY_DEFICIT,
    ForecastResult,
    PerformanceForecastEngine,
    _EventSummary,
    _clamp,
    _collect_valid_events,
    _compute_event_summary,
    _compute_injury_risk,
    _compute_readiness,
    _compute_recovery,
    _finite_or_zero,
    _is_valid_event,
    _make_probe_events,
    _project_fatigue,
    _step_fatigue,
)


# =============================================================================
# HELPERS
# =============================================================================

def _eng() -> PerformanceForecastEngine:
    return PerformanceForecastEngine()


def _ev(
    fi:  float = 0.5,
    sl:  float = 4_000.0,
    rh:  float = 24.0,
    inj: int   = 0,
) -> dict:
    return {
        "fatigue_index":  fi,
        "sprint_load":    sl,
        "recovery_hours": rh,
        "injury_flag":    inj,
    }


def _evlist(
    n:   int   = 10,
    fi:  float = 0.5,
    sl:  float = 4_000.0,
    rh:  float = 24.0,
    inj: int   = 0,
) -> list[dict]:
    return [_ev(fi=fi, sl=sl, rh=rh, inj=inj) for _ in range(n)]


def _summary(
    fi:  float = 0.5,
    sl:  float = 4_000.0,
    rh:  float = 24.0,
) -> _EventSummary:
    return _compute_event_summary([_ev(fi=fi, sl=sl, rh=rh)])


def _manual_delta(norm_load: float, norm_recovery: float) -> float:
    return ALPHA_LOAD * norm_load - BETA_RECOVERY * norm_recovery


# =============================================================================
# SECTION 1 — Module constants
# =============================================================================

class TestConstants:

    def test_1_1_alpha_load(self):
        assert ALPHA_LOAD == pytest.approx(0.35)

    def test_1_2_beta_recovery(self):
        assert BETA_RECOVERY == pytest.approx(0.45)

    def test_1_3_load_reference(self):
        assert LOAD_REFERENCE == pytest.approx(8_000.0)

    def test_1_4_w_fatigue(self):
        assert W_FATIGUE == pytest.approx(0.6)

    def test_1_5_w_recovery_deficit(self):
        assert W_RECOVERY_DEFICIT == pytest.approx(0.4)

    def test_1_6_injury_weights_sum_to_one(self):
        assert W_FATIGUE + W_RECOVERY_DEFICIT == pytest.approx(1.0, abs=1e-9)

    def test_1_7_readiness_injury_weight(self):
        assert READINESS_INJURY_WEIGHT == pytest.approx(0.5)

    def test_1_8_default_fatigue(self):
        assert DEFAULT_FATIGUE == 0.0

    def test_1_9_default_recovery(self):
        assert DEFAULT_RECOVERY == 24.0

    def test_1_10_default_load(self):
        assert DEFAULT_LOAD == 0.0

    def test_1_11_forecast_horizons(self):
        assert FORECAST_HORIZONS == (24, 48, 72)

    def test_1_12_engine_metadata(self):
        assert ENGINE_NAME    == "PerformanceForecastEngine"
        assert ENGINE_VERSION == "7.13.0"


# =============================================================================
# SECTION 2 — ForecastResult dataclass
# =============================================================================

class TestForecastResult:

    def _fr(self, **kw) -> ForecastResult:
        defaults = dict(
            fatigue_24h=0.3, fatigue_48h=0.35, fatigue_72h=0.38,
            recovery_forecast=0.70, injury_risk_forecast=0.25,
            readiness_score=0.60,
        )
        defaults.update(kw)
        return ForecastResult(**defaults)

    def test_2_1_instantiation(self):
        assert isinstance(self._fr(), ForecastResult)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            self._fr().fatigue_24h = 99.0  # type: ignore[misc]

    def test_2_3_fields_stored_exactly(self):
        fr = ForecastResult(0.3, 0.35, 0.38, 0.70, 0.25, 0.60)
        assert fr.fatigue_24h          == 0.3
        assert fr.fatigue_48h          == 0.35
        assert fr.fatigue_72h          == 0.38
        assert fr.recovery_forecast    == 0.70
        assert fr.injury_risk_forecast == 0.25
        assert fr.readiness_score      == 0.60

    def test_2_4_is_valid_true(self):
        assert self._fr().is_valid() is True

    def test_2_5_is_valid_nonfinite_field(self):
        assert self._fr(fatigue_24h=float("nan")).is_valid() is False
        assert self._fr(injury_risk_forecast=float("inf")).is_valid() is False

    def test_2_6_is_valid_field_below_zero(self):
        assert self._fr(readiness_score=-0.001).is_valid() is False

    def test_2_7_is_valid_field_above_one(self):
        assert self._fr(fatigue_48h=1.001).is_valid() is False

    def test_2_8_is_valid_all_fields_checked(self):
        for field in ("fatigue_24h", "fatigue_48h", "fatigue_72h",
                      "recovery_forecast", "injury_risk_forecast", "readiness_score"):
            assert self._fr(**{field: -0.5}).is_valid() is False, \
                f"is_valid() should be False for {field}=-0.5"

    def test_2_9_is_valid_boundary_values(self):
        assert self._fr(fatigue_24h=0.0, readiness_score=1.0).is_valid() is True

    def test_2_10_to_dict_six_keys(self):
        expected = {"fatigue_24h", "fatigue_48h", "fatigue_72h",
                    "recovery_forecast", "injury_risk_forecast", "readiness_score"}
        assert set(self._fr().to_dict()) == expected

    def test_2_11_to_dict_values_match_fields(self):
        fr = self._fr()
        d  = fr.to_dict()
        assert d["fatigue_24h"]          == fr.fatigue_24h
        assert d["fatigue_48h"]          == fr.fatigue_48h
        assert d["fatigue_72h"]          == fr.fatigue_72h
        assert d["recovery_forecast"]    == fr.recovery_forecast
        assert d["injury_risk_forecast"] == fr.injury_risk_forecast
        assert d["readiness_score"]      == fr.readiness_score

    def test_2_12_to_dict_fresh_dict(self):
        fr = self._fr()
        d  = fr.to_dict()
        d["fatigue_24h"] = 999.0
        assert fr.fatigue_24h != 999.0


# =============================================================================
# SECTION 3 — ForecastResult.to_dict()
# =============================================================================

class TestForecastResultToDict:

    def _fr(self):
        return ForecastResult(0.3, 0.35, 0.38, 0.70, 0.25, 0.60)

    def test_3_1_returns_dict(self):
        assert isinstance(self._fr().to_dict(), dict)

    def test_3_2_exactly_six_keys(self):
        assert len(self._fr().to_dict()) == 6

    def test_3_3_all_native_floats(self):
        for v in self._fr().to_dict().values():
            assert isinstance(v, float), f"got {type(v)}"

    def test_3_4_json_serialisable(self):
        json.dumps(self._fr().to_dict())

    def test_3_5_json_round_trip(self):
        fr   = self._fr()
        back = json.loads(json.dumps(fr.to_dict()))
        for k, v in fr.to_dict().items():
            assert abs(back[k] - v) < 1e-9

    def test_3_6_idempotent(self):
        fr = self._fr()
        assert fr.to_dict() == fr.to_dict()


# =============================================================================
# SECTION 4 — _is_valid_event() and _collect_valid_events()
# =============================================================================

class TestEventValidation:

    def test_4_1_valid_event_accepted(self):
        assert len(_collect_valid_events([_ev()])) == 1

    def test_4_2_missing_field_rejected(self):
        ev = {"fatigue_index": 0.5, "sprint_load": 4000, "recovery_hours": 24}
        assert _is_valid_event(ev) is False

    def test_4_3_nonfinite_fatigue_rejected(self):
        ev = _ev(fi=float("nan"))
        assert _is_valid_event(ev) is False

    def test_4_4_non_numeric_type_rejected(self):
        ev = {"fatigue_index": "high", "sprint_load": 4000,
              "recovery_hours": 24, "injury_flag": 0}
        assert _is_valid_event(ev) is False

    def test_4_5_non_dict_rejected(self):
        assert _is_valid_event("not_a_dict") is False
        assert _is_valid_event(42)           is False

    def test_4_6_none_rejected(self):
        assert _is_valid_event(None) is False

    def test_4_7_mix_only_valid_counted(self):
        invalids = [{"missing": "field"}, None, "bad"]
        valid    = [_ev(fi=0.4), _ev(fi=0.6)]
        result   = _collect_valid_events(invalids + valid)
        assert len(result) == 2

    def test_4_8_max_events_cap(self):
        events = [_ev()] * (MAX_EVENTS + 50)
        result = _collect_valid_events(events)
        assert len(result) == MAX_EVENTS


# =============================================================================
# SECTION 5 — _compute_event_summary()
# =============================================================================

class TestComputeEventSummary:

    def test_5_1_empty_uses_defaults(self):
        s = _compute_event_summary([])
        assert s.mean_fatigue  == DEFAULT_FATIGUE
        assert s.mean_load     == DEFAULT_LOAD
        assert s.mean_recovery == DEFAULT_RECOVERY
        assert s.event_count   == 0

    def test_5_2_single_event_means_equal_values(self):
        ev = _ev(fi=0.6, sl=5_000.0, rh=18.0)
        s  = _compute_event_summary([ev])
        assert s.mean_fatigue  == pytest.approx(0.6,     abs=1e-12)
        assert s.mean_load     == pytest.approx(5_000.0, abs=1e-12)
        assert s.mean_recovery == pytest.approx(18.0,    abs=1e-12)

    def test_5_3_two_events_arithmetic_mean(self):
        evs = [_ev(fi=0.4, sl=2_000.0, rh=20.0),
               _ev(fi=0.8, sl=6_000.0, rh=28.0)]
        s = _compute_event_summary(evs)
        assert s.mean_fatigue  == pytest.approx(0.6,     abs=1e-12)
        assert s.mean_load     == pytest.approx(4_000.0, abs=1e-12)
        assert s.mean_recovery == pytest.approx(24.0,    abs=1e-12)

    def test_5_4_mean_fatigue_clamped_to_unit(self):
        # Supply pre-validated events with fatigue already in [0,1]
        s = _compute_event_summary([_ev(fi=1.0)] * 5)
        assert s.mean_fatigue <= 1.0

    def test_5_5_norm_recovery_clamped_to_unit(self):
        s = _compute_event_summary([_ev(rh=1000.0)])  # huge recovery
        assert s.norm_recovery <= 1.0

    def test_5_6_norm_load_unbounded_for_large_load(self):
        s = _compute_event_summary([_ev(sl=160_000.0)])  # 20× reference
        assert s.norm_load > 1.0

    def test_5_7_event_count_correct(self):
        evs = [_ev()] * 7
        assert _compute_event_summary(evs).event_count == 7

    def test_5_8_does_not_mutate_input(self):
        evs    = [_ev(fi=0.5, sl=4_000.0, rh=24.0)]
        before = copy.deepcopy(evs)
        _compute_event_summary(evs)
        assert evs == before

    def test_5_9_all_summary_fields_finite(self):
        s = _compute_event_summary([_ev(fi=0.7, sl=7_000.0, rh=12.0)])
        for attr in ("mean_fatigue", "mean_load", "mean_recovery",
                     "norm_load", "norm_recovery"):
            assert math.isfinite(getattr(s, attr)), f"{attr} not finite"

    def test_5_10_default_norm_recovery_is_one(self):
        s = _compute_event_summary([])
        assert s.norm_recovery == pytest.approx(1.0, abs=1e-9)   # 24/24


# =============================================================================
# SECTION 6 — _step_fatigue()
# =============================================================================

class TestStepFatigue:

    def test_6_1_positive_delta_raises(self):
        assert _step_fatigue(0.4, 0.1) == pytest.approx(0.5, abs=1e-12)

    def test_6_2_negative_delta_lowers(self):
        assert _step_fatigue(0.5, -0.2) == pytest.approx(0.3, abs=1e-12)

    def test_6_3_zero_delta_identity(self):
        assert _step_fatigue(0.6, 0.0) == pytest.approx(0.6, abs=1e-12)

    def test_6_4_large_positive_clamped_to_one(self):
        assert _step_fatigue(0.8, 5.0) == 1.0

    def test_6_5_large_negative_clamped_to_zero(self):
        assert _step_fatigue(0.2, -5.0) == 0.0

    def test_6_6_nonfinite_input_returns_lo(self):
        assert _step_fatigue(float("nan"), 0.1) == 0.0

    def test_6_7_result_always_finite(self):
        for f, d in [(0.0, 0.0), (0.5, 0.3), (1.0, -0.5), (0.3, float("nan"))]:
            assert math.isfinite(_step_fatigue(f, d))


# =============================================================================
# SECTION 7 — _project_fatigue()
# =============================================================================

class TestProjectFatigue:

    def test_7_1_returns_tuple_of_three(self):
        result = _project_fatigue(_summary())
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_7_2_all_values_in_unit_interval(self):
        f24, f48, f72 = _project_fatigue(_summary(fi=0.5, sl=4_000.0, rh=12.0))
        for v in (f24, f48, f72):
            assert 0.0 <= v <= 1.0

    def test_7_3_all_values_finite(self):
        for v in _project_fatigue(_summary(fi=0.8, sl=8_000.0, rh=4.0)):
            assert math.isfinite(v)

    def test_7_4_positive_delta_monotone_increasing(self):
        # High load, zero recovery → positive delta
        f24, f48, f72 = _project_fatigue(_summary(fi=0.3, sl=8_000.0, rh=0.0))
        assert f48 >= f24 and f72 >= f48

    def test_7_5_negative_delta_monotone_decreasing(self):
        # Zero load, full recovery → negative delta
        f24, f48, f72 = _project_fatigue(_summary(fi=0.8, sl=0.0, rh=24.0))
        assert f48 <= f24 and f72 <= f48

    def test_7_6_stepwise_derivation(self):
        s   = _summary(fi=0.4, sl=4_000.0, rh=20.0)
        f24, f48, f72 = _project_fatigue(s)
        delta = ALPHA_LOAD * s.norm_load - BETA_RECOVERY * s.norm_recovery
        assert f48 == pytest.approx(_step_fatigue(f24, delta), abs=1e-12)
        assert f72 == pytest.approx(_step_fatigue(f48, delta), abs=1e-12)

    def test_7_7_zero_load_full_recovery_zero_fatigue_stays_zero(self):
        f24, f48, f72 = _project_fatigue(_summary(fi=0.0, sl=0.0, rh=24.0))
        assert f24 == 0.0
        assert f48 == 0.0
        assert f72 == 0.0

    def test_7_8_manual_formula_cross_check(self):
        fi, sl, rh = 0.5, 4_000.0, 24.0
        s    = _summary(fi=fi, sl=sl, rh=rh)
        f24, f48, f72 = _project_fatigue(s)
        delta   = ALPHA_LOAD * (sl / LOAD_REFERENCE) - BETA_RECOVERY * 1.0
        exp_24  = max(0.0, min(1.0, fi + delta))
        exp_48  = max(0.0, min(1.0, exp_24 + delta))
        exp_72  = max(0.0, min(1.0, exp_48 + delta))
        assert f24 == pytest.approx(exp_24, abs=1e-12)
        assert f48 == pytest.approx(exp_48, abs=1e-12)
        assert f72 == pytest.approx(exp_72, abs=1e-12)

    def test_7_9_default_summary_all_horizons_zero(self):
        s = _compute_event_summary([])
        f24, f48, f72 = _project_fatigue(s)
        assert f24 == 0.0
        assert f48 == 0.0
        assert f72 == 0.0


# =============================================================================
# SECTION 8 — forecast_fatigue()
# =============================================================================

class TestForecastFatigue:

    def test_8_1_returns_dict(self):
        assert isinstance(_eng().forecast_fatigue([]), dict)

    def test_8_2_exactly_three_keys(self):
        d = _eng().forecast_fatigue([])
        assert set(d) == {"fatigue_24h", "fatigue_48h", "fatigue_72h"}

    def test_8_3_all_values_in_unit_interval(self):
        d = _eng().forecast_fatigue(_evlist(20))
        for k, v in d.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_8_4_all_values_finite(self):
        for v in _eng().forecast_fatigue(_evlist(10)).values():
            assert math.isfinite(v)

    def test_8_5_empty_events_all_zero(self):
        d = _eng().forecast_fatigue([])
        assert d["fatigue_24h"] == 0.0
        assert d["fatigue_48h"] == 0.0
        assert d["fatigue_72h"] == 0.0

    def test_8_6_all_invalid_same_as_empty(self):
        invalids = [{"bad": "data"}, None, "string"]
        d_inv  = _eng().forecast_fatigue(invalids)
        d_empt = _eng().forecast_fatigue([])
        assert d_inv == d_empt

    def test_8_7_high_load_zero_recovery_fatigue_does_not_decrease(self):
        evs = _evlist(10, fi=0.8, sl=8_000.0, rh=0.0)
        d   = _eng().forecast_fatigue(evs)
        assert d["fatigue_24h"] >= 0.8

    def test_8_8_zero_load_full_recovery_fatigue_decreases(self):
        evs = _evlist(10, fi=0.8, sl=0.0, rh=24.0)
        d   = _eng().forecast_fatigue(evs)
        assert d["fatigue_24h"] <= 0.8

    def test_8_9_positive_delta_monotone_increasing(self):
        evs = _evlist(10, fi=0.3, sl=8_000.0, rh=0.0)
        d   = _eng().forecast_fatigue(evs)
        assert d["fatigue_48h"] >= d["fatigue_24h"]
        assert d["fatigue_72h"] >= d["fatigue_48h"]

    def test_8_10_negative_delta_monotone_decreasing(self):
        evs = _evlist(10, fi=0.9, sl=0.0, rh=24.0)
        d   = _eng().forecast_fatigue(evs)
        assert d["fatigue_48h"] <= d["fatigue_24h"]
        assert d["fatigue_72h"] <= d["fatigue_48h"]

    def test_8_11_json_serialisable(self):
        json.dumps(_eng().forecast_fatigue(_evlist(5)))

    def test_8_12_does_not_mutate_events(self):
        evs    = _evlist(5)
        before = copy.deepcopy(evs)
        _eng().forecast_fatigue(evs)
        assert evs == before


# =============================================================================
# SECTION 9 — forecast_recovery()
# =============================================================================

class TestForecastRecovery:

    def test_9_1_returns_float(self):
        assert isinstance(_eng().forecast_recovery([]), float)

    def test_9_2_result_in_unit_interval(self):
        r = _eng().forecast_recovery(_evlist(10))
        assert 0.0 <= r <= 1.0

    def test_9_3_result_finite(self):
        assert math.isfinite(_eng().forecast_recovery(_evlist(5)))

    def test_9_4_empty_events_returns_one(self):
        assert _eng().forecast_recovery([]) == 1.0

    def test_9_5_high_fatigue_lower_recovery(self):
        r_high = _eng().forecast_recovery(_evlist(10, fi=0.9, sl=8_000.0, rh=0.0))
        r_low  = _eng().forecast_recovery(_evlist(10, fi=0.1, sl=1_000.0, rh=24.0))
        assert r_high < r_low

    def test_9_6_low_fatigue_higher_recovery(self):
        r = _eng().forecast_recovery(_evlist(10, fi=0.05, sl=0.0, rh=24.0))
        assert r >= 0.9

    def test_9_7_formula_cross_check(self):
        evs = _evlist(1, fi=0.5, sl=4_000.0, rh=24.0)
        r   = _eng().forecast_recovery(evs)
        f24 = _eng().forecast_fatigue(evs)["fatigue_24h"]
        assert r == pytest.approx(max(0.0, min(1.0, 1.0 - f24)), abs=1e-12)

    def test_9_8_consistent_with_full_forecast(self):
        evs = _evlist(10, fi=0.6, sl=5_000.0, rh=16.0)
        assert _eng().forecast_recovery(evs) == pytest.approx(
            _eng().forecast(evs)["recovery_forecast"], abs=1e-12
        )

    def test_9_9_json_serialisable(self):
        json.dumps({"recovery": _eng().forecast_recovery(_evlist(5))})


# =============================================================================
# SECTION 10 — forecast_injury_risk()
# =============================================================================

class TestForecastInjuryRisk:

    def test_10_1_returns_float(self):
        assert isinstance(_eng().forecast_injury_risk([]), float)

    def test_10_2_result_in_unit_interval(self):
        r = _eng().forecast_injury_risk(_evlist(10))
        assert 0.0 <= r <= 1.0

    def test_10_3_result_finite(self):
        assert math.isfinite(_eng().forecast_injury_risk(_evlist(5)))

    def test_10_4_empty_events_default_value(self):
        # Default: fi=0, rec=24 → f24=0; norm_rec=1 → ir = 0.6*0 + 0.4*(1-1) = 0
        assert _eng().forecast_injury_risk([]) == pytest.approx(0.0, abs=1e-9)

    def test_10_5_manual_formula_cross_check(self):
        evs = _evlist(1, fi=0.5, sl=4_000.0, rh=24.0)
        ir  = _eng().forecast_injury_risk(evs)
        f24 = _eng().forecast_fatigue(evs)["fatigue_24h"]
        norm_rec = min(1.0, 24.0 / 24.0)
        expected = max(0.0, min(1.0, W_FATIGUE * f24 + W_RECOVERY_DEFICIT * (1.0 - norm_rec)))
        assert ir == pytest.approx(expected, abs=1e-12)

    def test_10_6_high_fatigue_no_recovery_high_risk(self):
        evs = _evlist(10, fi=0.9, sl=8_000.0, rh=0.0)
        assert _eng().forecast_injury_risk(evs) >= 0.7

    def test_10_7_zero_fatigue_full_recovery_min_risk(self):
        evs = _evlist(10, fi=0.0, sl=0.0, rh=24.0)
        assert _eng().forecast_injury_risk(evs) == pytest.approx(0.0, abs=1e-9)

    def test_10_8_result_clamped_at_one(self):
        evs = _evlist(10, fi=1.0, sl=0.0, rh=0.0)
        assert _eng().forecast_injury_risk(evs) <= 1.0

    def test_10_9_consistent_with_full_forecast(self):
        evs = _evlist(10, fi=0.7, sl=6_000.0, rh=10.0)
        assert _eng().forecast_injury_risk(evs) == pytest.approx(
            _eng().forecast(evs)["injury_risk_forecast"], abs=1e-12
        )

    def test_10_10_does_not_mutate_events(self):
        evs    = _evlist(5)
        before = copy.deepcopy(evs)
        _eng().forecast_injury_risk(evs)
        assert evs == before


# =============================================================================
# SECTION 11 — compute_readiness()
# =============================================================================

class TestComputeReadiness:

    def test_11_1_returns_float(self):
        assert isinstance(_eng().compute_readiness([]), float)

    def test_11_2_result_in_unit_interval(self):
        r = _eng().compute_readiness(_evlist(10))
        assert 0.0 <= r <= 1.0

    def test_11_3_result_finite(self):
        assert math.isfinite(_eng().compute_readiness(_evlist(5)))

    def test_11_4_empty_events_returns_one(self):
        # Default: f24=0, ir=0 → readiness = 1-0-0*0.5 = 1.0
        assert _eng().compute_readiness([]) == pytest.approx(1.0, abs=1e-9)

    def test_11_5_manual_formula_cross_check(self):
        evs = _evlist(1, fi=0.5, sl=4_000.0, rh=24.0)
        rdy = _eng().compute_readiness(evs)
        f24 = _eng().forecast_fatigue(evs)["fatigue_24h"]
        ir  = _eng().forecast_injury_risk(evs)
        expected = max(0.0, min(1.0, 1.0 - f24 - ir * READINESS_INJURY_WEIGHT))
        assert rdy == pytest.approx(expected, abs=1e-12)

    def test_11_6_high_fatigue_high_risk_low_readiness(self):
        evs = _evlist(10, fi=0.9, sl=8_000.0, rh=0.0)
        assert _eng().compute_readiness(evs) <= 0.3

    def test_11_7_low_fatigue_low_risk_high_readiness(self):
        evs = _evlist(10, fi=0.05, sl=0.0, rh=24.0)
        assert _eng().compute_readiness(evs) >= 0.9

    def test_11_8_consistent_with_full_forecast(self):
        evs = _evlist(10, fi=0.6, sl=5_000.0, rh=16.0)
        assert _eng().compute_readiness(evs) == pytest.approx(
            _eng().forecast(evs)["readiness_score"], abs=1e-12
        )

    def test_11_9_result_always_gte_zero(self):
        for fi, sl, rh in [(1.0, 8_000.0, 0.0), (0.8, 7_000.0, 2.0), (0.0, 0.0, 0.0)]:
            assert _eng().compute_readiness(_evlist(5, fi=fi, sl=sl, rh=rh)) >= 0.0


# =============================================================================
# SECTION 12 — forecast() output structure
# =============================================================================

class TestForecastOutputStructure:

    _KEYS = {"fatigue_24h", "fatigue_48h", "fatigue_72h",
             "recovery_forecast", "injury_risk_forecast", "readiness_score"}

    def test_12_1_returns_dict(self):
        assert isinstance(_eng().forecast([]), dict)

    def test_12_2_exactly_six_keys(self):
        assert set(_eng().forecast([])) == self._KEYS

    def test_12_3_all_values_in_unit_interval(self):
        r = _eng().forecast(_evlist(20))
        for k, v in r.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_12_4_all_values_finite(self):
        for v in _eng().forecast(_evlist(10)).values():
            assert math.isfinite(v)

    def test_12_5_all_values_native_float(self):
        for v in _eng().forecast(_evlist(5)).values():
            assert isinstance(v, float), f"got {type(v)}"

    def test_12_6_json_serialisable(self):
        json.dumps(_eng().forecast(_evlist(10)))

    def test_12_7_empty_events_known_outputs(self):
        r = _eng().forecast([])
        assert r["fatigue_24h"]   == 0.0
        assert r["fatigue_48h"]   == 0.0
        assert r["fatigue_72h"]   == 0.0
        assert r["readiness_score"] == pytest.approx(1.0, abs=1e-9)

    def test_12_8_all_invalid_same_as_empty(self):
        invalids = [None, "bad", {"missing": True}]
        assert _eng().forecast(invalids) == _eng().forecast([])

    def test_12_9_extra_event_fields_ignored(self):
        ev = _ev()
        ev["extra_sensor"] = 99.9
        r = _eng().forecast([ev] * 5)
        assert set(r) == self._KEYS

    def test_12_10_single_valid_event_valid_output(self):
        r = _eng().forecast([_ev(fi=0.6, sl=5_000.0, rh=18.0, inj=0)])
        assert all(0.0 <= v <= 1.0 for v in r.values())


# =============================================================================
# SECTION 13 — forecast() formula correctness
# =============================================================================

class TestForecastFormulaCorrectness:

    def _manual(self, fi=0.5, sl=4_000.0, rh=24.0):
        """Return a complete manually-computed forecast for one event."""
        norm_load = sl / LOAD_REFERENCE
        norm_rec  = min(1.0, rh / 24.0)
        delta     = ALPHA_LOAD * norm_load - BETA_RECOVERY * norm_rec
        f24 = max(0.0, min(1.0, fi + delta))
        f48 = max(0.0, min(1.0, f24 + delta))
        f72 = max(0.0, min(1.0, f48 + delta))
        ir  = max(0.0, min(1.0, W_FATIGUE * f24 + W_RECOVERY_DEFICIT * (1.0 - norm_rec)))
        rec = max(0.0, min(1.0, 1.0 - f24))
        rdy = max(0.0, min(1.0, 1.0 - f24 - ir * READINESS_INJURY_WEIGHT))
        return {"fatigue_24h": f24, "fatigue_48h": f48, "fatigue_72h": f72,
                "recovery_forecast": rec, "injury_risk_forecast": ir,
                "readiness_score": rdy}

    def test_13_1_fatigue_24h_formula(self):
        evs = [_ev(fi=0.5, sl=4_000.0, rh=24.0)]
        r   = _eng().forecast(evs)
        assert r["fatigue_24h"] == pytest.approx(self._manual()["fatigue_24h"], abs=1e-12)

    def test_13_2_fatigue_48h_derived_from_24h(self):
        evs = [_ev(fi=0.5, sl=4_000.0, rh=12.0)]
        r   = _eng().forecast(evs)
        norm_rec = 12.0 / 24.0
        delta    = ALPHA_LOAD * (4_000.0 / LOAD_REFERENCE) - BETA_RECOVERY * norm_rec
        expected_48 = max(0.0, min(1.0, r["fatigue_24h"] + delta))
        assert r["fatigue_48h"] == pytest.approx(expected_48, abs=1e-12)

    def test_13_3_fatigue_72h_derived_from_48h(self):
        evs = [_ev(fi=0.4, sl=6_000.0, rh=8.0)]
        r   = _eng().forecast(evs)
        norm_rec = min(1.0, 8.0 / 24.0)
        delta    = ALPHA_LOAD * (6_000.0 / LOAD_REFERENCE) - BETA_RECOVERY * norm_rec
        expected_72 = max(0.0, min(1.0, r["fatigue_48h"] + delta))
        assert r["fatigue_72h"] == pytest.approx(expected_72, abs=1e-12)

    def test_13_4_recovery_forecast_formula(self):
        evs = [_ev(fi=0.6, sl=5_000.0, rh=16.0)]
        r   = _eng().forecast(evs)
        expected = max(0.0, min(1.0, 1.0 - r["fatigue_24h"]))
        assert r["recovery_forecast"] == pytest.approx(expected, abs=1e-12)

    def test_13_5_injury_risk_formula(self):
        evs  = [_ev(fi=0.5, sl=4_000.0, rh=16.0)]
        r    = _eng().forecast(evs)
        norm_rec = min(1.0, 16.0 / 24.0)
        expected = max(0.0, min(1.0,
                       W_FATIGUE * r["fatigue_24h"]
                       + W_RECOVERY_DEFICIT * (1.0 - norm_rec)))
        assert r["injury_risk_forecast"] == pytest.approx(expected, abs=1e-12)

    def test_13_6_readiness_formula(self):
        evs = [_ev(fi=0.5, sl=4_000.0, rh=20.0)]
        r   = _eng().forecast(evs)
        expected = max(0.0, min(1.0,
                       1.0 - r["fatigue_24h"]
                       - r["injury_risk_forecast"] * READINESS_INJURY_WEIGHT))
        assert r["readiness_score"] == pytest.approx(expected, abs=1e-12)

    def test_13_7_multi_event_mean_used(self):
        evs = [_ev(fi=0.4, sl=2_000.0, rh=20.0),
               _ev(fi=0.6, sl=6_000.0, rh=28.0)]
        r   = _eng().forecast(evs)
        m   = self._manual(fi=0.5, sl=4_000.0, rh=24.0)
        assert r["fatigue_24h"] == pytest.approx(m["fatigue_24h"], abs=1e-12)

    def test_13_8_clamp_prevents_fatigue_above_one(self):
        evs = _evlist(10, fi=1.0, sl=80_000.0, rh=0.0)
        r   = _eng().forecast(evs)
        assert r["fatigue_24h"] <= 1.0
        assert r["fatigue_48h"] <= 1.0
        assert r["fatigue_72h"] <= 1.0

    def test_13_9_clamp_prevents_fatigue_below_zero(self):
        evs = _evlist(10, fi=0.0, sl=0.0, rh=240.0)
        r   = _eng().forecast(evs)
        assert r["fatigue_24h"] >= 0.0
        assert r["fatigue_48h"] >= 0.0
        assert r["fatigue_72h"] >= 0.0


# =============================================================================
# SECTION 14 — Determinism
# =============================================================================

class TestDeterminism:

    def test_14_1_hundred_identical_calls(self):
        evs   = _make_probe_events(40)
        first = _eng().forecast(evs)
        for _ in range(99):
            assert _eng().forecast(evs) == first

    def test_14_2_two_independent_engines_identical(self):
        evs = _make_probe_events(30)
        assert (
            PerformanceForecastEngine().forecast(evs)
            == PerformanceForecastEngine().forecast(evs)
        )

    def test_14_3_deterministic_check_passes(self):
        assert _eng().deterministic_check() is True

    def test_14_4_call_order_independence(self):
        evs1 = _evlist(5, fi=0.3)
        evs2 = _evlist(5, fi=0.7)
        r_before = _eng().forecast(evs1)
        _eng().forecast(evs2)               # different call "before"
        r_after  = _eng().forecast(evs1)
        assert r_before == r_after

    def test_14_5_stateless_across_multiple_calls(self):
        eng  = PerformanceForecastEngine()
        evs  = _evlist(10, fi=0.5)
        r1   = eng.forecast(evs)
        eng.forecast(_evlist(10, fi=0.9))  # different call on same instance
        r2   = eng.forecast(evs)
        assert r1 == r2


# =============================================================================
# SECTION 15 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_15_1_forecast_does_not_mutate_event_list(self):
        evs    = _evlist(10)
        before = copy.deepcopy(evs)
        _eng().forecast(evs)
        assert evs == before

    def test_15_2_forecast_does_not_mutate_event_dicts(self):
        ev = _ev(fi=0.6, sl=5_000.0, rh=18.0)
        before = copy.deepcopy(ev)
        _eng().forecast([ev] * 5)
        assert ev == before

    def test_15_3_forecast_fatigue_does_not_mutate(self):
        evs    = _evlist(5)
        before = copy.deepcopy(evs)
        _eng().forecast_fatigue(evs)
        assert evs == before

    def test_15_4_forecast_recovery_does_not_mutate(self):
        evs    = _evlist(5)
        before = copy.deepcopy(evs)
        _eng().forecast_recovery(evs)
        assert evs == before


# =============================================================================
# SECTION 16 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_16_1_forecast_passes_json_dumps(self):
        json.dumps(_eng().forecast(_evlist(10)))

    def test_16_2_full_round_trip(self):
        evs  = _evlist(10, fi=0.6, sl=5_000.0, rh=16.0)
        r    = _eng().forecast(evs)
        back = json.loads(json.dumps(r))
        for k in r:
            assert abs(back[k] - r[k]) < 1e-9

    def test_16_3_forecast_result_to_dict_serialisable(self):
        fr = ForecastResult(0.3, 0.35, 0.38, 0.70, 0.25, 0.60)
        json.dumps(fr.to_dict())

    def test_16_4_all_values_native_float(self):
        r = _eng().forecast(_evlist(5))
        for v in r.values():
            assert isinstance(v, float)

    def test_16_5_forecast_fatigue_serialisable(self):
        json.dumps(_eng().forecast_fatigue(_evlist(5)))


# =============================================================================
# SECTION 17 — self_test()
# =============================================================================

class TestSelfTest:

    def test_17_1_returns_dict(self):
        assert isinstance(_eng().self_test(), dict)

    def test_17_2_required_keys(self):
        st = _eng().self_test()
        assert {"engine", "version", "checks", "passed"} <= set(st)

    def test_17_3_six_checks(self):
        assert len(_eng().self_test()["checks"]) == 6

    def test_17_4_each_check_has_required_fields(self):
        for c in _eng().self_test()["checks"]:
            assert {"name", "passed", "detail"} <= set(c)

    def test_17_5_all_checks_pass(self):
        st       = _eng().self_test()
        failures = [c["name"] for c in st["checks"] if not c["passed"]]
        assert st["passed"] is True
        assert failures == [], f"Failed: {failures}"

    def test_17_6_engine_name_correct(self):
        assert _eng().self_test()["engine"] == ENGINE_NAME

    def test_17_7_version_correct(self):
        assert _eng().self_test()["version"] == ENGINE_VERSION

    def test_17_8_idempotent_calls(self):
        eng = _eng()
        st1 = eng.self_test()
        st2 = eng.self_test()
        assert st1["passed"] == st2["passed"]
        assert len(st1["checks"]) == len(st2["checks"])


# =============================================================================
# SECTION 18 — _clamp() helper
# =============================================================================

class TestClamp:

    def test_18_1_within_range_unchanged(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_18_2_below_lo_returns_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_18_3_above_hi_returns_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_18_4_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_18_5_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_18_6_boundary_values_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# =============================================================================
# SECTION 19 — _finite_or_zero() helper
# =============================================================================

class TestFiniteOrZero:

    def test_19_1_finite_unchanged(self):
        assert _finite_or_zero(0.42) == pytest.approx(0.42)

    def test_19_2_nan_returns_zero(self):
        assert _finite_or_zero(float("nan")) == 0.0

    def test_19_3_pos_inf_returns_zero(self):
        assert _finite_or_zero(float("inf")) == 0.0

    def test_19_4_neg_inf_returns_zero(self):
        assert _finite_or_zero(float("-inf")) == 0.0


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