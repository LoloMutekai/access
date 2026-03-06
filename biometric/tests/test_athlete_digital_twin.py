"""
A.C.C.E.S.S. — Athlete Digital Twin Test Suite (Phase 7.14)
tests/test_athlete_digital_twin.py

Full coverage of biometric/athlete_digital_twin.py

Coverage map
────────────
SECTION 1  — Module constants (14 tests)
  1.1   MAX_HISTORY == 5000
  1.2   LOAD_REFERENCE == 8000.0
  1.3   RECOVERY_REFERENCE == 24.0
  1.4   EMA_FATIGUE_RETAIN == 0.7
  1.5   EMA_FATIGUE_EVENT == 0.3
  1.6   EMA_FATIGUE weights sum to 1.0
  1.7   EMA_ADAPTATION_RETAIN == 0.9
  1.8   EMA_ADAPTATION_LOAD == 0.1
  1.9   EMA_ADAPTATION weights sum to 1.0
  1.10  W_FATIGUE_RISK == 0.6
  1.11  W_RECOVERY_DEFICIT_RISK == 0.4
  1.12  Injury risk weights sum to 1.0
  1.13  READINESS_INJURY_COEFF == 0.5
  1.14  ENGINE_NAME and ENGINE_VERSION correct

SECTION 2  — TwinEvent dataclass (14 tests)
  2.1   Instantiates with valid fields
  2.2   Frozen (FrozenInstanceError on reassignment)
  2.3   All fields stored exactly
  2.4   is_valid() True for well-formed event
  2.5   is_valid() False when fatigue_index non-finite
  2.6   is_valid() False when fatigue_index < 0
  2.7   is_valid() False when fatigue_index > 1
  2.8   is_valid() False when sprint_load < 0
  2.9   is_valid() False when sprint_load non-finite
  2.10  is_valid() False when recovery_hours < 0
  2.11  is_valid() False when injury_flag not in {0,1}
  2.12  norm_recovery property = clamp(recovery_hours/24, 0, 1)
  2.13  norm_load property = sprint_load / LOAD_REFERENCE
  2.14  to_dict() / from_dict() round-trip

SECTION 3  — TwinState dataclass (8 tests)
  3.1   Instantiates with valid fields
  3.2   Frozen
  3.3   is_valid() True for well-formed state
  3.4   is_valid() False when any float non-finite
  3.5   is_valid() False when any float out of [0,1]
  3.6   to_dict() has exactly seven expected keys
  3.7   to_dict() values match fields
  3.8   to_dict() is JSON-serialisable

SECTION 4  — AthleteDigitalTwin construction (12 tests)
  4.1   Default construction succeeds
  4.2   athlete_id stored correctly
  4.3   baseline_fatigue seeds fatigue_state
  4.4   baseline_load seeds adaptation_factor
  4.5   event_count == 0 at construction
  4.6   events property returns empty list
  4.7   All state values ∈ [0, 1] at construction
  4.8   All state values finite at construction
  4.9   Non-finite baseline_fatigue clamped to 0.0
  4.10  Non-finite baseline_load clamped to 0.0
  4.11  baseline_fatigue > 1 clamped to 1.0
  4.12  readiness_state and injury_risk_state initialised consistently

SECTION 5  — update() — happy path (12 tests)
  5.1   Valid event increments event_count by 1
  5.2   Stored TwinEvent fields match input
  5.3   fatigue_state changes after update
  5.4   injury_risk_state changes after update
  5.5   readiness_state changes after update
  5.6   adaptation_factor changes when load > 0
  5.7   Multiple updates accumulate in history
  5.8   Events stored in insertion order
  5.9   Extra keys in event dict silently ignored
  5.10  Integer numeric fields accepted (int type)
  5.11  State remains bounded [0,1] after 50 updates
  5.12  update() returns None

SECTION 6  — update() — invalid event rejection (10 tests)
  6.1   Non-dict input rejected, state unchanged
  6.2   None rejected
  6.3   Missing fatigue_index rejected
  6.4   Missing sprint_load rejected
  6.5   Missing recovery_hours rejected
  6.6   Missing injury_flag rejected
  6.7   Non-finite fatigue_index rejected
  6.8   Non-finite sprint_load rejected
  6.9   Non-finite recovery_hours rejected
  6.10  Non-finite injury_flag rejected

SECTION 7  — update() state formula correctness (16 tests)
  7.1   fatigue_state = EMA_RETAIN*prior + EMA_EVENT*fi (manual cross-check)
  7.2   fatigue_state clamped to [0, 1]
  7.3   High fi → fatigue_state increases
  7.4   Low fi → fatigue_state decreases
  7.5   adaptation_factor = EMA_ADAPT_RETAIN*prior + EMA_ADAPT_LOAD*norm_load
  7.6   adaptation_factor clamped to [0, 1]
  7.7   Zero sprint_load → adaptation_factor decreases toward 0
  7.8   injury_risk_state = W_FATIGUE*fatigue + W_RD*(1-norm_rec) (manual)
  7.9   injury_risk_state clamped to [0, 1]
  7.10  Full recovery (rec=24h) → lower injury risk than no recovery (rec=0)
  7.11  readiness_state = clamp(1 - fatigue - injury*0.5) (manual)
  7.12  readiness_state clamped to [0, 1]
  7.13  High fatigue → lower readiness
  7.14  injury_risk uses post-update fatigue (not prior)
  7.15  readiness uses post-update fatigue AND post-update injury_risk
  7.16  Two sequential updates produce consistent state chain

SECTION 8  — History management (10 tests)
  8.1   MAX_HISTORY+1 events → event_count == MAX_HISTORY
  8.2   Oldest event evicted (FIFO) when capacity exceeded
  8.3   Newest events retained after eviction
  8.4   Exactly MAX_HISTORY events → no eviction
  8.5   events property returns a copy (mutation does not affect engine)
  8.6   Each element of events is a TwinEvent
  8.7   Events insertion order matches update order
  8.8   History cleared by reset()
  8.9   New events accepted after reset()
  8.10  State unchanged after eviction (state is separate from history)

SECTION 9  — adaptation_factor update (8 tests)
  9.1   Starts at baseline_load (initial seed)
  9.2   Increases when norm_load > current adaptation_factor
  9.3   Decreases when load == 0 over many iterations
  9.4   Converges toward mean_load for constant input
  9.5   Never exceeds 1.0
  9.6   Never falls below 0.0
  9.7   Manual EMA formula cross-check (two steps)
  9.8   Large load (> LOAD_REFERENCE) clamped at 1.0

SECTION 10 — forecast() (12 tests)
  10.1  Returns dict
  10.2  Exactly 5 keys: fatigue_24h, fatigue_48h, fatigue_72h, injury_risk, readiness_score
  10.3  All values ∈ [0, 1]
  10.4  All values finite
  10.5  All values native Python floats
  10.6  Empty history → valid forecast (forecast engine defaults)
  10.7  JSON-serialisable output
  10.8  Consistent with direct PerformanceForecastEngine call on same history
  10.9  injury_risk key present (not injury_risk_forecast)
  10.10 High-fatigue event history → higher injury_risk forecast
  10.11 Low-fatigue event history → higher readiness_score
  10.12 Forecast does not mutate event history

SECTION 11 — simulate_training() (14 tests)
  11.1  Returns dict
  11.2  Exactly 5 output keys
  11.3  All values ∈ [0, 1]
  11.4  All values finite
  11.5  Original fatigue_state unchanged after simulation
  11.6  Original injury_risk_state unchanged after simulation
  11.7  Original readiness_state unchanged after simulation
  11.8  Original adaptation_factor unchanged after simulation
  11.9  Original event_count unchanged after simulation
  11.10 Non-finite load treated as 0.0
  11.11 Non-finite recovery_hours treated as 0.0
  11.12 High load + low recovery → higher fatigue_24h than baseline
  11.13 Output is JSON-serialisable
  11.14 Two identical calls produce identical results (determinism)

SECTION 12 — get_state() (8 tests)
  12.1  Returns dict
  12.2  Contains exactly the eight expected keys
  12.3  athlete_id matches
  12.4  fatigue_state matches
  12.5  injury_risk_state matches
  12.6  readiness_state matches
  12.7  event_count matches
  12.8  Output is JSON-serialisable

SECTION 13 — to_dict() / from_dict() (14 tests)
  13.1  to_dict() returns dict with "state" and "events" keys
  13.2  "events" is a list
  13.3  len("events") == event_count
  13.4  to_dict() is JSON-serialisable
  13.5  to_dict() is a fresh dict (mutation does not affect twin)
  13.6  from_dict() restores athlete_id
  13.7  from_dict() restores fatigue_state exactly
  13.8  from_dict() restores adaptation_factor exactly
  13.9  from_dict() restores event_count exactly
  13.10 from_dict() restores forecast() output
  13.11 Corrupt event in snapshot silently skipped
  13.12 Full json.dumps → json.loads → from_dict round-trip
  13.13 set_state() restores state variables (no history)
  13.14 from_dict() enforces MAX_HISTORY after restore

SECTION 14 — Determinism (8 tests)
  14.1  100 identical event sequences → identical final state
  14.2  deterministic_check() returns True
  14.3  Two independent twins, same events → same get_state()
  14.4  Two independent twins, same events → same forecast()
  14.5  Call-order independence: stateless between runs
  14.6  reset() + same events → same state as fresh twin
  14.7  simulate_training() with same inputs always gives same output
  14.8  from_dict(to_dict()) produces identical forecast

SECTION 15 — Input immutability (8 tests)
  15.1  update() does not mutate event dict
  15.2  update() does not mutate event list (when list passed as arg)
  15.3  from_dict() does not mutate the input dict
  15.4  simulate_training() does not mutate any external state
  15.5  events property is a copy
  15.6  to_dict() mutation does not affect internal state
  15.7  get_state() mutation does not affect internal state
  15.8  TwinEvent.to_dict() mutation does not affect the event

SECTION 16 — JSON serialisability (8 tests)
  16.1  to_dict() passes json.dumps
  16.2  get_state() passes json.dumps
  16.3  forecast() output passes json.dumps
  16.4  simulate_training() output passes json.dumps
  16.5  Full to_dict → json.dumps → json.loads → from_dict round-trip
  16.6  All get_state() float values are native Python floats
  16.7  All forecast() values are native Python floats
  16.8  TwinEvent.to_dict() passes json.dumps

SECTION 17 — self_test() (8 tests)
  17.1  Returns dict
  17.2  Contains "engine", "version", "checks", "passed" keys
  17.3  "checks" has exactly six items
  17.4  Each check has "name", "passed", "detail" keys
  17.5  All six checks pass
  17.6  "engine" == ENGINE_NAME
  17.7  "version" == ENGINE_VERSION
  17.8  Does not mutate the calling twin's state

SECTION 18 — _clamp() and _finite_or_zero() helpers (9 tests)
  18.1  _clamp() — within range unchanged
  18.2  _clamp() — below lo → lo
  18.3  _clamp() — above hi → hi
  18.4  _clamp() — NaN → lo
  18.5  _clamp() — +Inf → lo
  18.6  _clamp() — boundary values exact
  18.7  _finite_or_zero() — finite unchanged
  18.8  _finite_or_zero() — NaN → 0.0
  18.9  _finite_or_zero() — ±Inf → 0.0
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.athlete_digital_twin import (
    EMA_ADAPTATION_LOAD,
    EMA_ADAPTATION_RETAIN,
    EMA_FATIGUE_EVENT,
    EMA_FATIGUE_RETAIN,
    ENGINE_NAME,
    ENGINE_VERSION,
    LOAD_REFERENCE,
    MAX_HISTORY,
    READINESS_INJURY_COEFF,
    RECOVERY_REFERENCE,
    W_FATIGUE_RISK,
    W_RECOVERY_DEFICIT_RISK,
    AthleteDigitalTwin,
    TwinEvent,
    TwinState,
    _clamp,
    _compute_injury_risk,
    _compute_readiness,
    _finite_or_zero,
    _is_valid_event,
    _make_probe_events,
)
from biometric.performance_forecast_engine import PerformanceForecastEngine


# =============================================================================
# HELPERS
# =============================================================================

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


def _fresh(n: int = 0, fi: float = 0.5, sl: float = 4_000.0, rh: float = 20.0) -> AthleteDigitalTwin:
    """Return a twin pre-loaded with n identical events."""
    t = AthleteDigitalTwin()
    for _ in range(n):
        t.update(_ev(fi=fi, sl=sl, rh=rh))
    return t


def _twin_with_probe(n: int = 20) -> AthleteDigitalTwin:
    t = AthleteDigitalTwin()
    for ev in _make_probe_events(n):
        t.update(ev)
    return t


# =============================================================================
# SECTION 1 — Module constants
# =============================================================================

class TestConstants:

    def test_1_1_max_history(self):
        assert MAX_HISTORY == 5_000

    def test_1_2_load_reference(self):
        assert LOAD_REFERENCE == pytest.approx(8_000.0)

    def test_1_3_recovery_reference(self):
        assert RECOVERY_REFERENCE == pytest.approx(24.0)

    def test_1_4_ema_fatigue_retain(self):
        assert EMA_FATIGUE_RETAIN == pytest.approx(0.7)

    def test_1_5_ema_fatigue_event(self):
        assert EMA_FATIGUE_EVENT == pytest.approx(0.3)

    def test_1_6_ema_fatigue_weights_sum(self):
        assert EMA_FATIGUE_RETAIN + EMA_FATIGUE_EVENT == pytest.approx(1.0, abs=1e-9)

    def test_1_7_ema_adaptation_retain(self):
        assert EMA_ADAPTATION_RETAIN == pytest.approx(0.9)

    def test_1_8_ema_adaptation_load(self):
        assert EMA_ADAPTATION_LOAD == pytest.approx(0.1)

    def test_1_9_ema_adaptation_weights_sum(self):
        assert EMA_ADAPTATION_RETAIN + EMA_ADAPTATION_LOAD == pytest.approx(1.0, abs=1e-9)

    def test_1_10_w_fatigue_risk(self):
        assert W_FATIGUE_RISK == pytest.approx(0.6)

    def test_1_11_w_recovery_deficit_risk(self):
        assert W_RECOVERY_DEFICIT_RISK == pytest.approx(0.4)

    def test_1_12_injury_risk_weights_sum(self):
        assert W_FATIGUE_RISK + W_RECOVERY_DEFICIT_RISK == pytest.approx(1.0, abs=1e-9)

    def test_1_13_readiness_injury_coeff(self):
        assert READINESS_INJURY_COEFF == pytest.approx(0.5)

    def test_1_14_engine_metadata(self):
        assert ENGINE_NAME    == "AthleteDigitalTwin"
        assert ENGINE_VERSION == "7.14.0"


# =============================================================================
# SECTION 2 — TwinEvent dataclass
# =============================================================================

class TestTwinEvent:

    def _te(self, **kw) -> TwinEvent:
        defaults = dict(fatigue_index=0.5, sprint_load=4_000.0,
                        recovery_hours=24.0, injury_flag=0)
        defaults.update(kw)
        return TwinEvent(**defaults)

    def test_2_1_instantiation(self):
        assert isinstance(self._te(), TwinEvent)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            self._te().fatigue_index = 0.99  # type: ignore[misc]

    def test_2_3_fields_stored_exactly(self):
        te = TwinEvent(0.6, 5_000.0, 18.0, 1)
        assert te.fatigue_index  == 0.6
        assert te.sprint_load    == 5_000.0
        assert te.recovery_hours == 18.0
        assert te.injury_flag    == 1

    def test_2_4_is_valid_true(self):
        assert self._te().is_valid() is True

    def test_2_5_is_valid_nonfinite_fatigue(self):
        assert self._te(fatigue_index=float("nan")).is_valid() is False

    def test_2_6_is_valid_fatigue_below_zero(self):
        assert self._te(fatigue_index=-0.1).is_valid() is False

    def test_2_7_is_valid_fatigue_above_one(self):
        assert self._te(fatigue_index=1.001).is_valid() is False

    def test_2_8_is_valid_negative_sprint_load(self):
        assert self._te(sprint_load=-1.0).is_valid() is False

    def test_2_9_is_valid_nonfinite_sprint_load(self):
        assert self._te(sprint_load=float("inf")).is_valid() is False

    def test_2_10_is_valid_negative_recovery(self):
        assert self._te(recovery_hours=-0.5).is_valid() is False

    def test_2_11_is_valid_bad_injury_flag(self):
        assert self._te(injury_flag=2).is_valid() is False
        assert self._te(injury_flag=-1).is_valid() is False

    def test_2_12_norm_recovery_clamped(self):
        te_full  = TwinEvent(0.5, 4_000.0, 24.0, 0)
        te_over  = TwinEvent(0.5, 4_000.0, 100.0, 0)
        te_zero  = TwinEvent(0.5, 4_000.0, 0.0, 0)
        assert te_full.norm_recovery  == pytest.approx(1.0,  abs=1e-12)
        assert te_over.norm_recovery  == pytest.approx(1.0,  abs=1e-12)
        assert te_zero.norm_recovery  == pytest.approx(0.0,  abs=1e-12)
        assert TwinEvent(0.5, 4_000.0, 12.0, 0).norm_recovery == pytest.approx(0.5, abs=1e-12)

    def test_2_13_norm_load(self):
        te = TwinEvent(0.5, 8_000.0, 20.0, 0)
        assert te.norm_load == pytest.approx(1.0, abs=1e-12)
        te2 = TwinEvent(0.5, 4_000.0, 20.0, 0)
        assert te2.norm_load == pytest.approx(0.5, abs=1e-12)

    def test_2_14_to_dict_from_dict_roundtrip(self):
        te = self._te(fatigue_index=0.65, sprint_load=6_000.0,
                      recovery_hours=16.0, injury_flag=1)
        assert TwinEvent.from_dict(te.to_dict()) == te


# =============================================================================
# SECTION 3 — TwinState dataclass
# =============================================================================

class TestTwinState:

    def _ts(self, **kw) -> TwinState:
        defaults = dict(athlete_id="t", baseline_fatigue=0.2, baseline_load=0.3,
                        adaptation_factor=0.3, fatigue_state=0.2,
                        injury_risk_state=0.15, readiness_state=0.7)
        defaults.update(kw)
        return TwinState(**defaults)

    def test_3_1_instantiation(self):
        assert isinstance(self._ts(), TwinState)

    def test_3_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            self._ts().fatigue_state = 0.99  # type: ignore[misc]

    def test_3_3_is_valid_true(self):
        assert self._ts().is_valid() is True

    def test_3_4_is_valid_nonfinite_field(self):
        assert self._ts(fatigue_state=float("nan")).is_valid() is False

    def test_3_5_is_valid_field_out_of_unit(self):
        assert self._ts(readiness_state=1.5).is_valid() is False
        assert self._ts(injury_risk_state=-0.1).is_valid() is False

    def test_3_6_to_dict_seven_keys(self):
        expected = {"athlete_id", "baseline_fatigue", "baseline_load",
                    "adaptation_factor", "fatigue_state",
                    "injury_risk_state", "readiness_state"}
        assert set(self._ts().to_dict()) == expected

    def test_3_7_to_dict_values_match(self):
        ts = self._ts()
        d  = ts.to_dict()
        assert d["fatigue_state"]     == ts.fatigue_state
        assert d["readiness_state"]   == ts.readiness_state
        assert d["adaptation_factor"] == ts.adaptation_factor

    def test_3_8_to_dict_json_serialisable(self):
        json.dumps(self._ts().to_dict())


# =============================================================================
# SECTION 4 — AthleteDigitalTwin construction
# =============================================================================

class TestConstruction:

    def test_4_1_default_construction(self):
        assert isinstance(AthleteDigitalTwin(), AthleteDigitalTwin)

    def test_4_2_athlete_id_stored(self):
        assert AthleteDigitalTwin("carol_99").athlete_id == "carol_99"

    def test_4_3_baseline_fatigue_seeds_fatigue_state(self):
        t = AthleteDigitalTwin(baseline_fatigue=0.35)
        assert t.fatigue_state == pytest.approx(0.35, abs=1e-12)

    def test_4_4_baseline_load_seeds_adaptation_factor(self):
        t = AthleteDigitalTwin(baseline_load=0.25)
        assert t.adaptation_factor == pytest.approx(0.25, abs=1e-12)

    def test_4_5_event_count_zero(self):
        assert AthleteDigitalTwin().event_count == 0

    def test_4_6_events_empty_list(self):
        assert AthleteDigitalTwin().events == []

    def test_4_7_all_state_in_unit_interval(self):
        t = AthleteDigitalTwin(baseline_fatigue=0.5, baseline_load=0.3)
        for v in (t.fatigue_state, t.injury_risk_state,
                  t.readiness_state, t.adaptation_factor):
            assert 0.0 <= v <= 1.0

    def test_4_8_all_state_finite(self):
        t = AthleteDigitalTwin(baseline_fatigue=0.6, baseline_load=0.4)
        for v in (t.fatigue_state, t.injury_risk_state,
                  t.readiness_state, t.adaptation_factor):
            assert math.isfinite(v)

    def test_4_9_nonfinite_baseline_fatigue_clamped(self):
        t = AthleteDigitalTwin(baseline_fatigue=float("nan"))
        assert t.fatigue_state == 0.0

    def test_4_10_nonfinite_baseline_load_clamped(self):
        # _finite_or_zero(inf) → 0.0, then clamp(0.0, 0, 1) = 0.0
        t = AthleteDigitalTwin(baseline_load=float("inf"))
        assert t.adaptation_factor == 0.0

    def test_4_11_baseline_fatigue_above_one_clamped(self):
        t = AthleteDigitalTwin(baseline_fatigue=5.0)
        assert t.fatigue_state <= 1.0

    def test_4_12_initial_risk_and_readiness_consistent(self):
        t = AthleteDigitalTwin(baseline_fatigue=0.4)
        # injury_risk and readiness must be consistent with initial fatigue
        expected_ir  = _compute_injury_risk(t.fatigue_state, 1.0)
        expected_rdy = _compute_readiness(t.fatigue_state, expected_ir)
        assert abs(t.injury_risk_state - expected_ir)  < 1e-12
        assert abs(t.readiness_state   - expected_rdy) < 1e-12


# =============================================================================
# SECTION 5 — update() — happy path
# =============================================================================

class TestUpdateHappyPath:

    def test_5_1_event_count_increments(self):
        t = AthleteDigitalTwin()
        t.update(_ev())
        assert t.event_count == 1

    def test_5_2_stored_event_fields_match(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.6, sl=5_000.0, rh=18.0, inj=1))
        ev = t.events[0]
        assert ev.fatigue_index  == pytest.approx(0.6,     abs=1e-12)
        assert ev.sprint_load    == pytest.approx(5_000.0, abs=1e-12)
        assert ev.recovery_hours == pytest.approx(18.0,    abs=1e-12)
        assert ev.injury_flag    == 1

    def test_5_3_fatigue_state_changes(self):
        t = AthleteDigitalTwin()
        f_before = t.fatigue_state
        t.update(_ev(fi=0.8))
        assert t.fatigue_state != f_before

    def test_5_4_injury_risk_state_changes(self):
        t = AthleteDigitalTwin()
        ir_before = t.injury_risk_state
        t.update(_ev(fi=0.8, rh=0.0))
        assert t.injury_risk_state != ir_before

    def test_5_5_readiness_state_changes(self):
        t = AthleteDigitalTwin()
        rdy_before = t.readiness_state
        t.update(_ev(fi=0.8, rh=0.0))
        assert t.readiness_state != rdy_before

    def test_5_6_adaptation_factor_changes_with_load(self):
        t = AthleteDigitalTwin()
        a_before = t.adaptation_factor
        t.update(_ev(sl=8_000.0))
        assert t.adaptation_factor != a_before

    def test_5_7_multiple_updates_accumulate(self):
        t = AthleteDigitalTwin()
        for i in range(10):
            t.update(_ev(fi=0.4 + 0.02 * i))
        assert t.event_count == 10

    def test_5_8_insertion_order_preserved(self):
        t = AthleteDigitalTwin()
        for sl in (1_000.0, 3_000.0, 5_000.0):
            t.update(_ev(sl=sl))
        assert [e.sprint_load for e in t.events] == [1_000.0, 3_000.0, 5_000.0]

    def test_5_9_extra_keys_silently_ignored(self):
        t = AthleteDigitalTwin()
        ev = _ev()
        ev["extra_sensor"] = 99.9
        t.update(ev)
        assert t.event_count == 1

    def test_5_10_integer_numeric_fields_accepted(self):
        t = AthleteDigitalTwin()
        t.update({"fatigue_index": 0, "sprint_load": 4000,
                  "recovery_hours": 24, "injury_flag": 0})
        assert t.event_count == 1

    def test_5_11_state_bounded_after_many_updates(self):
        t = AthleteDigitalTwin()
        for ev in _make_probe_events(50):
            t.update(ev)
        for v in (t.fatigue_state, t.injury_risk_state,
                  t.readiness_state, t.adaptation_factor):
            assert 0.0 <= v <= 1.0

    def test_5_12_update_returns_none(self):
        t = AthleteDigitalTwin()
        assert t.update(_ev()) is None


# =============================================================================
# SECTION 6 — update() — invalid event rejection
# =============================================================================

class TestUpdateInvalidRejection:

    def _count_and_fatigue(self, ev) -> tuple:
        t = AthleteDigitalTwin()
        t.update(_ev())                   # valid baseline event
        f_before = t.fatigue_state
        ec_before = t.event_count
        t.update(ev)                      # candidate
        return t.event_count, t.fatigue_state, ec_before, f_before

    def test_6_1_non_dict_rejected(self):
        ec, f, ec_b, f_b = self._count_and_fatigue("not_a_dict")
        assert ec == ec_b and f == f_b

    def test_6_2_none_rejected(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(None)
        assert ec == ec_b and f == f_b

    def test_6_3_missing_fatigue_index(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            {"sprint_load": 4000, "recovery_hours": 20, "injury_flag": 0})
        assert ec == ec_b

    def test_6_4_missing_sprint_load(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            {"fatigue_index": 0.5, "recovery_hours": 20, "injury_flag": 0})
        assert ec == ec_b

    def test_6_5_missing_recovery_hours(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            {"fatigue_index": 0.5, "sprint_load": 4000, "injury_flag": 0})
        assert ec == ec_b

    def test_6_6_missing_injury_flag(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            {"fatigue_index": 0.5, "sprint_load": 4000, "recovery_hours": 20})
        assert ec == ec_b

    def test_6_7_nonfinite_fatigue_index(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            _ev(fi=float("nan")))
        assert ec == ec_b and f == f_b

    def test_6_8_nonfinite_sprint_load(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            _ev(sl=float("inf")))
        assert ec == ec_b

    def test_6_9_nonfinite_recovery_hours(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            _ev(rh=float("nan")))
        assert ec == ec_b

    def test_6_10_nonfinite_injury_flag(self):
        ec, f, ec_b, f_b = self._count_and_fatigue(
            {"fatigue_index": 0.5, "sprint_load": 4000,
             "recovery_hours": 20, "injury_flag": float("nan")})
        assert ec == ec_b


# =============================================================================
# SECTION 7 — update() state formula correctness
# =============================================================================

class TestUpdateFormulas:

    def test_7_1_fatigue_ema_manual(self):
        t = AthleteDigitalTwin()
        f0 = t.fatigue_state
        t.update(_ev(fi=0.7))
        expected = EMA_FATIGUE_RETAIN * f0 + EMA_FATIGUE_EVENT * 0.7
        assert t.fatigue_state == pytest.approx(expected, abs=1e-12)

    def test_7_2_fatigue_state_clamped(self):
        t = AthleteDigitalTwin(baseline_fatigue=1.0)
        t.update(_ev(fi=1.0))
        assert t.fatigue_state <= 1.0

    def test_7_3_high_fi_raises_fatigue(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.01))   # very low
        f_low = t.fatigue_state
        t2 = AthleteDigitalTwin()
        t2.update(_ev(fi=0.99))  # very high
        assert t2.fatigue_state > f_low

    def test_7_4_low_fi_from_high_base_decreases_fatigue(self):
        t = AthleteDigitalTwin(baseline_fatigue=0.8)
        f_before = t.fatigue_state
        t.update(_ev(fi=0.01))
        assert t.fatigue_state < f_before

    def test_7_5_adaptation_ema_manual(self):
        t = AthleteDigitalTwin()
        a0 = t.adaptation_factor
        t.update(_ev(sl=4_000.0))
        norm_load = 4_000.0 / LOAD_REFERENCE
        expected  = EMA_ADAPTATION_RETAIN * a0 + EMA_ADAPTATION_LOAD * norm_load
        assert t.adaptation_factor == pytest.approx(expected, abs=1e-12)

    def test_7_6_adaptation_clamped(self):
        t = AthleteDigitalTwin()
        for _ in range(50):
            t.update(_ev(sl=800_000.0))  # extreme load
        assert t.adaptation_factor <= 1.0

    def test_7_7_zero_load_adaptation_decays(self):
        t = AthleteDigitalTwin(baseline_load=0.8)
        a0 = t.adaptation_factor
        for _ in range(10):
            t.update(_ev(sl=0.0))
        assert t.adaptation_factor < a0

    def test_7_8_injury_risk_manual(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.6, rh=12.0))
        norm_rec  = 12.0 / RECOVERY_REFERENCE
        f_state   = t.fatigue_state
        expected  = W_FATIGUE_RISK * f_state + W_RECOVERY_DEFICIT_RISK * (1.0 - norm_rec)
        assert t.injury_risk_state == pytest.approx(expected, abs=1e-12)

    def test_7_9_injury_risk_clamped(self):
        t = AthleteDigitalTwin(baseline_fatigue=1.0)
        t.update(_ev(fi=1.0, rh=0.0))
        assert t.injury_risk_state <= 1.0

    def test_7_10_full_recovery_lowers_risk(self):
        t_full = AthleteDigitalTwin()
        t_full.update(_ev(fi=0.7, rh=24.0))
        t_zero = AthleteDigitalTwin()
        t_zero.update(_ev(fi=0.7, rh=0.0))
        assert t_full.injury_risk_state < t_zero.injury_risk_state

    def test_7_11_readiness_manual(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.5, rh=20.0))
        expected = max(0.0, 1.0 - t.fatigue_state - t.injury_risk_state * READINESS_INJURY_COEFF)
        assert t.readiness_state == pytest.approx(expected, abs=1e-12)

    def test_7_12_readiness_clamped(self):
        t = AthleteDigitalTwin(baseline_fatigue=1.0)
        t.update(_ev(fi=1.0, rh=0.0))
        assert t.readiness_state >= 0.0

    def test_7_13_high_fatigue_lowers_readiness(self):
        t_high = AthleteDigitalTwin()
        t_high.update(_ev(fi=0.9))
        t_low  = AthleteDigitalTwin()
        t_low.update(_ev(fi=0.1))
        assert t_high.readiness_state < t_low.readiness_state

    def test_7_14_injury_risk_uses_post_update_fatigue(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.8, rh=0.0))
        # injury_risk must be computed from the already-updated fatigue_state
        f_new    = t.fatigue_state
        norm_rec = 0.0 / RECOVERY_REFERENCE
        expected = W_FATIGUE_RISK * f_new + W_RECOVERY_DEFICIT_RISK * (1.0 - norm_rec)
        assert t.injury_risk_state == pytest.approx(expected, abs=1e-12)

    def test_7_15_readiness_uses_updated_fatigue_and_risk(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.6, rh=16.0))
        expected = max(0.0, 1.0 - t.fatigue_state - t.injury_risk_state * READINESS_INJURY_COEFF)
        assert t.readiness_state == pytest.approx(expected, abs=1e-12)

    def test_7_16_two_sequential_updates_consistent(self):
        t = AthleteDigitalTwin()
        t.update(_ev(fi=0.4, sl=3_000.0, rh=20.0))
        f1 = t.fatigue_state
        t.update(_ev(fi=0.7, sl=6_000.0, rh=12.0))
        f2_expected = EMA_FATIGUE_RETAIN * f1 + EMA_FATIGUE_EVENT * 0.7
        assert t.fatigue_state == pytest.approx(f2_expected, abs=1e-12)


# =============================================================================
# SECTION 8 — History management
# =============================================================================

class TestHistoryManagement:

    def test_8_1_over_max_enforces_cap(self):
        t = _fresh(MAX_HISTORY + 3)
        assert t.event_count == MAX_HISTORY

    def test_8_2_oldest_event_evicted_fifo(self):
        t = AthleteDigitalTwin()
        t.update(_ev(sl=1_000.0))       # sl=1000 → will be evicted
        for i in range(MAX_HISTORY):
            t.update(_ev(sl=5_000.0))
        assert t.events[0].sprint_load == pytest.approx(5_000.0, abs=1e-9)

    def test_8_3_newest_retained_after_eviction(self):
        t = AthleteDigitalTwin()
        for i in range(MAX_HISTORY + 5):
            t.update(_ev(sl=float(i + 1)))
        assert t.events[-1].sprint_load == pytest.approx(float(MAX_HISTORY + 5), abs=1e-9)

    def test_8_4_exactly_max_no_eviction(self):
        t = AthleteDigitalTwin()
        t.update(_ev(sl=111.0))         # first event — should survive
        for _ in range(MAX_HISTORY - 1):
            t.update(_ev(sl=5_000.0))
        assert t.events[0].sprint_load == pytest.approx(111.0, abs=1e-9)

    def test_8_5_events_property_is_copy(self):
        t = _fresh(5)
        copy_list = t.events
        copy_list.clear()
        assert t.event_count == 5

    def test_8_6_elements_are_twin_events(self):
        t = _fresh(3)
        assert all(isinstance(e, TwinEvent) for e in t.events)

    def test_8_7_order_matches_update_order(self):
        t = AthleteDigitalTwin()
        for sl in (1_000.0, 2_000.0, 3_000.0):
            t.update(_ev(sl=sl))
        assert [e.sprint_load for e in t.events] == [1_000.0, 2_000.0, 3_000.0]

    def test_8_8_reset_clears_history(self):
        t = _fresh(10)
        t.reset()
        assert t.event_count == 0

    def test_8_9_events_accepted_after_reset(self):
        t = _fresh(5)
        t.reset()
        t.update(_ev())
        assert t.event_count == 1

    def test_8_10_state_independent_of_eviction(self):
        """Eviction does not retroactively change current state variables."""
        t = AthleteDigitalTwin()
        for _ in range(MAX_HISTORY + 1):
            t.update(_ev(fi=0.5, sl=4_000.0, rh=20.0))
        f = t.fatigue_state
        # Adding one more event triggers another eviction; fatigue EMA not reset
        t.update(_ev(fi=0.5, sl=4_000.0, rh=20.0))
        assert t.fatigue_state == pytest.approx(
            EMA_FATIGUE_RETAIN * f + EMA_FATIGUE_EVENT * 0.5, abs=1e-12
        )


# =============================================================================
# SECTION 9 — adaptation_factor update
# =============================================================================

class TestAdaptationFactor:

    def test_9_1_starts_at_baseline_load(self):
        assert AthleteDigitalTwin(baseline_load=0.45).adaptation_factor == pytest.approx(0.45)

    def test_9_2_increases_when_load_above_current(self):
        t = AthleteDigitalTwin(baseline_load=0.0)
        t.update(_ev(sl=8_000.0))    # norm_load=1.0 > 0
        assert t.adaptation_factor > 0.0

    def test_9_3_decreases_toward_zero_with_zero_load(self):
        t = AthleteDigitalTwin(baseline_load=0.8)
        a0 = t.adaptation_factor
        for _ in range(20):
            t.update(_ev(sl=0.0))
        assert t.adaptation_factor < a0

    def test_9_4_converges_for_constant_load(self):
        """With constant normalised load L, factor converges to L."""
        norm_target = 0.5   # 4000/8000
        t = AthleteDigitalTwin(baseline_load=0.0)
        for _ in range(200):
            t.update(_ev(sl=4_000.0))
        # EMA converges toward norm_target; accept within 0.02 after 200 steps
        assert abs(t.adaptation_factor - norm_target) < 0.02

    def test_9_5_never_exceeds_one(self):
        t = AthleteDigitalTwin()
        for _ in range(30):
            t.update(_ev(sl=800_000.0))
        assert t.adaptation_factor <= 1.0

    def test_9_6_never_below_zero(self):
        t = AthleteDigitalTwin(baseline_load=0.5)
        for _ in range(30):
            t.update(_ev(sl=0.0))
        assert t.adaptation_factor >= 0.0

    def test_9_7_manual_two_step_cross_check(self):
        t = AthleteDigitalTwin(baseline_load=0.0)
        t.update(_ev(sl=4_000.0))
        a1 = EMA_ADAPTATION_RETAIN * 0.0 + EMA_ADAPTATION_LOAD * (4_000.0 / LOAD_REFERENCE)
        assert t.adaptation_factor == pytest.approx(a1, abs=1e-12)
        t.update(_ev(sl=4_000.0))
        a2 = EMA_ADAPTATION_RETAIN * a1 + EMA_ADAPTATION_LOAD * (4_000.0 / LOAD_REFERENCE)
        assert t.adaptation_factor == pytest.approx(a2, abs=1e-12)

    def test_9_8_large_load_clamped(self):
        t = AthleteDigitalTwin()
        t.update(_ev(sl=10_000_000.0))  # huge: norm_load >> 1
        assert t.adaptation_factor <= 1.0


# =============================================================================
# SECTION 10 — forecast()
# =============================================================================

class TestForecast:

    _KEYS = {"fatigue_24h", "fatigue_48h", "fatigue_72h",
             "injury_risk", "readiness_score"}

    def test_10_1_returns_dict(self):
        assert isinstance(AthleteDigitalTwin().forecast(), dict)

    def test_10_2_exactly_five_keys(self):
        assert set(AthleteDigitalTwin().forecast()) == self._KEYS

    def test_10_3_all_values_in_unit_interval(self):
        t = _twin_with_probe(20)
        for k, v in t.forecast().items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_10_4_all_values_finite(self):
        for v in _twin_with_probe(15).forecast().values():
            assert math.isfinite(v)

    def test_10_5_all_values_native_float(self):
        for v in _twin_with_probe(10).forecast().values():
            assert isinstance(v, float)

    def test_10_6_empty_history_valid_forecast(self):
        f = AthleteDigitalTwin().forecast()
        assert all(0.0 <= v <= 1.0 for v in f.values())

    def test_10_7_json_serialisable(self):
        json.dumps(_twin_with_probe(10).forecast())

    def test_10_8_consistent_with_forecast_engine(self):
        t   = _twin_with_probe(15)
        eng = PerformanceForecastEngine()
        raw = eng.forecast(t._events_as_forecast_input())
        f   = t.forecast()
        assert f["fatigue_24h"]    == pytest.approx(raw["fatigue_24h"],          abs=1e-12)
        assert f["injury_risk"]    == pytest.approx(raw["injury_risk_forecast"],  abs=1e-12)
        assert f["readiness_score"]== pytest.approx(raw["readiness_score"],       abs=1e-12)

    def test_10_9_injury_risk_key_present_not_injury_risk_forecast(self):
        f = _twin_with_probe(5).forecast()
        assert "injury_risk" in f
        assert "injury_risk_forecast" not in f

    def test_10_10_high_fatigue_history_high_injury_risk(self):
        t = AthleteDigitalTwin()
        for _ in range(20):
            t.update(_ev(fi=0.9, sl=8_000.0, rh=2.0))
        t_low = AthleteDigitalTwin()
        for _ in range(20):
            t_low.update(_ev(fi=0.1, sl=500.0, rh=24.0))
        assert t.forecast()["injury_risk"] > t_low.forecast()["injury_risk"]

    def test_10_11_low_fatigue_high_readiness(self):
        t = AthleteDigitalTwin()
        for _ in range(20):
            t.update(_ev(fi=0.05, sl=0.0, rh=24.0))
        assert t.forecast()["readiness_score"] >= 0.8

    def test_10_12_forecast_does_not_mutate_history(self):
        t    = _twin_with_probe(10)
        ec   = t.event_count
        evs  = t.events
        t.forecast()
        assert t.event_count == ec
        assert t.events == evs


# =============================================================================
# SECTION 11 — simulate_training()
# =============================================================================

class TestSimulateTraining:

    _KEYS = {"fatigue_24h", "fatigue_48h", "fatigue_72h",
             "injury_risk", "readiness_score"}

    def _ready_twin(self) -> AthleteDigitalTwin:
        t = AthleteDigitalTwin()
        for ev in _make_probe_events(10):
            t.update(ev)
        return t

    def test_11_1_returns_dict(self):
        assert isinstance(self._ready_twin().simulate_training(4_000.0, 20.0), dict)

    def test_11_2_exactly_five_keys(self):
        assert set(self._ready_twin().simulate_training(4_000.0, 20.0)) == self._KEYS

    def test_11_3_all_values_in_unit_interval(self):
        sim = self._ready_twin().simulate_training(6_000.0, 12.0)
        for k, v in sim.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_11_4_all_values_finite(self):
        sim = self._ready_twin().simulate_training(4_000.0, 20.0)
        for v in sim.values():
            assert math.isfinite(v)

    def test_11_5_original_fatigue_state_unchanged(self):
        t = self._ready_twin()
        f = t.fatigue_state
        t.simulate_training(6_000.0, 12.0)
        assert t.fatigue_state == f

    def test_11_6_original_injury_risk_unchanged(self):
        t = self._ready_twin()
        ir = t.injury_risk_state
        t.simulate_training(6_000.0, 12.0)
        assert t.injury_risk_state == ir

    def test_11_7_original_readiness_unchanged(self):
        t = self._ready_twin()
        rdy = t.readiness_state
        t.simulate_training(6_000.0, 12.0)
        assert t.readiness_state == rdy

    def test_11_8_original_adaptation_unchanged(self):
        t = self._ready_twin()
        af = t.adaptation_factor
        t.simulate_training(6_000.0, 12.0)
        assert t.adaptation_factor == af

    def test_11_9_original_event_count_unchanged(self):
        t = self._ready_twin()
        ec = t.event_count
        t.simulate_training(6_000.0, 12.0)
        assert t.event_count == ec

    def test_11_10_nonfinite_load_treated_as_zero(self):
        t = self._ready_twin()
        sim = t.simulate_training(float("nan"), 20.0)
        assert all(0.0 <= v <= 1.0 for v in sim.values())

    def test_11_11_nonfinite_recovery_treated_as_zero(self):
        t = self._ready_twin()
        sim = t.simulate_training(4_000.0, float("inf"))
        assert all(0.0 <= v <= 1.0 for v in sim.values())

    def test_11_12_high_load_low_recovery_worse_than_rest(self):
        t1 = self._ready_twin()
        t2 = self._ready_twin()
        sim_hard = t1.simulate_training(8_000.0, 0.0)
        sim_rest  = t2.simulate_training(0.0, 24.0)
        assert sim_hard["fatigue_24h"] >= sim_rest["fatigue_24h"]

    def test_11_13_output_json_serialisable(self):
        json.dumps(self._ready_twin().simulate_training(4_000.0, 20.0))

    def test_11_14_two_identical_calls_identical(self):
        t = self._ready_twin()
        assert (t.simulate_training(5_000.0, 16.0)
                == t.simulate_training(5_000.0, 16.0))


# =============================================================================
# SECTION 12 — get_state()
# =============================================================================

class TestGetState:

    def test_12_1_returns_dict(self):
        assert isinstance(AthleteDigitalTwin().get_state(), dict)

    def test_12_2_eight_required_keys(self):
        expected = {"athlete_id", "baseline_fatigue", "baseline_load",
                    "adaptation_factor", "fatigue_state", "injury_risk_state",
                    "readiness_state", "event_count"}
        assert set(AthleteDigitalTwin().get_state()) == expected

    def test_12_3_athlete_id_matches(self):
        t = AthleteDigitalTwin("probe_id")
        assert t.get_state()["athlete_id"] == "probe_id"

    def test_12_4_fatigue_state_matches(self):
        t = _twin_with_probe(5)
        assert t.get_state()["fatigue_state"] == t.fatigue_state

    def test_12_5_injury_risk_state_matches(self):
        t = _twin_with_probe(5)
        assert t.get_state()["injury_risk_state"] == t.injury_risk_state

    def test_12_6_readiness_state_matches(self):
        t = _twin_with_probe(5)
        assert t.get_state()["readiness_state"] == t.readiness_state

    def test_12_7_event_count_matches(self):
        t = _fresh(7)
        assert t.get_state()["event_count"] == 7

    def test_12_8_json_serialisable(self):
        json.dumps(_twin_with_probe(10).get_state())


# =============================================================================
# SECTION 13 — to_dict() / from_dict()
# =============================================================================

class TestToDictFromDict:

    def test_13_1_to_dict_has_state_and_events_keys(self):
        d = AthleteDigitalTwin().to_dict()
        assert "state" in d and "events" in d

    def test_13_2_events_is_list(self):
        assert isinstance(_fresh(5).to_dict()["events"], list)

    def test_13_3_events_len_matches_event_count(self):
        t = _fresh(8)
        assert len(t.to_dict()["events"]) == 8

    def test_13_4_to_dict_json_serialisable(self):
        json.dumps(_fresh(10).to_dict())

    def test_13_5_to_dict_fresh_dict(self):
        t = _fresh(3)
        d = t.to_dict()
        d["state"]["fatigue_state"] = 999.0
        assert t.fatigue_state != 999.0

    def test_13_6_from_dict_restores_athlete_id(self):
        t  = AthleteDigitalTwin("dana_42")
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        assert t2.athlete_id == "dana_42"

    def test_13_7_from_dict_restores_fatigue_state(self):
        t  = _twin_with_probe(15)
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        assert abs(t2.fatigue_state - t.fatigue_state) < 1e-12

    def test_13_8_from_dict_restores_adaptation_factor(self):
        t  = _twin_with_probe(15)
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        assert abs(t2.adaptation_factor - t.adaptation_factor) < 1e-12

    def test_13_9_from_dict_restores_event_count(self):
        t  = _fresh(12)
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        assert t2.event_count == t.event_count

    def test_13_10_from_dict_restores_forecast(self):
        t  = _twin_with_probe(20)
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        for k in t.forecast():
            assert t2.forecast()[k] == pytest.approx(t.forecast()[k], abs=1e-9)

    def test_13_11_corrupt_event_skipped(self):
        snap = AthleteDigitalTwin().to_dict()
        snap["events"] = [
            {"fatigue_index": 0.5, "sprint_load": 4_000.0,
             "recovery_hours": 20.0, "injury_flag": 0},
            {"corrupt": "data"},
        ]
        t = AthleteDigitalTwin.from_dict(snap)
        assert t.event_count == 1

    def test_13_12_json_round_trip(self):
        t  = _twin_with_probe(25)
        t2 = AthleteDigitalTwin.from_dict(json.loads(json.dumps(t.to_dict())))
        assert abs(t2.fatigue_state - t.fatigue_state) < 1e-9
        assert t2.event_count == t.event_count

    def test_13_13_set_state_restores_core_variables(self):
        t1 = _twin_with_probe(10)
        t2 = AthleteDigitalTwin()
        t2.set_state(t1.get_state())
        assert abs(t2.fatigue_state     - t1.fatigue_state)     < 1e-12
        assert abs(t2.injury_risk_state - t1.injury_risk_state) < 1e-12
        assert abs(t2.readiness_state   - t1.readiness_state)   < 1e-12

    def test_13_14_from_dict_enforces_max_history(self):
        d = AthleteDigitalTwin().to_dict()
        d["events"] = [
            {"fatigue_index": 0.5, "sprint_load": 4_000.0,
             "recovery_hours": 20.0, "injury_flag": 0}
            for _ in range(MAX_HISTORY + 20)
        ]
        t = AthleteDigitalTwin.from_dict(d)
        assert t.event_count == MAX_HISTORY


# =============================================================================
# SECTION 14 — Determinism
# =============================================================================

class TestDeterminism:

    def test_14_1_hundred_identical_sequences(self):
        evs   = _make_probe_events(30)
        def _run() -> dict:
            t = AthleteDigitalTwin()
            for ev in evs: t.update(ev)
            return t.get_state()
        first = _run()
        for _ in range(99):
            assert _run() == first

    def test_14_2_deterministic_check_passes(self):
        assert AthleteDigitalTwin().deterministic_check() is True

    def test_14_3_two_twins_same_events_same_state(self):
        evs = _make_probe_events(20)
        t1  = AthleteDigitalTwin()
        t2  = AthleteDigitalTwin()
        for ev in evs:
            t1.update(ev)
            t2.update(ev)
        assert t1.get_state() == t2.get_state()

    def test_14_4_two_twins_same_events_same_forecast(self):
        evs = _make_probe_events(15)
        t1  = AthleteDigitalTwin()
        t2  = AthleteDigitalTwin()
        for ev in evs:
            t1.update(ev)
            t2.update(ev)
        assert t1.forecast() == t2.forecast()

    def test_14_5_stateless_between_runs(self):
        evs = _make_probe_events(10)
        t   = AthleteDigitalTwin()
        for ev in evs: t.update(ev)
        r1  = t.forecast()
        r2  = t.forecast()
        assert r1 == r2

    def test_14_6_reset_plus_same_events_equals_fresh(self):
        evs = _make_probe_events(15)
        t1  = AthleteDigitalTwin()
        for ev in evs: t1.update(ev)
        t2  = AthleteDigitalTwin()
        for ev in evs: t2.update(ev)
        t2.reset()
        for ev in evs: t2.update(ev)
        assert t1.get_state() == t2.get_state()

    def test_14_7_simulate_training_deterministic(self):
        t = _twin_with_probe(15)
        assert (t.simulate_training(5_000.0, 18.0)
                == t.simulate_training(5_000.0, 18.0))

    def test_14_8_from_dict_to_dict_identical_forecast(self):
        t  = _twin_with_probe(20)
        t2 = AthleteDigitalTwin.from_dict(t.to_dict())
        assert t.forecast() == t2.forecast()


# =============================================================================
# SECTION 15 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_15_1_update_does_not_mutate_event_dict(self):
        ev     = _ev(fi=0.6, sl=5_000.0, rh=18.0)
        before = copy.deepcopy(ev)
        AthleteDigitalTwin().update(ev)
        assert ev == before

    def test_15_2_update_with_extra_fields_no_mutation(self):
        ev = _ev()
        ev["extra"] = "unchanged"
        AthleteDigitalTwin().update(ev)
        assert ev["extra"] == "unchanged"

    def test_15_3_from_dict_does_not_mutate_input(self):
        t    = _fresh(5)
        snap = t.to_dict()
        orig = copy.deepcopy(snap)
        AthleteDigitalTwin.from_dict(snap)
        assert snap == orig

    def test_15_4_simulate_training_no_external_mutation(self):
        t      = _twin_with_probe(10)
        f_snap = t.fatigue_state
        ec_snap = t.event_count
        t.simulate_training(6_000.0, 12.0)
        assert t.fatigue_state == f_snap
        assert t.event_count   == ec_snap

    def test_15_5_events_property_is_copy(self):
        t = _fresh(5)
        copy_list = t.events
        copy_list.clear()
        assert t.event_count == 5

    def test_15_6_to_dict_mutation_does_not_affect_state(self):
        t = _twin_with_probe(5)
        d = t.to_dict()
        d["state"]["fatigue_state"] = 0.0
        assert t.fatigue_state > 0.0   # twin untouched

    def test_15_7_get_state_mutation_does_not_affect_state(self):
        t = _twin_with_probe(5)
        s = t.get_state()
        s["fatigue_state"] = 0.0
        assert t.fatigue_state > 0.0

    def test_15_8_twin_event_to_dict_mutation_safe(self):
        te = TwinEvent(0.5, 4_000.0, 20.0, 0)
        d  = te.to_dict()
        d["fatigue_index"] = 999.0
        assert te.fatigue_index == 0.5


# =============================================================================
# SECTION 16 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_16_1_to_dict_passes_json_dumps(self):
        json.dumps(_twin_with_probe(10).to_dict())

    def test_16_2_get_state_passes_json_dumps(self):
        json.dumps(_twin_with_probe(10).get_state())

    def test_16_3_forecast_passes_json_dumps(self):
        json.dumps(_twin_with_probe(10).forecast())

    def test_16_4_simulate_training_passes_json_dumps(self):
        t = _twin_with_probe(10)
        json.dumps(t.simulate_training(4_000.0, 20.0))

    def test_16_5_full_round_trip(self):
        t  = _twin_with_probe(20)
        t2 = AthleteDigitalTwin.from_dict(json.loads(json.dumps(t.to_dict())))
        assert abs(t2.fatigue_state - t.fatigue_state) < 1e-9
        assert t2.event_count == t.event_count

    def test_16_6_get_state_float_values_are_native(self):
        s = _twin_with_probe(5).get_state()
        for k in ("fatigue_state", "injury_risk_state", "readiness_state", "adaptation_factor"):
            assert isinstance(s[k], float)

    def test_16_7_forecast_values_are_native_float(self):
        for v in _twin_with_probe(5).forecast().values():
            assert isinstance(v, float)

    def test_16_8_twin_event_to_dict_json_serialisable(self):
        json.dumps(TwinEvent(0.5, 4_000.0, 20.0, 0).to_dict())


# =============================================================================
# SECTION 17 — self_test()
# =============================================================================

class TestSelfTest:

    def test_17_1_returns_dict(self):
        assert isinstance(AthleteDigitalTwin().self_test(), dict)

    def test_17_2_required_keys(self):
        st = AthleteDigitalTwin().self_test()
        assert {"engine", "version", "checks", "passed"} <= set(st)

    def test_17_3_six_checks(self):
        assert len(AthleteDigitalTwin().self_test()["checks"]) == 6

    def test_17_4_each_check_has_required_fields(self):
        for c in AthleteDigitalTwin().self_test()["checks"]:
            assert {"name", "passed", "detail"} <= set(c)

    def test_17_5_all_checks_pass(self):
        st       = AthleteDigitalTwin().self_test()
        failures = [c["name"] for c in st["checks"] if not c["passed"]]
        assert st["passed"] is True
        assert failures == [], f"Failed: {failures}"

    def test_17_6_engine_name_correct(self):
        assert AthleteDigitalTwin().self_test()["engine"] == ENGINE_NAME

    def test_17_7_version_correct(self):
        assert AthleteDigitalTwin().self_test()["version"] == ENGINE_VERSION

    def test_17_8_does_not_mutate_live_state(self):
        t        = _twin_with_probe(10)
        f_before = t.fatigue_state
        ec_before = t.event_count
        t.self_test()
        assert t.fatigue_state == f_before
        assert t.event_count   == ec_before


# =============================================================================
# SECTION 18 — _clamp() and _finite_or_zero() helpers
# =============================================================================

class TestHelpers:

    def test_18_1_clamp_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_18_2_clamp_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_18_3_clamp_above_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_18_4_clamp_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_18_5_clamp_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_18_6_clamp_boundary_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_18_7_finite_or_zero_finite_unchanged(self):
        assert _finite_or_zero(0.42) == pytest.approx(0.42)

    def test_18_8_finite_or_zero_nan(self):
        assert _finite_or_zero(float("nan")) == 0.0

    def test_18_9_finite_or_zero_inf(self):
        assert _finite_or_zero(float("inf"))  == 0.0
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