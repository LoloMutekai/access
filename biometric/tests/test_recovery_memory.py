"""
A.C.C.E.S.S. — Recovery Memory Test Suite (Phase 7.10)
tests/test_recovery_memory.py

Full coverage of biometric/recovery_memory.py

Coverage map
────────────
SECTION 1  — Constants
  1.1   MAX_EVENTS == 200
  1.2   RECOVERY_FACTOR_MIN == 0.5
  1.3   RECOVERY_FACTOR_MAX == 1.5
  1.4   CORRECTED_HOURS_MIN == 4.0
  1.5   CORRECTED_HOURS_MAX == 96.0
  1.6   All constants are the expected Python types
  1.7   RECOVERY_FACTOR_MIN < RECOVERY_FACTOR_MAX
  1.8   CORRECTED_HOURS_MIN < CORRECTED_HOURS_MAX

SECTION 2  — RecoveryEvent dataclass
  2.1   Instantiates with valid fields
  2.2   Dataclass is frozen (FrozenInstanceError on reassignment)
  2.3   All fields stored exactly
  2.4   is_valid() True for well-formed event
  2.5   is_valid() False when predicted_hours is non-finite
  2.6   is_valid() False when predicted_hours == 0
  2.7   is_valid() False when predicted_hours < 0
  2.8   is_valid() False when actual_hours is non-finite
  2.9   is_valid() False when actual_hours == 0
  2.10  is_valid() False when actual_hours < 0
  2.11  is_valid() False when fatigue_index is non-finite
  2.12  is_valid() False when injury_risk is non-finite
  2.13  is_valid() False when timestamp is non-finite
  2.14  ratio property == actual_hours / predicted_hours

SECTION 3  — RecoveryEvent.to_dict()
  3.1   Returns a dict with exactly the five expected keys
  3.2   All values match the fields
  3.3   Returns a fresh dict (mutation does not affect instance)
  3.4   Output is JSON-serialisable
  3.5   All values are native Python floats

SECTION 4  — RecoveryEvent.from_dict()
  4.1   Reconstructs an identical event from to_dict() output
  4.2   Round-trip through json.dumps / json.loads preserves all values

SECTION 5  — RecoveryMemory construction
  5.1   Initialises with zero events
  5.2   Initial compute_recovery_factor() == 1.0
  5.3   events property returns a list
  5.4   event_count property == 0

SECTION 6  — RecoveryMemory.add_event() — happy path
  6.1   Valid event is stored
  6.2   event_count increments by 1 per add
  6.3   Stored event fields match arguments
  6.4   Two valid events are both stored
  6.5   Events stored in insertion order

SECTION 7  — RecoveryMemory.add_event() — invalid input rejection
  7.1   Non-finite predicted_hours silently ignored
  7.2   predicted_hours == 0 silently ignored
  7.3   predicted_hours < 0 silently ignored
  7.4   Non-finite actual_hours silently ignored
  7.5   actual_hours == 0 silently ignored
  7.6   actual_hours < 0 silently ignored
  7.7   Non-finite fatigue_index silently ignored
  7.8   Non-finite injury_risk silently ignored
  7.9   Non-finite timestamp silently ignored
  7.10  Mixed invalid/valid: valid events still accepted after invalid ones

SECTION 8  — RecoveryMemory.add_event() — capacity enforcement
  8.1   MAX_EVENTS+1 insertions → event_count == MAX_EVENTS
  8.2   Oldest event evicted (FIFO) when limit exceeded
  8.3   Newest events retained after eviction
  8.4   Exactly MAX_EVENTS insertions → no eviction
  8.5   Factor computed from retained events only

SECTION 9  — RecoveryMemory.compute_recovery_factor() — formula
  9.1   No events → returns 1.0
  9.2   Single event → factor == clamp(actual/predicted)
  9.3   Two events → factor == clamp(mean of two ratios)
  9.4   All-equal ratios → factor == that ratio (when in bounds)
  9.5   Factor is clamped high to RECOVERY_FACTOR_MAX
  9.6   Factor is clamped low to RECOVERY_FACTOR_MIN
  9.7   Factor == 1.0 when actual == predicted on average
  9.8   Returns float
  9.9   Returns finite value
  9.10  Returns value in [RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX]

SECTION 10 — compute_recovery_factor() — monotonicity
  10.1  Adding an event with ratio > current mean raises the factor
  10.2  Adding an event with ratio < current mean lowers the factor

SECTION 11 — RecoveryMemory.events property
  11.1  Returns a list
  11.2  Returned list is a copy — mutation does not affect internal state
  11.3  Each element is a RecoveryEvent
  11.4  Order matches insertion order

SECTION 12 — RecoveryMemory.reset()
  12.1  Clears all events
  12.2  event_count == 0 after reset
  12.3  compute_recovery_factor() == 1.0 after reset
  12.4  New events can be added after reset

SECTION 13 — RecoveryMemory.to_dict()
  13.1  Returns a dict
  13.2  Contains exactly {"events_count", "recovery_factor"}
  13.3  events_count matches event_count
  13.4  recovery_factor matches compute_recovery_factor()
  13.5  Output is JSON-serialisable
  13.6  events_count is int
  13.7  recovery_factor is float
  13.8  Returns a fresh dict (mutation does not affect engine)

SECTION 14 — RecoveryMemory.get_state() / set_state()
  14.1  get_state() returns a dict
  14.2  get_state() is JSON-serialisable
  14.3  get_state() contains "events" key
  14.4  get_state()["events"] is a list
  14.5  len(get_state()["events"]) == event_count
  14.6  set_state() restores all events
  14.7  set_state() restores event_count exactly
  14.8  set_state() restores compute_recovery_factor() exactly
  14.9  set_state() with corrupt event skips it silently
  14.10 set_state() with empty events list clears engine
  14.11 set_state() enforces MAX_EVENTS capacity after restore

SECTION 15 — apply_recovery_factor()
  15.1  Returns float
  15.2  Result is finite
  15.3  Result ∈ [CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX]
  15.4  factor=1.0 → result == predicted_hours (when in bounds)
  15.5  Correct multiplication: result == predicted_hours * factor (in bounds)
  15.6  Result clamped to CORRECTED_HOURS_MIN when product is too small
  15.7  Result clamped to CORRECTED_HOURS_MAX when product is too large
  15.8  Non-finite predicted_hours → CORRECTED_HOURS_MIN
  15.9  Non-finite factor → neutral (factor treated as 1.0)
  15.10 JSON-serialisable return value

SECTION 16 — Determinism
  16.1  100 identical add_event() sequences → identical factor
  16.2  deterministic_check() returns True
  16.3  call-order independence: factor depends only on stored events

SECTION 17 — Input immutability
  17.1  add_event() arguments are not mutated (float value semantics)
  17.2  set_state() does not mutate the input dict
  17.3  events property returns a copy (caller mutation does not affect engine)

SECTION 18 — JSON serialisability
  18.1  to_dict() passes json.dumps
  18.2  get_state() passes json.dumps
  18.3  Full get_state() → json.dumps → json.loads → set_state() round-trip
  18.4  to_dict() values are native Python types

SECTION 19 — self_test()
  19.1  Returns a dict
  19.2  Contains "engine", "version", "checks", "passed"
  19.3  "checks" has exactly five items
  19.4  Each check has "name", "passed", "detail"
  19.5  All five checks pass on a fresh engine
  19.6  Does not mutate the engine's own event list

SECTION 20 — _clamp() helper
  20.1  Value within range returned unchanged
  20.2  Value below lo → lo
  20.3  Value above hi → hi
  20.4  NaN → lo
  20.5  +Inf → lo
  20.6  Boundary values exact

SECTION 21 — _finite_or_zero() helper
  21.1  Finite value returned unchanged
  21.2  NaN → 0.0
  21.3  +Inf → 0.0
  21.4  -Inf → 0.0
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.recovery_memory import (
    CORRECTED_HOURS_MAX,
    CORRECTED_HOURS_MIN,
    MAX_EVENTS,
    RECOVERY_FACTOR_MAX,
    RECOVERY_FACTOR_MIN,
    RecoveryEvent,
    RecoveryMemory,
    _clamp,
    _finite_or_zero,
    apply_recovery_factor,
)


# =============================================================================
# HELPERS
# =============================================================================

def _event(
    pred: float = 40.0,
    act:  float = 36.0,
    fi:   float = 0.4,
    ir:   float = 0.2,
    ts:   float = 1_000.0,
) -> None:
    """Return keyword kwargs for add_event."""
    return dict(
        predicted_hours=pred, actual_hours=act,
        fatigue_index=fi, injury_risk=ir, timestamp=ts,
    )


def _add(m: RecoveryMemory, **kw) -> None:
    """Convenience wrapper around add_event using _event defaults."""
    defaults = _event()
    defaults.update(kw)
    m.add_event(**defaults)


def _fresh(n: int = 0, ratio: float = 1.0) -> RecoveryMemory:
    """Return a RecoveryMemory pre-filled with n identical events at the given ratio."""
    m = RecoveryMemory()
    for i in range(n):
        m.add_event(40.0, 40.0 * ratio, 0.4, 0.2, float(i + 1))
    return m


# =============================================================================
# SECTION 1 — Constants
# =============================================================================

class TestConstants:

    def test_1_1_max_events(self):
        assert MAX_EVENTS == 200

    def test_1_2_recovery_factor_min(self):
        assert RECOVERY_FACTOR_MIN == pytest.approx(0.5)

    def test_1_3_recovery_factor_max(self):
        assert RECOVERY_FACTOR_MAX == pytest.approx(1.5)

    def test_1_4_corrected_hours_min(self):
        assert CORRECTED_HOURS_MIN == pytest.approx(4.0)

    def test_1_5_corrected_hours_max(self):
        assert CORRECTED_HOURS_MAX == pytest.approx(96.0)

    def test_1_6_types(self):
        assert isinstance(MAX_EVENTS, int)
        for c in (RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX,
                  CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX):
            assert isinstance(c, float)

    def test_1_7_factor_min_lt_max(self):
        assert RECOVERY_FACTOR_MIN < RECOVERY_FACTOR_MAX

    def test_1_8_hours_min_lt_max(self):
        assert CORRECTED_HOURS_MIN < CORRECTED_HOURS_MAX


# =============================================================================
# SECTION 2 — RecoveryEvent dataclass
# =============================================================================

class TestRecoveryEventDataclass:

    def _ev(self, **kw):
        defaults = dict(predicted_hours=40.0, actual_hours=36.0,
                        fatigue_index=0.4, injury_risk=0.2, timestamp=1_000.0)
        defaults.update(kw)
        return RecoveryEvent(**defaults)

    def test_2_1_instantiation(self):
        assert isinstance(self._ev(), RecoveryEvent)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        ev = self._ev()
        with pytest.raises(FrozenInstanceError):
            ev.predicted_hours = 99.0   # type: ignore[misc]

    def test_2_3_fields_stored_exactly(self):
        ev = self._ev(predicted_hours=38.5, actual_hours=32.1,
                      fatigue_index=0.55, injury_risk=0.18, timestamp=1_234_567.0)
        assert ev.predicted_hours == 38.5
        assert ev.actual_hours    == 32.1
        assert ev.fatigue_index   == 0.55
        assert ev.injury_risk     == 0.18
        assert ev.timestamp       == 1_234_567.0

    def test_2_4_is_valid_true(self):
        assert self._ev().is_valid() is True

    def test_2_5_is_valid_nonfinite_predicted(self):
        assert self._ev(predicted_hours=float("inf")).is_valid() is False

    def test_2_6_is_valid_zero_predicted(self):
        assert self._ev(predicted_hours=0.0).is_valid() is False

    def test_2_7_is_valid_negative_predicted(self):
        assert self._ev(predicted_hours=-1.0).is_valid() is False

    def test_2_8_is_valid_nonfinite_actual(self):
        assert self._ev(actual_hours=float("nan")).is_valid() is False

    def test_2_9_is_valid_zero_actual(self):
        assert self._ev(actual_hours=0.0).is_valid() is False

    def test_2_10_is_valid_negative_actual(self):
        assert self._ev(actual_hours=-5.0).is_valid() is False

    def test_2_11_is_valid_nonfinite_fatigue(self):
        assert self._ev(fatigue_index=float("nan")).is_valid() is False

    def test_2_12_is_valid_nonfinite_injury(self):
        assert self._ev(injury_risk=float("inf")).is_valid() is False

    def test_2_13_is_valid_nonfinite_timestamp(self):
        assert self._ev(timestamp=float("nan")).is_valid() is False

    def test_2_14_ratio_property(self):
        ev = self._ev(predicted_hours=40.0, actual_hours=30.0)
        assert ev.ratio == pytest.approx(0.75, abs=1e-12)


# =============================================================================
# SECTION 3 — RecoveryEvent.to_dict()
# =============================================================================

class TestRecoveryEventToDict:

    def _ev(self):
        return RecoveryEvent(40.0, 36.0, 0.4, 0.2, 1_000.0)

    def test_3_1_exact_keys(self):
        assert set(self._ev().to_dict()) == {
            "predicted_hours", "actual_hours",
            "fatigue_index", "injury_risk", "timestamp",
        }

    def test_3_2_values_match_fields(self):
        ev = self._ev()
        d  = ev.to_dict()
        assert d["predicted_hours"] == ev.predicted_hours
        assert d["actual_hours"]    == ev.actual_hours
        assert d["fatigue_index"]   == ev.fatigue_index
        assert d["injury_risk"]     == ev.injury_risk
        assert d["timestamp"]       == ev.timestamp

    def test_3_3_fresh_dict_no_leakback(self):
        ev = self._ev()
        d  = ev.to_dict()
        d["predicted_hours"] = 999.0
        assert ev.predicted_hours != 999.0

    def test_3_4_json_serialisable(self):
        json.dumps(self._ev().to_dict())

    def test_3_5_native_float_values(self):
        for v in self._ev().to_dict().values():
            assert isinstance(v, float)


# =============================================================================
# SECTION 4 — RecoveryEvent.from_dict()
# =============================================================================

class TestRecoveryEventFromDict:

    def test_4_1_roundtrip_identical(self):
        ev = RecoveryEvent(38.0, 30.0, 0.5, 0.3, 2_000.0)
        assert RecoveryEvent.from_dict(ev.to_dict()) == ev

    def test_4_2_json_roundtrip(self):
        ev = RecoveryEvent(40.123, 36.456, 0.423, 0.189, 1_700_000_000.0)
        restored = RecoveryEvent.from_dict(json.loads(json.dumps(ev.to_dict())))
        assert abs(restored.predicted_hours - ev.predicted_hours) < 1e-9
        assert abs(restored.actual_hours    - ev.actual_hours)    < 1e-9
        assert abs(restored.fatigue_index   - ev.fatigue_index)   < 1e-9
        assert abs(restored.injury_risk     - ev.injury_risk)     < 1e-9
        assert abs(restored.timestamp       - ev.timestamp)       < 1e-9


# =============================================================================
# SECTION 5 — RecoveryMemory construction
# =============================================================================

class TestRecoveryMemoryConstruction:

    def test_5_1_starts_empty(self):
        assert RecoveryMemory().event_count == 0

    def test_5_2_initial_factor_neutral(self):
        assert RecoveryMemory().compute_recovery_factor() == 1.0

    def test_5_3_events_returns_list(self):
        assert isinstance(RecoveryMemory().events, list)

    def test_5_4_event_count_property_zero(self):
        assert RecoveryMemory().event_count == 0


# =============================================================================
# SECTION 6 — add_event() — happy path
# =============================================================================

class TestAddEventHappyPath:

    def test_6_1_valid_event_stored(self):
        m = RecoveryMemory()
        _add(m)
        assert m.event_count == 1

    def test_6_2_count_increments_per_add(self):
        m = RecoveryMemory()
        for i in range(1, 6):
            _add(m, timestamp=float(i))
            assert m.event_count == i

    def test_6_3_stored_fields_match_args(self):
        m = RecoveryMemory()
        m.add_event(predicted_hours=36.0, actual_hours=30.0,
                    fatigue_index=0.55, injury_risk=0.25, timestamp=9_999.0)
        ev = m.events[0]
        assert ev.predicted_hours == 36.0
        assert ev.actual_hours    == 30.0
        assert ev.fatigue_index   == 0.55
        assert ev.injury_risk     == 0.25
        assert ev.timestamp       == 9_999.0

    def test_6_4_two_events_both_stored(self):
        m = RecoveryMemory()
        _add(m, timestamp=1.0)
        _add(m, timestamp=2.0)
        assert m.event_count == 2

    def test_6_5_insertion_order_preserved(self):
        m = RecoveryMemory()
        for ts in (100.0, 200.0, 300.0):
            _add(m, timestamp=ts)
        assert [e.timestamp for e in m.events] == [100.0, 200.0, 300.0]


# =============================================================================
# SECTION 7 — add_event() — invalid input rejection
# =============================================================================

class TestAddEventInvalidRejection:

    def _add_and_count(self, **kw) -> int:
        m = RecoveryMemory()
        defaults = dict(predicted_hours=40.0, actual_hours=36.0,
                        fatigue_index=0.4, injury_risk=0.2, timestamp=1.0)
        defaults.update(kw)
        m.add_event(**defaults)
        return m.event_count

    def test_7_1_nan_predicted_ignored(self):
        assert self._add_and_count(predicted_hours=float("nan")) == 0

    def test_7_2_zero_predicted_ignored(self):
        assert self._add_and_count(predicted_hours=0.0) == 0

    def test_7_3_negative_predicted_ignored(self):
        assert self._add_and_count(predicted_hours=-10.0) == 0

    def test_7_4_nan_actual_ignored(self):
        assert self._add_and_count(actual_hours=float("nan")) == 0

    def test_7_5_zero_actual_ignored(self):
        assert self._add_and_count(actual_hours=0.0) == 0

    def test_7_6_negative_actual_ignored(self):
        assert self._add_and_count(actual_hours=-5.0) == 0

    def test_7_7_nan_fatigue_ignored(self):
        assert self._add_and_count(fatigue_index=float("nan")) == 0

    def test_7_8_nan_injury_ignored(self):
        assert self._add_and_count(injury_risk=float("inf")) == 0

    def test_7_9_nan_timestamp_ignored(self):
        assert self._add_and_count(timestamp=float("nan")) == 0

    def test_7_10_valid_after_invalid_accepted(self):
        m = RecoveryMemory()
        m.add_event(float("nan"), 30.0, 0.5, 0.2, 1.0)  # invalid
        _add(m, timestamp=2.0)                            # valid
        assert m.event_count == 1


# =============================================================================
# SECTION 8 — add_event() — capacity enforcement
# =============================================================================

class TestCapacityEnforcement:

    def test_8_1_over_limit_enforces_max(self):
        m = _fresh(MAX_EVENTS + 1)
        assert m.event_count == MAX_EVENTS

    def test_8_2_oldest_event_evicted(self):
        m = RecoveryMemory()
        m.add_event(40.0, 30.0, 0.4, 0.2, 0.0)  # timestamp=0 → oldest
        for i in range(1, MAX_EVENTS + 1):        # +MAX_EVENTS more
            m.add_event(40.0, 36.0, 0.4, 0.2, float(i))
        assert m.events[0].timestamp == 1.0       # ts=0 gone; ts=1 is now oldest

    def test_8_3_newest_events_retained(self):
        m = RecoveryMemory()
        for i in range(MAX_EVENTS + 5):
            m.add_event(40.0, 36.0, 0.4, 0.2, float(i))
        assert m.events[-1].timestamp == float(MAX_EVENTS + 4)

    def test_8_4_exactly_max_events_no_eviction(self):
        m = RecoveryMemory()
        for i in range(MAX_EVENTS):
            m.add_event(40.0, 36.0, 0.4, 0.2, float(i))
        assert m.events[0].timestamp == 0.0   # oldest still present

    def test_8_5_factor_from_retained_events(self):
        """After eviction, factor reflects only retained events."""
        m = RecoveryMemory()
        # First event has ratio = 2.0 (actual=80, pred=40) — will be evicted
        m.add_event(40.0, 80.0, 0.4, 0.2, 0.0)
        # Fill remaining MAX_EVENTS with ratio = 0.9
        for i in range(1, MAX_EVENTS + 1):
            m.add_event(40.0, 36.0, 0.4, 0.2, float(i))
        # All retained events have ratio 0.9 → factor = 0.9
        assert m.compute_recovery_factor() == pytest.approx(0.9, abs=1e-9)


# =============================================================================
# SECTION 9 — compute_recovery_factor() — formula
# =============================================================================

class TestComputeRecoveryFactor:

    def test_9_1_no_events_returns_one(self):
        assert RecoveryMemory().compute_recovery_factor() == 1.0

    def test_9_2_single_event_ratio(self):
        m = RecoveryMemory()
        m.add_event(40.0, 30.0, 0.4, 0.2, 1.0)
        expected = _clamp(30.0 / 40.0, RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)
        assert m.compute_recovery_factor() == pytest.approx(expected, abs=1e-12)

    def test_9_3_two_events_mean(self):
        m = RecoveryMemory()
        m.add_event(40.0, 30.0, 0.4, 0.2, 1.0)   # ratio = 0.75
        m.add_event(20.0, 25.0, 0.5, 0.3, 2.0)   # ratio = 1.25
        expected = _clamp((0.75 + 1.25) / 2.0, RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)
        assert m.compute_recovery_factor() == pytest.approx(expected, abs=1e-12)

    def test_9_4_all_equal_ratios(self):
        m = _fresh(5, ratio=0.8)
        assert m.compute_recovery_factor() == pytest.approx(0.8, abs=1e-9)

    def test_9_5_clamped_high(self):
        m = RecoveryMemory()
        m.add_event(1.0, 9_999.0, 0.5, 0.5, 1.0)
        assert m.compute_recovery_factor() == RECOVERY_FACTOR_MAX

    def test_9_6_clamped_low(self):
        m = RecoveryMemory()
        m.add_event(9_999.0, 1.0, 0.5, 0.5, 1.0)
        assert m.compute_recovery_factor() == RECOVERY_FACTOR_MIN

    def test_9_7_equal_actual_predicted_factor_one(self):
        m = _fresh(10, ratio=1.0)
        assert m.compute_recovery_factor() == pytest.approx(1.0, abs=1e-12)

    def test_9_8_returns_float(self):
        assert isinstance(_fresh(3).compute_recovery_factor(), float)

    def test_9_9_returns_finite(self):
        assert math.isfinite(_fresh(3).compute_recovery_factor())

    def test_9_10_always_in_factor_bounds(self):
        for ratio in (0.1, 0.5, 0.8, 1.0, 1.2, 1.8, 5.0):
            m = _fresh(5, ratio=ratio)
            f = m.compute_recovery_factor()
            assert RECOVERY_FACTOR_MIN <= f <= RECOVERY_FACTOR_MAX, f"ratio={ratio} → f={f}"


# =============================================================================
# SECTION 10 — compute_recovery_factor() — monotonicity
# =============================================================================

class TestFactorMonotonicity:

    def test_10_1_adding_high_ratio_event_raises_factor(self):
        m = _fresh(5, ratio=0.8)
        f_before = m.compute_recovery_factor()
        m.add_event(40.0, 60.0, 0.5, 0.3, 999.0)   # ratio = 1.5
        assert m.compute_recovery_factor() > f_before

    def test_10_2_adding_low_ratio_event_lowers_factor(self):
        m = _fresh(5, ratio=1.2)
        f_before = m.compute_recovery_factor()
        m.add_event(40.0, 10.0, 0.3, 0.1, 999.0)   # ratio = 0.25 → clamped
        assert m.compute_recovery_factor() < f_before


# =============================================================================
# SECTION 11 — events property
# =============================================================================

class TestEventsProperty:

    def test_11_1_returns_list(self):
        assert isinstance(RecoveryMemory().events, list)

    def test_11_2_is_copy(self):
        m = _fresh(3)
        copy = m.events
        copy.clear()
        assert m.event_count == 3

    def test_11_3_elements_are_recovery_events(self):
        m = _fresh(3)
        for ev in m.events:
            assert isinstance(ev, RecoveryEvent)

    def test_11_4_order_matches_insertion(self):
        m = RecoveryMemory()
        for ts in (10.0, 20.0, 30.0):
            _add(m, timestamp=ts)
        assert [e.timestamp for e in m.events] == [10.0, 20.0, 30.0]


# =============================================================================
# SECTION 12 — reset()
# =============================================================================

class TestReset:

    def test_12_1_clears_all_events(self):
        m = _fresh(10)
        m.reset()
        assert m.event_count == 0

    def test_12_2_event_count_zero(self):
        m = _fresh(5)
        m.reset()
        assert m.event_count == 0

    def test_12_3_factor_neutral_after_reset(self):
        m = _fresh(5, ratio=0.7)
        m.reset()
        assert m.compute_recovery_factor() == 1.0

    def test_12_4_events_accepted_after_reset(self):
        m = _fresh(3)
        m.reset()
        _add(m, timestamp=999.0)
        assert m.event_count == 1


# =============================================================================
# SECTION 13 — to_dict()
# =============================================================================

class TestToDict:

    def test_13_1_returns_dict(self):
        assert isinstance(RecoveryMemory().to_dict(), dict)

    def test_13_2_exact_keys(self):
        assert set(RecoveryMemory().to_dict()) == {"events_count", "recovery_factor"}

    def test_13_3_events_count_matches(self):
        m = _fresh(7)
        assert m.to_dict()["events_count"] == m.event_count

    def test_13_4_recovery_factor_matches(self):
        m = _fresh(5, ratio=0.9)
        assert m.to_dict()["recovery_factor"] == pytest.approx(
            m.compute_recovery_factor(), abs=1e-12
        )

    def test_13_5_json_serialisable(self):
        json.dumps(_fresh(3).to_dict())

    def test_13_6_events_count_is_int(self):
        assert isinstance(_fresh(3).to_dict()["events_count"], int)

    def test_13_7_recovery_factor_is_float(self):
        assert isinstance(_fresh(3).to_dict()["recovery_factor"], float)

    def test_13_8_fresh_dict_no_leakback(self):
        m = _fresh(3)
        d = m.to_dict()
        d["events_count"] = 999
        assert m.event_count != 999


# =============================================================================
# SECTION 14 — get_state() / set_state()
# =============================================================================

class TestGetSetState:

    def test_14_1_get_state_returns_dict(self):
        assert isinstance(RecoveryMemory().get_state(), dict)

    def test_14_2_get_state_json_serialisable(self):
        json.dumps(_fresh(5, ratio=0.8).get_state())

    def test_14_3_get_state_has_events_key(self):
        assert "events" in RecoveryMemory().get_state()

    def test_14_4_events_key_is_list(self):
        assert isinstance(RecoveryMemory().get_state()["events"], list)

    def test_14_5_events_len_matches_count(self):
        m = _fresh(4)
        assert len(m.get_state()["events"]) == m.event_count

    def test_14_6_set_state_restores_all_events(self):
        m1 = _fresh(5, ratio=0.85)
        snap = m1.get_state()
        m2 = RecoveryMemory()
        m2.set_state(snap)
        assert m2.event_count == m1.event_count
        for e1, e2 in zip(m1.events, m2.events):
            assert e1 == e2

    def test_14_7_set_state_restores_count(self):
        m1 = _fresh(7)
        m2 = RecoveryMemory()
        m2.set_state(m1.get_state())
        assert m2.event_count == 7

    def test_14_8_set_state_restores_factor(self):
        m1 = _fresh(8, ratio=0.75)
        m2 = RecoveryMemory()
        m2.set_state(m1.get_state())
        assert m2.compute_recovery_factor() == pytest.approx(
            m1.compute_recovery_factor(), abs=1e-12
        )

    def test_14_9_corrupt_event_skipped(self):
        snap = {
            "events": [
                {"predicted_hours": 40.0, "actual_hours": 36.0,
                 "fatigue_index": 0.4, "injury_risk": 0.2, "timestamp": 1.0},
                {"predicted_hours": "bad", "actual_hours": 30.0,  # corrupt
                 "fatigue_index": 0.5, "injury_risk": 0.3, "timestamp": 2.0},
            ]
        }
        m = RecoveryMemory()
        m.set_state(snap)
        assert m.event_count == 1

    def test_14_10_empty_events_clears_engine(self):
        m = _fresh(5)
        m.set_state({"events": []})
        assert m.event_count == 0

    def test_14_11_set_state_enforces_max_events(self):
        snap = {
            "events": [
                {"predicted_hours": 40.0, "actual_hours": 36.0,
                 "fatigue_index": 0.4, "injury_risk": 0.2, "timestamp": float(i)}
                for i in range(MAX_EVENTS + 10)
            ]
        }
        m = RecoveryMemory()
        m.set_state(snap)
        assert m.event_count == MAX_EVENTS


# =============================================================================
# SECTION 15 — apply_recovery_factor()
# =============================================================================

class TestApplyRecoveryFactor:

    def test_15_1_returns_float(self):
        assert isinstance(apply_recovery_factor(40.0, 1.0), float)

    def test_15_2_result_finite(self):
        assert math.isfinite(apply_recovery_factor(40.0, 0.8))

    def test_15_3_result_in_bounds(self):
        for pred, factor in [(10.0, 0.5), (40.0, 1.0), (100.0, 1.5)]:
            r = apply_recovery_factor(pred, factor)
            assert CORRECTED_HOURS_MIN <= r <= CORRECTED_HOURS_MAX, \
                f"pred={pred} factor={factor} → {r}"

    def test_15_4_neutral_factor_identity(self):
        assert apply_recovery_factor(40.0, 1.0) == pytest.approx(40.0, abs=1e-9)

    def test_15_5_correct_multiplication(self):
        assert apply_recovery_factor(40.0, 0.75) == pytest.approx(30.0, abs=1e-9)
        assert apply_recovery_factor(40.0, 1.25) == pytest.approx(50.0, abs=1e-9)

    def test_15_6_clamp_low(self):
        assert apply_recovery_factor(1.0, 0.5) == CORRECTED_HOURS_MIN

    def test_15_7_clamp_high(self):
        assert apply_recovery_factor(1_000.0, 1.5) == CORRECTED_HOURS_MAX

    def test_15_8_nonfinite_predicted_returns_min(self):
        assert apply_recovery_factor(float("nan"), 1.0) == CORRECTED_HOURS_MIN
        assert apply_recovery_factor(float("inf"), 1.0) == CORRECTED_HOURS_MIN

    def test_15_9_nonfinite_factor_neutral(self):
        # NaN factor → neutral (1.0) → result == predicted (if in bounds)
        result = apply_recovery_factor(40.0, float("nan"))
        assert result == pytest.approx(40.0, abs=1e-9)

    def test_15_10_json_serialisable(self):
        json.dumps({"hours": apply_recovery_factor(40.0, 0.9)})


# =============================================================================
# SECTION 16 — Determinism
# =============================================================================

class TestDeterminism:

    def test_16_1_100_identical_sequences(self):
        def _run() -> float:
            m = RecoveryMemory()
            for i, (pred, act) in enumerate(
                [(40.0, 36.0), (24.0, 28.0), (48.0, 44.0)]
            ):
                m.add_event(pred, act, 0.4, 0.2, float(i + 1))
            return m.compute_recovery_factor()

        first = _run()
        for _ in range(99):
            assert _run() == first

    def test_16_2_deterministic_check_returns_true(self):
        assert RecoveryMemory().deterministic_check() is True

    def test_16_3_call_order_independence(self):
        m = RecoveryMemory()
        m.add_event(predicted_hours=40.0, actual_hours=36.0,
                    fatigue_index=0.4, injury_risk=0.2, timestamp=1.0)
        f_before_extra = m.compute_recovery_factor()
        # Call another engine — must not affect m
        m2 = RecoveryMemory()
        m2.add_event(predicted_hours=20.0, actual_hours=30.0,
                     fatigue_index=0.5, injury_risk=0.3, timestamp=1.0)
        assert m.compute_recovery_factor() == f_before_extra


# =============================================================================
# SECTION 17 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_17_1_add_event_float_value_semantics(self):
        pred, act, fi, ir, ts = 40.0, 36.0, 0.4, 0.2, 1_000.0
        snap = (pred, act, fi, ir, ts)
        RecoveryMemory().add_event(pred, act, fi, ir, ts)
        assert (pred, act, fi, ir, ts) == snap

    def test_17_2_set_state_does_not_mutate_input(self):
        import copy
        snap = _fresh(3).get_state()
        snap_copy = copy.deepcopy(snap)
        m = RecoveryMemory()
        m.set_state(snap)
        assert snap == snap_copy

    def test_17_3_events_property_is_copy(self):
        m = _fresh(3)
        ev_copy = m.events
        ev_copy.clear()
        assert m.event_count == 3


# =============================================================================
# SECTION 18 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_18_1_to_dict_serialisable(self):
        json.dumps(_fresh(4, ratio=0.9).to_dict())

    def test_18_2_get_state_serialisable(self):
        json.dumps(_fresh(4, ratio=0.9).get_state())

    def test_18_3_full_round_trip(self):
        m1 = _fresh(5, ratio=0.85)
        raw = json.dumps(m1.get_state())
        m2  = RecoveryMemory()
        m2.set_state(json.loads(raw))
        assert m2.event_count == m1.event_count
        assert m2.compute_recovery_factor() == pytest.approx(
            m1.compute_recovery_factor(), abs=1e-12
        )

    def test_18_4_to_dict_native_types(self):
        d = _fresh(3).to_dict()
        assert isinstance(d["events_count"],    int)
        assert isinstance(d["recovery_factor"], float)


# =============================================================================
# SECTION 19 — self_test()
# =============================================================================

class TestSelfTest:

    def test_19_1_returns_dict(self):
        assert isinstance(RecoveryMemory().self_test(), dict)

    def test_19_2_required_keys(self):
        st = RecoveryMemory().self_test()
        assert {"engine", "version", "checks", "passed"} <= set(st)

    def test_19_3_five_checks(self):
        assert len(RecoveryMemory().self_test()["checks"]) == 5

    def test_19_4_each_check_has_required_fields(self):
        for c in RecoveryMemory().self_test()["checks"]:
            assert {"name", "passed", "detail"} <= set(c)

    def test_19_5_all_checks_pass(self):
        st = RecoveryMemory().self_test()
        assert st["passed"] is True
        failures = [c["name"] for c in st["checks"] if not c["passed"]]
        assert failures == [], f"Failed: {failures}"

    def test_19_6_does_not_mutate_live_state(self):
        m = _fresh(5, ratio=0.9)
        factor_before = m.compute_recovery_factor()
        count_before  = m.event_count
        m.self_test()
        assert m.event_count             == count_before
        assert m.compute_recovery_factor() == factor_before


# =============================================================================
# SECTION 20 — _clamp() helper
# =============================================================================

class TestClampHelper:

    def test_20_1_within_range_unchanged(self):
        assert _clamp(0.8, 0.5, 1.5) == pytest.approx(0.8)

    def test_20_2_below_lo_returns_lo(self):
        assert _clamp(0.1, 0.5, 1.5) == 0.5

    def test_20_3_above_hi_returns_hi(self):
        assert _clamp(2.0, 0.5, 1.5) == 1.5

    def test_20_4_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.5, 1.5) == 0.5

    def test_20_5_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.5, 1.5) == 0.5

    def test_20_6_boundary_values_exact(self):
        assert _clamp(0.5, 0.5, 1.5) == 0.5
        assert _clamp(1.5, 0.5, 1.5) == 1.5


# =============================================================================
# SECTION 21 — _finite_or_zero() helper
# =============================================================================

class TestFiniteOrZero:

    def test_21_1_finite_unchanged(self):
        assert _finite_or_zero(0.42) == pytest.approx(0.42)

    def test_21_2_nan_returns_zero(self):
        assert _finite_or_zero(float("nan")) == 0.0

    def test_21_3_pos_inf_returns_zero(self):
        assert _finite_or_zero(float("inf")) == 0.0

    def test_21_4_neg_inf_returns_zero(self):
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