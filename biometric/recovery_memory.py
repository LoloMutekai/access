"""
A.C.C.E.S.S. — Recovery Memory Engine (Phase 7.10)
biometric/recovery_memory.py

Learns how quickly the athlete *actually* recovers compared with what the
RecoveryPrediction engine forecast, and computes a personal correction factor
that downstream logic can apply to future predictions.

───────────────────────────────────────────────────────────────────────────────
Concept
───────────────────────────────────────────────────────────────────────────────

The RecoveryPrediction engine (Phase 7.9) estimates how many hours an athlete
should rest.  Real-world recovery may diverge — some athletes consistently
bounce back faster than predicted; others need longer.

RecoveryMemory closes this loop:

    1. After each training block an operator records the predicted recovery
       time and the actual hours the athlete needed.
    2. RecoveryMemory stores the event (capped at MAX_EVENTS = 200).
    3. compute_recovery_factor() returns the mean actual/predicted ratio
       across all stored events, clamped to [0.5, 1.5].
    4. apply_recovery_factor() multiplies a new prediction by this factor to
       produce a personalised correction.

───────────────────────────────────────────────────────────────────────────────
Data model
───────────────────────────────────────────────────────────────────────────────

RecoveryEvent (frozen dataclass)
    predicted_hours : float   — predicted recovery time (h)
    actual_hours    : float   — observed recovery time (h)
    fatigue_index   : float   — fatigue reading at event time ∈ [0, 1]
    injury_risk     : float   — injury risk reading at event time ∈ [0, 1]
    timestamp       : float   — POSIX timestamp or arbitrary monotone counter

RecoveryMemory
    Internal storage  : list[RecoveryEvent], bounded by MAX_EVENTS
    Eviction policy   : oldest event (index 0) is removed when limit is hit

───────────────────────────────────────────────────────────────────────────────
Recovery factor formula
───────────────────────────────────────────────────────────────────────────────

    ratio_i = actual_hours_i / predicted_hours_i      (per event)
    factor  = mean(ratio_i for all valid events)
    factor  = clamp(factor, RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)

A ratio > 1 means the athlete took longer than predicted.
A ratio < 1 means the athlete recovered faster than predicted.
factor defaults to 1.0 when no events are stored.

Corrected prediction
────────────────────
    corrected_hours = clamp(predicted_hours * factor,
                            CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX)

    CORRECTED_HOURS_MIN = 4.0   (same floor as RecoveryPrediction)
    CORRECTED_HOURS_MAX = 96.0  (extended ceiling for slow recoverers)

───────────────────────────────────────────────────────────────────────────────
Safety guarantees
───────────────────────────────────────────────────────────────────────────────

    G1  factor always finite
    G2  factor ∈ [RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX] = [0.5, 1.5]
    G3  corrected_hours always finite
    G4  corrected_hours ∈ [CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX] = [4, 96]
    G5  Input arguments to add_event() are never mutated
    G6  Deterministic — identical call sequences → identical state
    G7  Invalid / non-finite events silently ignored; no exception raised
    G8  stored events list never exceeds MAX_EVENTS

───────────────────────────────────────────────────────────────────────────────
Integration position in the BiometricEngine pipeline
───────────────────────────────────────────────────────────────────────────────

    raw signals
        → metrics          (CoreMetrics)
        → fatigue_index    (FatigueResult.value)
        → drift_score      (DriftResult)
        → injury_risk      (InjuryRiskResult.score)
        → anomaly_score    (AnomalyResult.score)
        → training_state   (str)
        → recommended_load (float)
        → baseline_deviation (float)
        → recovery_prediction (RecoveryPrediction)      Phase 7.9
        → recovery_memory correction                    ← THIS MODULE  Phase 7.10
        → final_recovery_hours (float)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

#: Maximum number of recovery events held in memory.
#: When exceeded, the oldest event is evicted (FIFO).
MAX_EVENTS: int = 200

#: Minimum allowed recovery correction factor.
RECOVERY_FACTOR_MIN: float = 0.5

#: Maximum allowed recovery correction factor.
RECOVERY_FACTOR_MAX: float = 1.5

#: Minimum corrected recovery hours (same floor as RecoveryPrediction).
CORRECTED_HOURS_MIN: float = 4.0

#: Maximum corrected recovery hours (extended ceiling for extreme slow recoverers).
CORRECTED_HOURS_MAX: float = 96.0


# =============================================================================
# RECOVERY EVENT  (immutable observation)
# =============================================================================

@dataclass(frozen=True)
class RecoveryEvent:
    """
    Immutable record of one recovery observation.

    Fields
    ──────
    predicted_hours : float
        Recovery time estimated by RecoveryPrediction, in hours.  Must be > 0.

    actual_hours : float
        Recovery time actually observed for the athlete, in hours.  Must be > 0.

    fatigue_index : float
        Fatigue reading at the time of the event.  Stored for future model
        enrichment; not used in the current factor computation.

    injury_risk : float
        Injury risk reading at the time of the event.  Stored for future
        model enrichment; not used in the current factor computation.

    timestamp : float
        POSIX timestamp or an arbitrary monotone counter that establishes
        ordering.  Events are stored in insertion order; timestamp is
        preserved for auditability but not used for ordering internally.

    Invariants
    ──────────
    - All fields are finite floats (guaranteed by RecoveryMemory.add_event).
    - predicted_hours > 0 and actual_hours > 0.
    - The dataclass is frozen — snapshots can be safely shared.
    """

    predicted_hours: float
    actual_hours:    float
    fatigue_index:   float
    injury_risk:     float
    timestamp:       float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable plain-dict representation.

        Returns
        ───────
        dict
            Keys: ``"predicted_hours"``, ``"actual_hours"``,
            ``"fatigue_index"``, ``"injury_risk"``, ``"timestamp"``.
        """
        return {
            "predicted_hours": self.predicted_hours,
            "actual_hours":    self.actual_hours,
            "fatigue_index":   self.fatigue_index,
            "injury_risk":     self.injury_risk,
            "timestamp":       self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "RecoveryEvent":
        """Reconstruct a ``RecoveryEvent`` from a ``to_dict()`` snapshot."""
        return RecoveryEvent(
            predicted_hours = float(d["predicted_hours"]),
            actual_hours    = float(d["actual_hours"]),
            fatigue_index   = float(d["fatigue_index"]),
            injury_risk     = float(d["injury_risk"]),
            timestamp       = float(d["timestamp"]),
        )

    def is_valid(self) -> bool:
        """
        Return ``True`` when all fields are finite and both hour fields > 0.
        """
        return (
            math.isfinite(self.predicted_hours) and self.predicted_hours > 0.0
            and math.isfinite(self.actual_hours)    and self.actual_hours    > 0.0
            and math.isfinite(self.fatigue_index)
            and math.isfinite(self.injury_risk)
            and math.isfinite(self.timestamp)
        )

    @property
    def ratio(self) -> float:
        """
        Actual-to-predicted ratio for this event.

        Returns ``actual_hours / predicted_hours``.  Defined only when
        ``predicted_hours > 0``; callers should guard via ``is_valid()``.
        """
        return self.actual_hours / self.predicted_hours


# =============================================================================
# RECOVERY MEMORY ENGINE
# =============================================================================

class RecoveryMemory:
    """
    Deterministic personal recovery memory engine.

    Stores a bounded list of ``RecoveryEvent`` observations and computes a
    correction factor that personalises future recovery predictions.

    Eviction policy
    ───────────────
    When the internal event list reaches ``MAX_EVENTS``, the *oldest* entry
    (index 0) is removed before the new one is appended.  This ensures the
    factor is always computed from the most recent observations.

    Thread-safety
    ─────────────
    Not thread-safe.  ``add_event()`` mutates internal state.  Use a lock
    or a separate instance per thread when concurrent access is required.

    State persistence
    ─────────────────
    ``get_state()`` / ``set_state()`` provide a JSON-serialisable snapshot
    round-trip, matching the rollback protocol of ``BaselineEngine``.

    Example
    ───────
        mem = RecoveryMemory()
        mem.add_event(predicted_hours=36.0, actual_hours=30.0,
                      fatigue_index=0.5, injury_risk=0.2,
                      timestamp=1_700_000_000.0)
        factor = mem.compute_recovery_factor()  # 30/36 ≈ 0.833
        corrected = apply_recovery_factor(predicted_hours=40.0, factor=factor)
        assert 4.0 <= corrected <= 96.0
    """

    def __init__(self) -> None:
        self._events: List[RecoveryEvent] = []

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def events(self) -> List[RecoveryEvent]:
        """
        A *copy* of the current event list.

        Returns a new list so callers cannot mutate internal storage.
        Each ``RecoveryEvent`` is itself frozen and safe to share.
        """
        return list(self._events)

    @property
    def event_count(self) -> int:
        """Number of events currently in memory."""
        return len(self._events)

    # ── Core API ──────────────────────────────────────────────────────────────

    def add_event(
        self,
        predicted_hours: float,
        actual_hours:    float,
        fatigue_index:   float,
        injury_risk:     float,
        timestamp:       float,
    ) -> None:
        """
        Record one recovery observation.

        Invalid events (non-finite fields, or either hour field ≤ 0) are
        silently discarded so a single bad sensor reading can never corrupt
        the memory state.

        When the internal list would exceed ``MAX_EVENTS`` the oldest event
        is evicted first (FIFO).

        Parameters
        ──────────
        predicted_hours : float   Recovery time predicted by the model (hours). Must be > 0.
        actual_hours    : float   Observed recovery time (hours). Must be > 0.
        fatigue_index   : float   Fatigue score at observation time (typically ∈ [0, 1]).
        injury_risk     : float   Injury risk score at observation time (typically ∈ [0, 1]).
        timestamp       : float   POSIX timestamp or monotone counter for ordering.

        Returns
        ───────
        None — mutates internal state only.
        """
        # Validate: all fields must be finite, hour fields must be positive.
        if not (
            math.isfinite(predicted_hours) and predicted_hours > 0.0
            and math.isfinite(actual_hours)    and actual_hours    > 0.0
            and math.isfinite(fatigue_index)
            and math.isfinite(injury_risk)
            and math.isfinite(timestamp)
        ):
            return  # silently discard invalid event

        event = RecoveryEvent(
            predicted_hours = predicted_hours,
            actual_hours    = actual_hours,
            fatigue_index   = fatigue_index,
            injury_risk     = injury_risk,
            timestamp       = timestamp,
        )

        if len(self._events) >= MAX_EVENTS:
            self._events.pop(0)   # evict oldest

        self._events.append(event)

    def compute_recovery_factor(self) -> float:
        """
        Compute the personal recovery correction factor.

        Algorithm
        ─────────
        For each stored event compute ``ratio_i = actual_hours / predicted_hours``.
        Return the arithmetic mean of all valid ratios, clamped to
        ``[RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX]``.

        Returns
        ───────
        float
            Correction factor ∈ ``[0.5, 1.5]``.
            Returns ``1.0`` (neutral) when no events are stored.

        Interpretation
        ──────────────
        < 1.0  → athlete recovers faster than predicted (reduce future estimate)
        = 1.0  → prediction matches reality on average
        > 1.0  → athlete recovers slower than predicted (increase future estimate)
        """
        if not self._events:
            return 1.0

        ratios = [
            e.ratio for e in self._events
            if math.isfinite(e.ratio)
        ]

        if not ratios:
            return 1.0

        mean_ratio = sum(ratios) / len(ratios)
        return _clamp(mean_ratio, RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)

    def reset(self) -> None:
        """
        Clear all stored events, returning the engine to its initial state.

        Useful for unit tests and session rollback.
        """
        self._events.clear()

    # ── Serialisation / rollback ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a compact JSON-serialisable summary of the current state.

        This is the *public* output format — a lightweight snapshot
        appropriate for embedding in pipeline results.

        Returns
        ───────
        dict
            ``{"events_count": int, "recovery_factor": float}``
        """
        return {
            "events_count":    self.event_count,
            "recovery_factor": self.compute_recovery_factor(),
        }

    def get_state(self) -> dict:
        """
        Return a *full* JSON-serialisable snapshot of all internal state.

        The snapshot includes every stored event so it can be passed to
        ``set_state()`` to restore the engine exactly.

        Returns
        ───────
        dict
            ``{"events": list[dict]}`` where each inner dict is the output
            of ``RecoveryEvent.to_dict()``.
        """
        return {
            "events": [e.to_dict() for e in self._events],
        }

    def set_state(self, state: dict) -> None:
        """
        Restore internal state from a ``get_state()`` snapshot.

        Events that fail validation during restoration are silently skipped,
        so a corrupt snapshot cannot produce an invalid engine state.

        Parameters
        ──────────
        state : dict
            Must contain ``"events"``: a list of ``RecoveryEvent.to_dict()``
            dicts.
        """
        self._events.clear()
        for raw in state.get("events", []):
            try:
                ev = RecoveryEvent.from_dict(raw)
                if ev.is_valid():
                    self._events.append(ev)
            except (KeyError, TypeError, ValueError):
                pass  # corrupt snapshot entry — skip silently

        # Enforce capacity limit after restore
        if len(self._events) > MAX_EVENTS:
            self._events = self._events[-MAX_EVENTS:]

    # ── Self-diagnostics ──────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run five named invariant checks on *fresh* probe instances so the
        live event list is never touched.

        Checks
        ──────
        1. ``no_events_returns_neutral_factor``
               Empty memory → factor == 1.0.
        2. ``single_event_ratio_exact``
               One event with known ratio → factor equals that ratio.
        3. ``mean_ratio_correct``
               Three events → factor == arithmetic mean of ratios.
        4. ``factor_clamped_high``
               Ratio > RECOVERY_FACTOR_MAX → factor clamped to max.
        5. ``json_serialisable``
               to_dict() and get_state() round-trip through json.

        Returns
        ───────
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        checks: list[dict] = []

        # 1 — empty → neutral
        def _no_events() -> bool:
            return RecoveryMemory().compute_recovery_factor() == 1.0
        checks.append(_run_check("no_events_returns_neutral_factor", _no_events))

        # 2 — single event ratio
        def _single_event() -> bool:
            m = RecoveryMemory()
            m.add_event(predicted_hours=40.0, actual_hours=30.0,
                        fatigue_index=0.5, injury_risk=0.2, timestamp=1.0)
            expected = _clamp(30.0 / 40.0, RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)
            return abs(m.compute_recovery_factor() - expected) < 1e-12
        checks.append(_run_check("single_event_ratio_exact", _single_event))

        # 3 — mean of three ratios
        def _mean_ratio() -> bool:
            m = RecoveryMemory()
            data = [(40.0, 30.0), (20.0, 25.0), (48.0, 48.0)]
            for pred, act in data:
                m.add_event(pred, act, 0.3, 0.2, float(len(m._events)))
            ratios = [act / pred for pred, act in data]
            expected = _clamp(sum(ratios) / len(ratios),
                              RECOVERY_FACTOR_MIN, RECOVERY_FACTOR_MAX)
            return abs(m.compute_recovery_factor() - expected) < 1e-12
        checks.append(_run_check("mean_ratio_correct", _mean_ratio))

        # 4 — clamp high
        def _clamp_high() -> bool:
            m = RecoveryMemory()
            m.add_event(10.0, 999.0, 0.5, 0.5, 1.0)  # ratio = 99.9 → clamped
            return m.compute_recovery_factor() == RECOVERY_FACTOR_MAX
        checks.append(_run_check("factor_clamped_high", _clamp_high))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            m = RecoveryMemory()
            m.add_event(36.0, 30.0, 0.4, 0.2, 1_700_000_000.0)
            m.add_event(24.0, 28.0, 0.6, 0.3, 1_700_001_000.0)
            raw_d = json.dumps(m.to_dict())
            raw_s = json.dumps(m.get_state())
            d = json.loads(raw_d)
            s = json.loads(raw_s)
            m2 = RecoveryMemory()
            m2.set_state(s)
            return (
                isinstance(d["events_count"], int)
                and isinstance(d["recovery_factor"], float)
                and m2.event_count == m.event_count
                and abs(m2.compute_recovery_factor() - m.compute_recovery_factor()) < 1e-12
            )
        checks.append(_run_check("json_serialisable", _json_rt))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  "RecoveryMemory",
            "version": "7.10.0",
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run a canonical 5-event sequence twice on independent fresh instances
        and verify the recovery factor is bit-for-bit identical.

        Returns
        ───────
        bool
            ``True`` if both runs produce identical factors for every step.
        """
        canonical = [
            (36.0, 30.0, 0.4, 0.2, 1_000.0),
            (24.0, 28.0, 0.6, 0.3, 2_000.0),
            (48.0, 48.0, 0.5, 0.4, 3_000.0),
            (20.0, 15.0, 0.3, 0.1, 4_000.0),
            (40.0, 44.0, 0.7, 0.5, 5_000.0),
        ]

        def _run() -> list[float]:
            m = RecoveryMemory()
            factors: list[float] = []
            for args in canonical:
                m.add_event(*args)
                factors.append(m.compute_recovery_factor())
            return factors

        return _run() == _run()


# =============================================================================
# MODULE-LEVEL PUBLIC HELPER
# =============================================================================

def apply_recovery_factor(predicted_hours: float, factor: float) -> float:
    """
    Apply a recovery correction factor to a raw predicted recovery time.

    Multiplies ``predicted_hours`` by ``factor`` and clamps the result to
    ``[CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX]``.

    Parameters
    ──────────
    predicted_hours : float
        Raw predicted recovery time from ``RecoveryPrediction`` (hours).
        Non-finite values produce ``CORRECTED_HOURS_MIN`` (safe fallback).

    factor : float
        Correction factor from ``RecoveryMemory.compute_recovery_factor()``.
        Non-finite values are replaced with ``1.0`` (neutral).

    Returns
    ───────
    float
        Corrected recovery hours ∈ ``[CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX]``
        = ``[4.0, 96.0]``.

    Examples
    ────────
        apply_recovery_factor(40.0, 0.75)  →  30.0   (fast recoverer)
        apply_recovery_factor(40.0, 1.25)  →  50.0   (slow recoverer)
        apply_recovery_factor(40.0, 1.00)  →  40.0   (neutral)
    """
    safe_hours  = _finite_or_default(predicted_hours, CORRECTED_HOURS_MIN)
    safe_factor = _finite_or_default(factor,          1.0)
    corrected   = safe_hours * safe_factor
    return _clamp(corrected, CORRECTED_HOURS_MIN, CORRECTED_HOURS_MAX)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to the closed interval ``[lo, hi]``.

    Returns ``lo`` for any non-finite input (NaN, ±Inf) so a corrupt
    intermediate can never propagate beyond the module boundary.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _finite_or_zero(value: float) -> float:
    """Return ``value`` if finite, otherwise ``0.0``."""
    return value if math.isfinite(value) else 0.0


def _finite_or_default(value: float, default: float) -> float:
    """Return ``value`` if finite, otherwise ``default``."""
    return value if math.isfinite(value) else default


def _run_check(name: str, fn) -> dict:
    """Execute a named check function and catch all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}