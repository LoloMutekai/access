"""
A.C.C.E.S.S. — Athlete Digital Twin (Phase B7 / 7.14)
biometric/athlete_digital_twin.py

Implements a deterministic long-term digital twin of an athlete's
physiological system.  The twin integrates outputs from all upstream
biometric engines, maintains a bounded event history, updates core state
variables after each new biometric event, and delegates short-term
forecasting to the PerformanceForecastEngine.

───────────────────────────────────────────────────────────────────────────────
Pipeline position
───────────────────────────────────────────────────────────────────────────────

    raw signals
        → biometric metrics
        → recovery prediction
        → recovery memory correction
        → rule evolution
        → performance forecast
        → athlete digital twin          ← THIS MODULE  (Phase 7.14)

───────────────────────────────────────────────────────────────────────────────
Core state variables
───────────────────────────────────────────────────────────────────────────────

All state values are bounded to [0, 1] at all times:

    athlete_id          str     — unique athlete identifier
    baseline_fatigue    float   — initial physiological fatigue prior
    baseline_load       float   — initial training load prior
    adaptation_factor   float   — long-term training tolerance (EMA of load)
    fatigue_state       float   — current EMA-smoothed fatigue level
    injury_risk_state   float   — current injury risk derived from fatigue/recovery
    readiness_state     float   — current training readiness

───────────────────────────────────────────────────────────────────────────────
State update rules  (applied in this exact order per event)
───────────────────────────────────────────────────────────────────────────────

    norm_load     = sprint_load / LOAD_REFERENCE
    norm_recovery = clamp(recovery_hours / RECOVERY_REFERENCE, 0, 1)

    Step 1 — Fatigue EMA
        fatigue_state = clamp(
            EMA_FATIGUE_RETAIN * fatigue_state
            + EMA_FATIGUE_EVENT * event.fatigue_index
        )

    Step 2 — Adaptation factor EMA
        adaptation_factor = clamp(
            EMA_ADAPTATION_RETAIN * adaptation_factor
            + EMA_ADAPTATION_LOAD * norm_load
        )

    Step 3 — Injury risk  (uses post-step-1 fatigue)
        injury_risk_state = clamp(
            W_FATIGUE_RISK * fatigue_state
            + W_RECOVERY_DEFICIT_RISK * (1 − norm_recovery)
        )

    Step 4 — Readiness  (uses post-step-1 fatigue and post-step-3 injury_risk)
        readiness_state = clamp(
            1 − fatigue_state − injury_risk_state * READINESS_INJURY_COEFF
        )

    Update constants:
        EMA_FATIGUE_RETAIN       = 0.7    prior fatigue weight
        EMA_FATIGUE_EVENT        = 0.3    new event fatigue weight
        EMA_ADAPTATION_RETAIN    = 0.9    prior adaptation weight
        EMA_ADAPTATION_LOAD      = 0.1    new load weight
        W_FATIGUE_RISK           = 0.6    fatigue→injury risk weight
        W_RECOVERY_DEFICIT_RISK  = 0.4    recovery deficit→injury risk weight
        READINESS_INJURY_COEFF   = 0.5    injury risk coefficient in readiness
        LOAD_REFERENCE           = 8000.0 load normalisation denominator
        RECOVERY_REFERENCE       = 24.0   recovery normalisation denominator

───────────────────────────────────────────────────────────────────────────────
History
───────────────────────────────────────────────────────────────────────────────

    MAX_HISTORY = 5000
    Oldest event evicted (FIFO) when capacity is exceeded.
    Each stored event is a frozen TwinEvent dataclass.

───────────────────────────────────────────────────────────────────────────────
Simulation
───────────────────────────────────────────────────────────────────────────────

    simulate_training(load, recovery_hours) → dict

    Clones the current twin via to_dict() → from_dict() (zero aliasing risk),
    applies a synthetic event (fatigue_index = current fatigue_state),
    runs the forecast engine on the clone, and returns the result.
    The original twin is guaranteed to remain unmodified.

───────────────────────────────────────────────────────────────────────────────
Forecast integration
───────────────────────────────────────────────────────────────────────────────

    forecast() calls PerformanceForecastEngine.forecast() on the stored
    event history and returns a 5-key dict remapping
        "injury_risk_forecast" → "injury_risk"
    and dropping "recovery_forecast".

───────────────────────────────────────────────────────────────────────────────
Mathematical guarantees
───────────────────────────────────────────────────────────────────────────────

    G1  fatigue_state, injury_risk_state, readiness_state, adaptation_factor
        always ∈ [0.0, 1.0]
    G2  All output floats finite
    G3  Deterministic — identical event sequences → identical state
    G4  Input events never mutated
    G5  JSON-serialisable via to_dict() / from_dict()
    G6  Bounded execution — O(N) per update; history ≤ MAX_HISTORY
    G7  simulate_training() leaves original twin state unmodified
    G8  Invalid events silently rejected; state unchanged
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from biometric.performance_forecast_engine import (
    PerformanceForecastEngine,
    _clamp         as _forecast_clamp,
    _finite_or_zero as _forecast_foz,
)


# =============================================================================
# VERSION
# =============================================================================

ENGINE_NAME:    str = "AthleteDigitalTwin"
ENGINE_VERSION: str = "7.14.0"


# =============================================================================
# CONSTANTS
# =============================================================================

#: Maximum events retained in history.
MAX_HISTORY: int = 5_000

#: Load normalisation denominator (arbitrary units).
LOAD_REFERENCE: float = 8_000.0

#: Recovery normalisation denominator (hours).
RECOVERY_REFERENCE: float = 24.0

# ── Fatigue EMA ────────────────────────────────────────────────────────────────
#: Weight of the prior fatigue state in the EMA update.
EMA_FATIGUE_RETAIN: float = 0.7

#: Weight of the new event's fatigue_index in the EMA update.
EMA_FATIGUE_EVENT: float = 0.3

_EMA_FATIGUE_SUM: float = EMA_FATIGUE_RETAIN + EMA_FATIGUE_EVENT
assert abs(_EMA_FATIGUE_SUM - 1.0) < 1e-12, (
    f"Fatigue EMA weights must sum to 1.0, got {_EMA_FATIGUE_SUM}"
)

# ── Adaptation EMA ─────────────────────────────────────────────────────────────
#: Weight of the prior adaptation factor in the EMA update.
EMA_ADAPTATION_RETAIN: float = 0.9

#: Weight of the new load contribution in the EMA update.
EMA_ADAPTATION_LOAD: float = 0.1

_EMA_ADAPTATION_SUM: float = EMA_ADAPTATION_RETAIN + EMA_ADAPTATION_LOAD
assert abs(_EMA_ADAPTATION_SUM - 1.0) < 1e-12, (
    f"Adaptation EMA weights must sum to 1.0, got {_EMA_ADAPTATION_SUM}"
)

# ── Injury risk weights ────────────────────────────────────────────────────────
#: Fatigue weight in the injury risk formula.
W_FATIGUE_RISK: float = 0.6

#: Recovery deficit weight in the injury risk formula.
W_RECOVERY_DEFICIT_RISK: float = 0.4

_IR_WEIGHT_SUM: float = W_FATIGUE_RISK + W_RECOVERY_DEFICIT_RISK
assert abs(_IR_WEIGHT_SUM - 1.0) < 1e-12, (
    f"Injury risk weights must sum to 1.0, got {_IR_WEIGHT_SUM}"
)

#: Injury risk coefficient in the readiness formula.
READINESS_INJURY_COEFF: float = 0.5

# ── Required event fields ──────────────────────────────────────────────────────
_REQUIRED_FIELDS: tuple[str, ...] = (
    "fatigue_index", "sprint_load", "recovery_hours", "injury_flag",
)

#: Output keys for forecast() and simulate_training().
_FORECAST_OUTPUT_KEYS: frozenset[str] = frozenset({
    "fatigue_24h", "fatigue_48h", "fatigue_72h",
    "injury_risk", "readiness_score",
})


# =============================================================================
# TWIN EVENT  (immutable observation)
# =============================================================================

@dataclass(frozen=True)
class TwinEvent:
    """
    Immutable record of one biometric observation stored in the twin's history.

    Fields
    ──────
    fatigue_index   : float  — fatigue score ∈ [0, 1]
    sprint_load     : float  — session load ≥ 0  (arbitrary units)
    recovery_hours  : float  — rest hours before session ≥ 0
    injury_flag     : int    — ground-truth injury outcome ∈ {0, 1}

    The dataclass is frozen and fully hashable.

    Invariants
    ──────────
    - All float fields are finite.
    - fatigue_index ∈ [0, 1].
    - sprint_load ≥ 0.
    - recovery_hours ≥ 0.
    - injury_flag ∈ {0, 1}.
    """

    fatigue_index:  float
    sprint_load:    float
    recovery_hours: float
    injury_flag:    int

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "fatigue_index":  self.fatigue_index,
            "sprint_load":    self.sprint_load,
            "recovery_hours": self.recovery_hours,
            "injury_flag":    self.injury_flag,
        }

    @staticmethod
    def from_dict(d: dict) -> "TwinEvent":
        """Reconstruct from a ``to_dict()`` snapshot."""
        return TwinEvent(
            fatigue_index  = float(d["fatigue_index"]),
            sprint_load    = float(d["sprint_load"]),
            recovery_hours = float(d["recovery_hours"]),
            injury_flag    = int(d["injury_flag"]),
        )

    def is_valid(self) -> bool:
        """Return True when all fields satisfy the documented invariants."""
        return (
            math.isfinite(self.fatigue_index)
            and 0.0 <= self.fatigue_index <= 1.0
            and math.isfinite(self.sprint_load)
            and self.sprint_load >= 0.0
            and math.isfinite(self.recovery_hours)
            and self.recovery_hours >= 0.0
            and self.injury_flag in (0, 1)
        )

    @property
    def norm_recovery(self) -> float:
        """Normalised recovery ∈ [0, 1] = clamp(recovery_hours / 24, 0, 1)."""
        return _clamp(self.recovery_hours / RECOVERY_REFERENCE, 0.0, 1.0)

    @property
    def norm_load(self) -> float:
        """Normalised load = sprint_load / LOAD_REFERENCE (unbounded above 1)."""
        return _finite_or_zero(self.sprint_load / LOAD_REFERENCE)


# =============================================================================
# TWIN STATE  (immutable snapshot — used for get_state / to_dict / from_dict)
# =============================================================================

@dataclass(frozen=True)
class TwinState:
    """
    Immutable snapshot of the twin's core state variables (without history).

    Created by ``AthleteDigitalTwin.get_state()`` and embedded in
    ``to_dict()`` serialisation.  Never directly exposed to callers — they
    interact via the AthleteDigitalTwin API.

    All float fields are bounded to [0.0, 1.0].
    """

    athlete_id:        str
    baseline_fatigue:  float
    baseline_load:     float
    adaptation_factor: float
    fatigue_state:     float
    injury_risk_state: float
    readiness_state:   float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of state variables."""
        return {
            "athlete_id":        self.athlete_id,
            "baseline_fatigue":  self.baseline_fatigue,
            "baseline_load":     self.baseline_load,
            "adaptation_factor": self.adaptation_factor,
            "fatigue_state":     self.fatigue_state,
            "injury_risk_state": self.injury_risk_state,
            "readiness_state":   self.readiness_state,
        }

    def is_valid(self) -> bool:
        """Return True when all float fields are finite and ∈ [0, 1]."""
        floats = (
            self.baseline_fatigue, self.baseline_load,
            self.adaptation_factor, self.fatigue_state,
            self.injury_risk_state, self.readiness_state,
        )
        return (
            isinstance(self.athlete_id, str)
            and all(math.isfinite(f) and 0.0 <= f <= 1.0 for f in floats)
        )


# =============================================================================
# ATHLETE DIGITAL TWIN
# =============================================================================

class AthleteDigitalTwin:
    """
    Deterministic, stateful digital twin of an athlete's physiological system.

    The twin evolves its internal state via exponential moving average (EMA)
    smoothing on each new biometric event, maintains a bounded FIFO event
    history, and integrates the PerformanceForecastEngine for 24/48/72 h
    horizon predictions.

    Statefulness
    ────────────
    Unlike the stateless PerformanceForecastEngine, this class retains mutable
    state across calls.  Use ``get_state()`` / ``set_state()`` or
    ``to_dict()`` / ``from_dict()`` for serialisation and restore.

    Thread-safety
    ─────────────
    Not thread-safe.  Protect with a lock if ``update()`` may be called
    concurrently.

    Example
    ───────
        twin = AthleteDigitalTwin(athlete_id="alice_007")
        twin.update({"fatigue_index": 0.6, "sprint_load": 5000.0,
                     "recovery_hours": 18.0, "injury_flag": 0})
        result = twin.forecast()
        assert 0.0 <= result["readiness_score"] <= 1.0
    """

    def __init__(
        self,
        athlete_id:       str   = "athlete_001",
        baseline_fatigue: float = 0.0,
        baseline_load:    float = 0.0,
    ) -> None:
        """
        Initialise a fresh digital twin.

        Parameters
        ──────────
        athlete_id       : str    — unique identifier (default "athlete_001")
        baseline_fatigue : float  — starting fatigue prior ∈ [0, 1]
        baseline_load    : float  — starting load prior ∈ [0, 1]
        """
        self._athlete_id: str = str(athlete_id)

        # Sanitise constructor arguments
        bf = _clamp(_finite_or_zero(baseline_fatigue), 0.0, 1.0)
        bl = _clamp(_finite_or_zero(baseline_load),    0.0, 1.0)

        self._baseline_fatigue: float = bf
        self._baseline_load:    float = bl

        # Core state — initialised from priors
        self._adaptation_factor: float = bl         # load prior seeds adaptation
        self._fatigue_state:     float = bf         # fatigue prior seeds state
        self._injury_risk_state: float = _compute_injury_risk(bf, 1.0)
        self._readiness_state:   float = _compute_readiness(bf, self._injury_risk_state)

        # Bounded event history
        self._events: list[TwinEvent] = []

        # Shared forecast engine (stateless — safe to reuse)
        self._forecast_engine: PerformanceForecastEngine = PerformanceForecastEngine()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def athlete_id(self) -> str:
        """Unique athlete identifier."""
        return self._athlete_id

    @property
    def fatigue_state(self) -> float:
        """Current EMA-smoothed fatigue level ∈ [0, 1]."""
        return self._fatigue_state

    @property
    def injury_risk_state(self) -> float:
        """Current injury risk ∈ [0, 1]."""
        return self._injury_risk_state

    @property
    def readiness_state(self) -> float:
        """Current training readiness ∈ [0, 1]."""
        return self._readiness_state

    @property
    def adaptation_factor(self) -> float:
        """Long-term training tolerance ∈ [0, 1]."""
        return self._adaptation_factor

    @property
    def event_count(self) -> int:
        """Number of events currently in history."""
        return len(self._events)

    @property
    def events(self) -> list[TwinEvent]:
        """A shallow copy of the event history list (caller-safe)."""
        return list(self._events)

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, event: dict) -> None:
        """
        Ingest one new biometric event and update all state variables.

        Invalid events (missing fields, non-finite values, wrong types) are
        silently rejected — state is not changed.

        Parameters
        ──────────
        event : dict
            Must contain: ``fatigue_index``, ``sprint_load``,
            ``recovery_hours``, ``injury_flag``.
            Additional keys are silently ignored.
            The dict is never mutated.

        Update order
        ────────────
        1. fatigue_state     (EMA of prior + new fatigue_index)
        2. adaptation_factor (EMA of prior + new norm_load)
        3. injury_risk_state (function of updated fatigue + norm_recovery)
        4. readiness_state   (function of updated fatigue + updated injury_risk)
        5. Append to history (evict oldest if MAX_HISTORY exceeded)
        """
        if not _is_valid_event(event):
            return

        ev = TwinEvent(
            fatigue_index  = _clamp(float(event["fatigue_index"]),  0.0, 1.0),
            sprint_load    = max(0.0, _finite_or_zero(float(event["sprint_load"]))),
            recovery_hours = max(0.0, _finite_or_zero(float(event["recovery_hours"]))),
            injury_flag    = int(event["injury_flag"]),
        )

        # Step 1 — Fatigue EMA
        self._fatigue_state = _clamp(
            EMA_FATIGUE_RETAIN * self._fatigue_state
            + EMA_FATIGUE_EVENT * ev.fatigue_index,
            0.0, 1.0,
        )

        # Step 2 — Adaptation EMA
        self._adaptation_factor = _clamp(
            EMA_ADAPTATION_RETAIN * self._adaptation_factor
            + EMA_ADAPTATION_LOAD * ev.norm_load,
            0.0, 1.0,
        )

        # Step 3 — Injury risk (uses updated fatigue)
        self._injury_risk_state = _compute_injury_risk(
            self._fatigue_state, ev.norm_recovery
        )

        # Step 4 — Readiness (uses updated fatigue and injury risk)
        self._readiness_state = _compute_readiness(
            self._fatigue_state, self._injury_risk_state
        )

        # Step 5 — History
        if len(self._events) >= MAX_HISTORY:
            self._events.pop(0)
        self._events.append(ev)

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast(self) -> dict[str, float]:
        """
        Compute a 3-horizon performance forecast using the stored event history.

        Delegates to ``PerformanceForecastEngine.forecast()`` on the stored
        event history.  If no events are stored the forecast engine applies
        its own safe defaults (fully rested, unloaded athlete).

        Returns
        ───────
        dict[str, float]
            Keys: ``"fatigue_24h"``, ``"fatigue_48h"``, ``"fatigue_72h"``,
            ``"injury_risk"``, ``"readiness_score"``.
            All values ∈ [0.0, 1.0] and finite.
        """
        raw = self._forecast_engine.forecast(self._events_as_forecast_input())
        return {
            "fatigue_24h":   raw["fatigue_24h"],
            "fatigue_48h":   raw["fatigue_48h"],
            "fatigue_72h":   raw["fatigue_72h"],
            "injury_risk":   raw["injury_risk_forecast"],
            "readiness_score": raw["readiness_score"],
        }

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_training(
        self,
        load:           float,
        recovery_hours: float,
    ) -> dict[str, float]:
        """
        Simulate a hypothetical training session and return the forecast.

        The simulation:
        1. Clones the current twin (deep copy via ``to_dict()`` / ``from_dict()``).
        2. Constructs a synthetic event:
               fatigue_index  = current fatigue_state
               sprint_load    = load
               recovery_hours = recovery_hours
               injury_flag    = 0  (unknown future outcome)
        3. Applies the event to the **clone** only.
        4. Runs ``clone.forecast()`` and returns the result.

        The original twin is **guaranteed to be unmodified**.

        Parameters
        ──────────
        load           : float  — hypothetical sprint load (≥ 0).
        recovery_hours : float  — hypothetical recovery hours (≥ 0).
                                   Non-finite values default to 0.0.

        Returns
        ───────
        dict[str, float]
            Same 5-key structure as ``forecast()``.
        """
        safe_load     = max(0.0, _finite_or_zero(float(load)))            if _is_finite_num(load)           else 0.0
        safe_recovery = max(0.0, _finite_or_zero(float(recovery_hours)))  if _is_finite_num(recovery_hours)  else 0.0

        # Deep clone via serialisation round-trip (zero aliasing risk)
        clone = AthleteDigitalTwin.from_dict(self.to_dict())

        # Synthetic event: current fatigue_state as fatigue_index
        clone.update({
            "fatigue_index":  self._fatigue_state,
            "sprint_load":    safe_load,
            "recovery_hours": safe_recovery,
            "injury_flag":    0,
        })

        return clone.forecast()

    # ── State management ──────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """
        Return a lightweight JSON-serialisable snapshot of the core state.

        Does **not** include the full event history — use ``to_dict()`` for
        a complete serialisable snapshot.

        Returns
        ───────
        dict
            Keys: ``"athlete_id"``, ``"baseline_fatigue"``,
            ``"baseline_load"``, ``"adaptation_factor"``,
            ``"fatigue_state"``, ``"injury_risk_state"``,
            ``"readiness_state"``, ``"event_count"``.
        """
        return {
            "athlete_id":        self._athlete_id,
            "baseline_fatigue":  self._baseline_fatigue,
            "baseline_load":     self._baseline_load,
            "adaptation_factor": self._adaptation_factor,
            "fatigue_state":     self._fatigue_state,
            "injury_risk_state": self._injury_risk_state,
            "readiness_state":   self._readiness_state,
            "event_count":       self.event_count,
        }

    def set_state(self, state: dict) -> None:
        """
        Restore core state variables from a ``get_state()`` snapshot.

        The event history is **not** restored by this method.  For a full
        restore (state + history) use ``from_dict(to_dict())``.

        Corrupt or non-finite values are silently clamped or ignored.
        """
        self._athlete_id        = str(state.get("athlete_id", self._athlete_id))
        self._baseline_fatigue  = _clamp(_finite_or_zero(float(state.get("baseline_fatigue",  0.0))), 0.0, 1.0)
        self._baseline_load     = _clamp(_finite_or_zero(float(state.get("baseline_load",     0.0))), 0.0, 1.0)
        self._adaptation_factor = _clamp(_finite_or_zero(float(state.get("adaptation_factor", 0.0))), 0.0, 1.0)
        self._fatigue_state     = _clamp(_finite_or_zero(float(state.get("fatigue_state",     0.0))), 0.0, 1.0)
        self._injury_risk_state = _clamp(_finite_or_zero(float(state.get("injury_risk_state", 0.0))), 0.0, 1.0)
        self._readiness_state   = _clamp(_finite_or_zero(float(state.get("readiness_state",   1.0))), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a complete JSON-serialisable snapshot including event history.

        Can be restored via ``AthleteDigitalTwin.from_dict()``.

        Returns
        ───────
        dict
            ``{"state": {...}, "events": [TwinEvent.to_dict(), ...]}``
        """
        return {
            "state":  TwinState(
                athlete_id        = self._athlete_id,
                baseline_fatigue  = self._baseline_fatigue,
                baseline_load     = self._baseline_load,
                adaptation_factor = self._adaptation_factor,
                fatigue_state     = self._fatigue_state,
                injury_risk_state = self._injury_risk_state,
                readiness_state   = self._readiness_state,
            ).to_dict(),
            "events": [ev.to_dict() for ev in self._events],
        }

    @staticmethod
    def from_dict(d: dict) -> "AthleteDigitalTwin":
        """
        Reconstruct an ``AthleteDigitalTwin`` from a ``to_dict()`` snapshot.

        Invalid or corrupt event entries are silently skipped.

        Parameters
        ──────────
        d : dict
            A dict produced by ``to_dict()``.  Must contain ``"state"`` and
            ``"events"`` keys.

        Returns
        ───────
        AthleteDigitalTwin
            Fully restored twin with identical state and history.
        """
        state_d = d.get("state", {})
        twin = AthleteDigitalTwin(
            athlete_id       = str(state_d.get("athlete_id", "athlete_001")),
            baseline_fatigue = float(state_d.get("baseline_fatigue", 0.0)),
            baseline_load    = float(state_d.get("baseline_load",    0.0)),
        )
        # Restore remaining state variables directly
        twin._adaptation_factor = _clamp(
            _finite_or_zero(float(state_d.get("adaptation_factor", 0.0))), 0.0, 1.0
        )
        twin._fatigue_state     = _clamp(
            _finite_or_zero(float(state_d.get("fatigue_state",     0.0))), 0.0, 1.0
        )
        twin._injury_risk_state = _clamp(
            _finite_or_zero(float(state_d.get("injury_risk_state", 0.0))), 0.0, 1.0
        )
        twin._readiness_state   = _clamp(
            _finite_or_zero(float(state_d.get("readiness_state",   1.0))), 0.0, 1.0
        )

        # Restore event history
        twin._events.clear()
        for raw_ev in d.get("events", []):
            try:
                ev = TwinEvent.from_dict(raw_ev)
                if ev.is_valid():
                    twin._events.append(ev)
            except (KeyError, TypeError, ValueError):
                pass  # corrupt entry — skip

        # Enforce cap after restore
        if len(twin._events) > MAX_HISTORY:
            twin._events = twin._events[-MAX_HISTORY:]

        return twin

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run six named invariant checks on fresh probe instances.

        The live twin's state is never touched.

        Checks
        ──────
        1. ``initial_state_valid``       — fresh twin satisfies all invariants
        2. ``update_changes_state``      — one valid event changes fatigue_state
        3. ``invalid_event_rejected``    — NaN input leaves state unchanged
        4. ``simulate_training_isolated`` — simulation does not mutate original
        5. ``json_round_trip``           — to_dict → from_dict → same state
        6. ``deterministic_output``      — two twins, same events → same state

        Returns
        ───────
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        checks: list[dict] = []

        # 1 — fresh twin valid
        def _initial_valid() -> bool:
            t = AthleteDigitalTwin()
            s = t.get_state()
            return all(0.0 <= s[k] <= 1.0 for k in (
                "baseline_fatigue", "baseline_load", "adaptation_factor",
                "fatigue_state", "injury_risk_state", "readiness_state",
            ))
        checks.append(_run_check("initial_state_valid", _initial_valid))

        # 2 — update changes state
        def _update_changes() -> bool:
            t = AthleteDigitalTwin()
            f_before = t.fatigue_state
            t.update({"fatigue_index": 0.8, "sprint_load": 6000.0,
                      "recovery_hours": 12.0, "injury_flag": 0})
            return t.fatigue_state != f_before
        checks.append(_run_check("update_changes_state", _update_changes))

        # 3 — invalid event rejected
        def _invalid_rejected() -> bool:
            t = AthleteDigitalTwin()
            t.update({"fatigue_index": 0.5, "sprint_load": 4000.0,
                      "recovery_hours": 20.0, "injury_flag": 0})
            f_before = t.fatigue_state
            t.update({"fatigue_index": float("nan"), "sprint_load": 4000.0,
                      "recovery_hours": 20.0, "injury_flag": 0})
            return t.fatigue_state == f_before and t.event_count == 1
        checks.append(_run_check("invalid_event_rejected", _invalid_rejected))

        # 4 — simulate_training isolation
        def _simulate_isolated() -> bool:
            t = AthleteDigitalTwin()
            t.update({"fatigue_index": 0.5, "sprint_load": 4000.0,
                      "recovery_hours": 20.0, "injury_flag": 0})
            f_before = t.fatigue_state
            ec_before = t.event_count
            t.simulate_training(6000.0, 12.0)
            return t.fatigue_state == f_before and t.event_count == ec_before
        checks.append(_run_check("simulate_training_isolated", _simulate_isolated))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            t = AthleteDigitalTwin(athlete_id="probe")
            for i in range(10):
                t.update({"fatigue_index": 0.3 + 0.05 * i, "sprint_load": 4000.0 + 200.0 * i,
                           "recovery_hours": 20.0 - i, "injury_flag": 0})
            raw  = json.dumps(t.to_dict())
            t2   = AthleteDigitalTwin.from_dict(json.loads(raw))
            return (
                abs(t2.fatigue_state     - t.fatigue_state)     < 1e-9
                and abs(t2.adaptation_factor - t.adaptation_factor) < 1e-9
                and t2.event_count == t.event_count
            )
        checks.append(_run_check("json_round_trip", _json_rt))

        # 6 — determinism
        def _deterministic() -> bool:
            evs = _make_probe_events(20)
            def _run() -> dict:
                t = AthleteDigitalTwin(athlete_id="det_probe")
                for ev in evs:
                    t.update(ev)
                return t.get_state()
            return _run() == _run()
        checks.append(_run_check("deterministic_output", _deterministic))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  ENGINE_NAME,
            "version": ENGINE_VERSION,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run the canonical probe sequence twice on independent fresh twins and
        verify bit-for-bit state equality.

        Returns
        ───────
        bool
            ``True`` when both runs produce identical states and forecasts.
        """
        evs = _make_probe_events(40)

        def _run() -> tuple:
            t = AthleteDigitalTwin(athlete_id="canon_probe")
            for ev in evs:
                t.update(ev)
            s = t.get_state()
            f = t.forecast()
            return (
                s["fatigue_state"],
                s["injury_risk_state"],
                s["readiness_state"],
                s["adaptation_factor"],
                t.event_count,
                f["fatigue_24h"],
                f["injury_risk"],
            )

        return _run() == _run()

    # ── Convenience ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all history and restore state to construction defaults.

        Equivalent to creating a new twin with the same ``athlete_id``,
        ``baseline_fatigue``, and ``baseline_load``.
        """
        self.__init__(
            athlete_id       = self._athlete_id,
            baseline_fatigue = self._baseline_fatigue,
            baseline_load    = self._baseline_load,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _events_as_forecast_input(self) -> list[dict]:
        """Convert stored TwinEvents to dicts for PerformanceForecastEngine."""
        return [ev.to_dict() for ev in self._events]


# =============================================================================
# PRIVATE — state computation helpers
# =============================================================================

def _compute_injury_risk(fatigue: float, norm_recovery: float) -> float:
    """
    Compute injury risk from current fatigue and normalised recovery.

    Formula:
        clamp(W_FATIGUE_RISK * fatigue + W_RECOVERY_DEFICIT_RISK * (1 − norm_recovery))
    """
    raw = W_FATIGUE_RISK * fatigue + W_RECOVERY_DEFICIT_RISK * (1.0 - norm_recovery)
    return _clamp(raw, 0.0, 1.0)


def _compute_readiness(fatigue: float, injury_risk: float) -> float:
    """
    Compute readiness from current fatigue and injury risk.

    Formula:
        clamp(1 − fatigue − injury_risk * READINESS_INJURY_COEFF)
    """
    raw = 1.0 - fatigue - injury_risk * READINESS_INJURY_COEFF
    return _clamp(raw, 0.0, 1.0)


# =============================================================================
# PRIVATE — event validation
# =============================================================================

def _is_valid_event(ev: object) -> bool:
    """Return True when ev has all required fields with finite numeric values."""
    if not isinstance(ev, dict):
        return False
    for key in _REQUIRED_FIELDS:
        if key not in ev:
            return False
        val = ev[key]
        if not isinstance(val, (int, float)):
            return False
        if not math.isfinite(float(val)):
            return False
    return True


def _is_finite_num(v: object) -> bool:
    """Return True when v is a finite int or float."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


# =============================================================================
# PRIVATE — probe data
# =============================================================================

def _make_probe_events(n: int) -> list[dict]:
    """
    Generate a deterministic synthetic event list of length ``n``.

    Values follow a linear sweep — no randomness used.
    """
    events: list[dict] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        events.append({
            "fatigue_index":  round(0.2 + 0.6 * t, 6),
            "sprint_load":    round(2_000.0 + 5_000.0 * t, 2),
            "recovery_hours": round(40.0 - 34.0 * t, 2),
            "injury_flag":    1 if t > 0.7 else 0,
        })
    return events


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to ``[lo, hi]``.

    Returns ``lo`` for any non-finite input so a corrupt intermediate cannot
    propagate beyond the module boundary.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _finite_or_zero(value: float) -> float:
    """Return ``value`` when finite, otherwise ``0.0``."""
    return value if math.isfinite(value) else 0.0


def _run_check(name: str, fn) -> dict:
    """Execute a named boolean check, catching all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}