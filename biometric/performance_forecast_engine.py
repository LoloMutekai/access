"""
A.C.C.E.S.S. — Performance Forecast Engine (Phase B6 / 7.13)
biometric/performance_forecast_engine.py

Forecasts an athlete's short-term physiological state across three time
horizons (24 h, 48 h, 72 h) using a deterministic, bounded, closed-form
model derived from recent biometric events.

The engine is **stateless** — it holds no mutable data between calls.
All predictions are derived purely from the provided event list on each
invocation.  Long-term personal learning is delegated to the Athlete
Digital Twin (future phase).

───────────────────────────────────────────────────────────────────────────────
Pipeline position
───────────────────────────────────────────────────────────────────────────────

    raw signals
        → metrics
        → fatigue_index
        → injury_risk
        → anomaly_score
        → recommended_load
        → baseline_deviation
        → recovery_prediction
        → recovery_memory correction
        → rule_evolution_engine
        → performance_forecast_engine   ← THIS MODULE  (Phase 7.13)
        → Athlete Digital Twin

───────────────────────────────────────────────────────────────────────────────
Input events
───────────────────────────────────────────────────────────────────────────────

Each event must contain these fields (others are silently ignored):

    fatigue_index   ∈ [0, 1]   physiological fatigue score
    sprint_load     ≥ 0         session load (arbitrary units)
    recovery_hours  ≥ 0         rest hours before the session
    injury_flag     ∈ {0, 1}   ground-truth injury outcome

Invalid events (missing or non-finite fields) are silently skipped.

───────────────────────────────────────────────────────────────────────────────
Fatigue evolution model  (applied iteratively per 24-h step)
───────────────────────────────────────────────────────────────────────────────

    norm_load     = mean_sprint_load / LOAD_REFERENCE
    norm_recovery = clamp(mean_recovery_hours / 24.0, 0, 1)

    delta = ALPHA_LOAD * norm_load − BETA_RECOVERY * norm_recovery

    fatigue_t+24 = clamp(fatigue_t + delta, 0, 1)

    Constants:
        ALPHA_LOAD      = 0.35   — load-to-fatigue amplification
        BETA_RECOVERY   = 0.45   — recovery-to-fatigue reduction
        LOAD_REFERENCE  = 8000   — reference load for normalisation (AU)

    Starting point:
        fatigue_0 = mean(fatigue_index) across all valid events.
        Defaults to 0.0 (fully recovered) when no valid events exist.

───────────────────────────────────────────────────────────────────────────────
Injury risk forecast
───────────────────────────────────────────────────────────────────────────────

    injury_risk_forecast =
        clamp( W_FATIGUE * fatigue_24h
             + W_RECOVERY_DEFICIT * (1 − norm_recovery), 0, 1 )

    Constants:
        W_FATIGUE          = 0.6
        W_RECOVERY_DEFICIT = 0.4
        Sum                = 1.0 ✓

───────────────────────────────────────────────────────────────────────────────
Recovery forecast
───────────────────────────────────────────────────────────────────────────────

    recovery_forecast = clamp(1.0 − fatigue_24h, 0, 1)

    Interpretation: projected recovery capacity at the 24 h horizon.
    When fatigue is high the athlete has little capacity to recover further
    in the immediate window.

───────────────────────────────────────────────────────────────────────────────
Readiness score
───────────────────────────────────────────────────────────────────────────────

    readiness_score =
        clamp( 1.0 − fatigue_24h − injury_risk_forecast * 0.5, 0, 1 )

    0 → not ready to train
    1 → optimal readiness

───────────────────────────────────────────────────────────────────────────────
Default state (no valid events)
───────────────────────────────────────────────────────────────────────────────

    mean_fatigue  = DEFAULT_FATIGUE  = 0.0
    mean_load     = DEFAULT_LOAD     = 0.0
    mean_recovery = DEFAULT_RECOVERY = 24.0

    This reflects a fully rested, unloaded athlete — the most conservative
    (optimistic) prior.

───────────────────────────────────────────────────────────────────────────────
Mathematical guarantees
───────────────────────────────────────────────────────────────────────────────

    G1  fatigue_24h, fatigue_48h, fatigue_72h  ∈ [0, 1]
    G2  recovery_forecast                       ∈ [0, 1]
    G3  injury_risk_forecast                    ∈ [0, 1]
    G4  readiness_score                         ∈ [0, 1]
    G5  All output floats are finite
    G6  Deterministic — identical event list → identical forecast
    G7  Inputs never mutated
    G8  Output is JSON-serialisable
    G9  Bounded execution — O(|events|); no unbounded loops

───────────────────────────────────────────────────────────────────────────────
Design note: stateless engine
───────────────────────────────────────────────────────────────────────────────

    PerformanceForecastEngine.__init__() accepts no configuration.
    All four public methods accept the full event list independently
    (each re-derives the event summary internally for composability and
    independent testability).  forecast() is the primary integration entry
    point and computes all metrics in a single pass over the event list.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


# =============================================================================
# VERSION
# =============================================================================

ENGINE_NAME:    str = "PerformanceForecastEngine"
ENGINE_VERSION: str = "7.13.0"


# =============================================================================
# CONSTANTS
# =============================================================================

#: Forecast time horizons in hours.
FORECAST_HORIZONS: tuple[int, ...] = (24, 48, 72)

#: Load-to-fatigue amplification factor.
ALPHA_LOAD: float = 0.35

#: Recovery-to-fatigue reduction factor.
BETA_RECOVERY: float = 0.45

#: Reference load for normalisation (arbitrary units).
LOAD_REFERENCE: float = 8_000.0

#: Weight of fatigue in the injury risk formula.
W_FATIGUE: float = 0.6

#: Weight of recovery deficit in the injury risk formula.
W_RECOVERY_DEFICIT: float = 0.4

# Import-time sanity check — injury risk weights must sum to 1.0.
_IR_WEIGHT_SUM: float = W_FATIGUE + W_RECOVERY_DEFICIT
assert abs(_IR_WEIGHT_SUM - 1.0) < 1e-12, (
    f"Injury risk weights must sum to 1.0, got {_IR_WEIGHT_SUM}"
)

#: Coefficient applied to injury_risk in the readiness formula.
READINESS_INJURY_WEIGHT: float = 0.5

#: Default fatigue when no valid events are available.
DEFAULT_FATIGUE: float = 0.0

#: Default sprint load when no valid events are available.
DEFAULT_LOAD: float = 0.0

#: Default recovery hours when no valid events are available.
DEFAULT_RECOVERY: float = 24.0

#: Required fields in each biometric event.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "fatigue_index", "sprint_load", "recovery_hours", "injury_flag",
)

#: Maximum events processed per call (bounds execution time).
MAX_EVENTS: int = 10_000


# =============================================================================
# INTERNAL SUMMARY  (derived from a valid event list)
# =============================================================================

@dataclass(frozen=True)
class _EventSummary:
    """
    Aggregated statistics derived from a validated event list.

    Frozen and private — created by ``_compute_event_summary()`` and
    consumed by the projection helpers.  Never exposed to callers.

    Fields
    ──────
    mean_fatigue  : float  — mean fatigue_index across all valid events
    mean_load     : float  — mean sprint_load across all valid events
    mean_recovery : float  — mean recovery_hours across all valid events
    norm_load     : float  — mean_load / LOAD_REFERENCE  (unbounded above)
    norm_recovery : float  — clamp(mean_recovery / 24.0, 0, 1)
    event_count   : int    — number of valid events consumed
    """

    mean_fatigue:  float
    mean_load:     float
    mean_recovery: float
    norm_load:     float
    norm_recovery: float
    event_count:   int


# =============================================================================
# FORECAST RESULT  (immutable output snapshot)
# =============================================================================

@dataclass(frozen=True)
class ForecastResult:
    """
    Immutable snapshot of a complete 3-horizon performance forecast.

    Fields
    ──────
    fatigue_24h          : float  — projected fatigue at 24 h  ∈ [0, 1]
    fatigue_48h          : float  — projected fatigue at 48 h  ∈ [0, 1]
    fatigue_72h          : float  — projected fatigue at 72 h  ∈ [0, 1]
    recovery_forecast    : float  — projected recovery capacity ∈ [0, 1]
    injury_risk_forecast : float  — projected injury risk       ∈ [0, 1]
    readiness_score      : float  — projected training readiness ∈ [0, 1]

    The dataclass is frozen so snapshots can be safely shared, cached, or
    embedded in pipeline outputs without risk of accidental mutation.

    Invariants
    ──────────
    - All fields are finite.
    - All fields ∈ [0.0, 1.0].

    Example
    ───────
        engine = PerformanceForecastEngine()
        result_dict = engine.forecast(events)
        # {"fatigue_24h": ..., "fatigue_48h": ..., ...}
    """

    fatigue_24h:          float
    fatigue_48h:          float
    fatigue_72h:          float
    recovery_forecast:    float
    injury_risk_forecast: float
    readiness_score:      float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, float]:
        """
        Return a JSON-serialisable plain-dict representation.

        Returns
        ───────
        dict
            Keys: ``"fatigue_24h"``, ``"fatigue_48h"``, ``"fatigue_72h"``,
            ``"recovery_forecast"``, ``"injury_risk_forecast"``,
            ``"readiness_score"``.

        The returned dict is a fresh object — mutating it does not affect
        this frozen instance.
        """
        return {
            "fatigue_24h":          self.fatigue_24h,
            "fatigue_48h":          self.fatigue_48h,
            "fatigue_72h":          self.fatigue_72h,
            "recovery_forecast":    self.recovery_forecast,
            "injury_risk_forecast": self.injury_risk_forecast,
            "readiness_score":      self.readiness_score,
        }

    def is_valid(self) -> bool:
        """
        Return ``True`` when all six fields satisfy the documented invariants.

        Checks
        ──────
        - All fields are finite.
        - All fields ∈ [0.0, 1.0].
        """
        fields = (
            self.fatigue_24h, self.fatigue_48h, self.fatigue_72h,
            self.recovery_forecast, self.injury_risk_forecast, self.readiness_score,
        )
        return all(math.isfinite(f) and 0.0 <= f <= 1.0 for f in fields)


# =============================================================================
# PERFORMANCE FORECAST ENGINE  (stateless)
# =============================================================================

class PerformanceForecastEngine:
    """
    Deterministic, stateless performance forecast engine.

    Forecasts fatigue evolution, recovery needs, injury risk trend, and
    training readiness across 24 h, 48 h, and 72 h horizons.

    Stateless design
    ────────────────
    No mutable state is retained between calls.  Every public method
    derives its predictions entirely from the supplied event list.
    Future adaptation (personalisation over time) is handled by the
    Athlete Digital Twin module.

    Example
    ───────
        engine = PerformanceForecastEngine()
        result = engine.forecast(events)
        assert 0.0 <= result["fatigue_24h"] <= 1.0
    """

    def __init__(self) -> None:
        # No configuration parameters — engine is fully stateless.
        pass

    # ── Core public API ────────────────────────────────────────────────────────

    def forecast(self, events: list[dict]) -> dict[str, float]:
        """
        Compute a complete 3-horizon performance forecast.

        This is the primary integration entry point.  It processes the event
        list once and returns all six forecast metrics.

        Parameters
        ──────────
        events : list[dict]
            Biometric event records.  Each must contain
            ``fatigue_index``, ``sprint_load``, ``recovery_hours``,
            ``injury_flag``.  Invalid entries are silently skipped.
            The list is never mutated.

        Returns
        ───────
        dict[str, float]
            ``{"fatigue_24h", "fatigue_48h", "fatigue_72h",
               "recovery_forecast", "injury_risk_forecast",
               "readiness_score"}``
            All values ∈ [0.0, 1.0] and finite.
        """
        summary  = _compute_event_summary(_collect_valid_events(events))
        f24, f48, f72 = _project_fatigue(summary)
        ir  = _compute_injury_risk(f24, summary.norm_recovery)
        rec = _compute_recovery(f24)
        rdy = _compute_readiness(f24, ir)

        return ForecastResult(
            fatigue_24h          = f24,
            fatigue_48h          = f48,
            fatigue_72h          = f72,
            recovery_forecast    = rec,
            injury_risk_forecast = ir,
            readiness_score      = rdy,
        ).to_dict()

    def forecast_fatigue(self, events: list[dict]) -> dict[str, float]:
        """
        Return only the three fatigue horizon projections.

        Returns
        ───────
        dict[str, float]
            ``{"fatigue_24h", "fatigue_48h", "fatigue_72h"}``
            All values ∈ [0.0, 1.0] and finite.
        """
        summary = _compute_event_summary(_collect_valid_events(events))
        f24, f48, f72 = _project_fatigue(summary)
        return {
            "fatigue_24h": f24,
            "fatigue_48h": f48,
            "fatigue_72h": f72,
        }

    def forecast_recovery(self, events: list[dict]) -> float:
        """
        Return the projected recovery capacity at the 24 h horizon.

        Formula: ``clamp(1.0 − fatigue_24h, 0, 1)``.

        Returns
        ───────
        float
            Recovery forecast ∈ [0.0, 1.0].
        """
        summary = _compute_event_summary(_collect_valid_events(events))
        f24, _, _ = _project_fatigue(summary)
        return _compute_recovery(f24)

    def forecast_injury_risk(self, events: list[dict]) -> float:
        """
        Return the projected injury risk score at the 24 h horizon.

        Formula:
            ``clamp(W_FATIGUE * fatigue_24h
                    + W_RECOVERY_DEFICIT * (1 − norm_recovery), 0, 1)``

        Returns
        ───────
        float
            Injury risk forecast ∈ [0.0, 1.0].
        """
        summary = _compute_event_summary(_collect_valid_events(events))
        f24, _, _ = _project_fatigue(summary)
        return _compute_injury_risk(f24, summary.norm_recovery)

    def compute_readiness(self, events: list[dict]) -> float:
        """
        Return the projected training readiness score at the 24 h horizon.

        Formula:
            ``clamp(1.0 − fatigue_24h − injury_risk_forecast * 0.5, 0, 1)``

        Returns
        ───────
        float
            Readiness score ∈ [0.0, 1.0].
        """
        summary = _compute_event_summary(_collect_valid_events(events))
        f24, _, _ = _project_fatigue(summary)
        ir  = _compute_injury_risk(f24, summary.norm_recovery)
        return _compute_readiness(f24, ir)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run six named invariant checks on synthetic probe data.

        The engine is stateless so no live state is at risk.

        Checks
        ──────
        1. ``output_keys_always_present``
        2. ``all_outputs_in_unit_interval``
        3. ``empty_events_uses_defaults``
        4. ``high_fatigue_high_load_raises_forecast``
        5. ``json_serialisable``
        6. ``deterministic_output``

        Returns
        ───────
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        checks: list[dict] = []

        # 1 — required keys always present
        def _keys() -> bool:
            r = PerformanceForecastEngine().forecast([])
            return set(r) == {
                "fatigue_24h", "fatigue_48h", "fatigue_72h",
                "recovery_forecast", "injury_risk_forecast", "readiness_score",
            }
        checks.append(_run_check("output_keys_always_present", _keys))

        # 2 — all outputs ∈ [0, 1]
        def _unit_interval() -> bool:
            events = _make_probe_events(40)
            r = PerformanceForecastEngine().forecast(events)
            return all(0.0 <= v <= 1.0 for v in r.values())
        checks.append(_run_check("all_outputs_in_unit_interval", _unit_interval))

        # 3 — empty events → defaults produce valid output
        def _empty_defaults() -> bool:
            r = PerformanceForecastEngine().forecast([])
            # With default load=0, recovery=24: norm_load=0, norm_recovery=1
            # delta = 0.35*0 - 0.45*1 = -0.45 → clamped: f24=0, f48=0, f72=0
            # recovery=1, injury_risk=0.6*0+0.4*0=0, readiness=1-0-0=1
            return r["fatigue_24h"] == 0.0 and r["readiness_score"] == 1.0
        checks.append(_run_check("empty_events_uses_defaults", _empty_defaults))

        # 4 — high fatigue, high load, no recovery → high forecast fatigue
        def _high_load_raises() -> bool:
            events = [{"fatigue_index": 0.9, "sprint_load": 8000.0,
                       "recovery_hours": 0.0, "injury_flag": 1}] * 20
            r = PerformanceForecastEngine().forecast(events)
            return r["fatigue_24h"] >= 0.9  # cannot decrease with zero recovery
        checks.append(_run_check("high_fatigue_high_load_raises_forecast", _high_load_raises))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            events = _make_probe_events(30)
            r = PerformanceForecastEngine().forecast(events)
            back = json.loads(json.dumps(r))
            return all(abs(back[k] - r[k]) < 1e-9 for k in r)
        checks.append(_run_check("json_serialisable", _json_rt))

        # 6 — determinism
        def _det() -> bool:
            events = _make_probe_events(50)
            return (
                PerformanceForecastEngine().forecast(events)
                == PerformanceForecastEngine().forecast(events)
            )
        checks.append(_run_check("deterministic_output", _det))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  ENGINE_NAME,
            "version": ENGINE_VERSION,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run the canonical probe sequence twice and verify bit-for-bit identity.

        Returns
        ───────
        bool
            ``True`` when both runs produce identical results.
        """
        events = _make_probe_events(60)
        return (
            PerformanceForecastEngine().forecast(events)
            == PerformanceForecastEngine().forecast(events)
        )


# =============================================================================
# PRIVATE — event collection and validation
# =============================================================================

def _collect_valid_events(raw_events: list[dict]) -> list[dict]:
    """
    Validate and return up to MAX_EVENTS events.

    Each event must contain all ``_REQUIRED_FIELDS`` with finite numeric
    values.  The input list is never mutated.

    Parameters
    ──────────
    raw_events : list[dict]   Candidate event records.

    Returns
    ───────
    list[dict]
        Validated events; each has all required fields with finite values.
    """
    valid: list[dict] = []
    for ev in raw_events:
        if len(valid) >= MAX_EVENTS:
            break
        if _is_valid_event(ev):
            valid.append(ev)
    return valid


def _is_valid_event(ev: object) -> bool:
    """Return True when ``ev`` has all required fields with finite numeric values."""
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


# =============================================================================
# PRIVATE — event aggregation
# =============================================================================

def _compute_event_summary(events: list[dict]) -> _EventSummary:
    """
    Aggregate a validated event list into an ``_EventSummary``.

    All arithmetic is finite-guarded.  Defaults are applied when no valid
    events are present so downstream projections always receive finite inputs.

    Parameters
    ──────────
    events : list[dict]   Pre-validated events (from ``_collect_valid_events``).

    Returns
    ───────
    _EventSummary
        Immutable aggregate; all fields finite.
    """
    if not events:
        return _EventSummary(
            mean_fatigue  = DEFAULT_FATIGUE,
            mean_load     = DEFAULT_LOAD,
            mean_recovery = DEFAULT_RECOVERY,
            norm_load     = _finite_or_zero(DEFAULT_LOAD / LOAD_REFERENCE),
            norm_recovery = _clamp(DEFAULT_RECOVERY / 24.0, 0.0, 1.0),
            event_count   = 0,
        )

    n = len(events)
    mean_fatigue  = sum(_finite_or_zero(float(ev["fatigue_index"]))  for ev in events) / n
    mean_load     = sum(_finite_or_zero(float(ev["sprint_load"]))    for ev in events) / n
    mean_recovery = sum(_finite_or_zero(float(ev["recovery_hours"])) for ev in events) / n

    # Clamp aggregated means to physiologically sensible ranges.
    mean_fatigue  = _clamp(mean_fatigue,  0.0, 1.0)
    mean_load     = max(0.0, mean_load)
    mean_recovery = max(0.0, mean_recovery)

    norm_load     = _finite_or_zero(mean_load / LOAD_REFERENCE)   # unbounded above; clamp handled by fatigue projection
    norm_recovery = _clamp(mean_recovery / 24.0, 0.0, 1.0)

    return _EventSummary(
        mean_fatigue  = mean_fatigue,
        mean_load     = mean_load,
        mean_recovery = mean_recovery,
        norm_load     = norm_load,
        norm_recovery = norm_recovery,
        event_count   = n,
    )


# =============================================================================
# PRIVATE — fatigue projection
# =============================================================================

def _step_fatigue(fatigue_current: float, delta: float) -> float:
    """
    Apply one 24-h fatigue step.

    Parameters
    ──────────
    fatigue_current : float  — fatigue level entering the step  ∈ [0, 1]
    delta           : float  — net change = ALPHA_LOAD*norm_load − BETA_RECOVERY*norm_recovery

    Returns
    ───────
    float
        Projected fatigue ∈ [0.0, 1.0].
    """
    return _clamp(fatigue_current + delta, 0.0, 1.0)


def _project_fatigue(
    summary: _EventSummary,
) -> tuple[float, float, float]:
    """
    Project fatigue across 24 h, 48 h, and 72 h horizons.

    The same ``delta`` is applied at each step, reflecting the assumption
    that future load and recovery patterns mirror historical averages.

    Returns
    ───────
    tuple[float, float, float]
        (fatigue_24h, fatigue_48h, fatigue_72h), all ∈ [0.0, 1.0].
    """
    delta = ALPHA_LOAD * summary.norm_load - BETA_RECOVERY * summary.norm_recovery

    f0  = summary.mean_fatigue
    f24 = _step_fatigue(f0,  delta)
    f48 = _step_fatigue(f24, delta)
    f72 = _step_fatigue(f48, delta)

    return f24, f48, f72


# =============================================================================
# PRIVATE — derived forecast metrics
# =============================================================================

def _compute_injury_risk(fatigue_24h: float, norm_recovery: float) -> float:
    """
    Compute projected injury risk from 24 h fatigue and recovery capacity.

    Formula:
        ``clamp(W_FATIGUE * fatigue_24h + W_RECOVERY_DEFICIT * (1 − norm_recovery), 0, 1)``

    Parameters
    ──────────
    fatigue_24h   : float  — projected 24 h fatigue  ∈ [0, 1]
    norm_recovery : float  — normalised recovery rate ∈ [0, 1]

    Returns
    ───────
    float
        Injury risk ∈ [0.0, 1.0].
    """
    raw = W_FATIGUE * fatigue_24h + W_RECOVERY_DEFICIT * (1.0 - norm_recovery)
    return _clamp(raw, 0.0, 1.0)


def _compute_recovery(fatigue_24h: float) -> float:
    """
    Compute projected recovery capacity from the 24 h fatigue forecast.

    Formula: ``clamp(1.0 − fatigue_24h, 0, 1)``

    Parameters
    ──────────
    fatigue_24h : float  — projected 24 h fatigue  ∈ [0, 1]

    Returns
    ───────
    float
        Recovery forecast ∈ [0.0, 1.0].
    """
    return _clamp(1.0 - fatigue_24h, 0.0, 1.0)


def _compute_readiness(fatigue_24h: float, injury_risk: float) -> float:
    """
    Compute projected training readiness from 24 h fatigue and injury risk.

    Formula: ``clamp(1.0 − fatigue_24h − injury_risk * 0.5, 0, 1)``

    Parameters
    ──────────
    fatigue_24h  : float  — projected 24 h fatigue  ∈ [0, 1]
    injury_risk  : float  — projected injury risk   ∈ [0, 1]

    Returns
    ───────
    float
        Readiness score ∈ [0.0, 1.0].
    """
    raw = 1.0 - fatigue_24h - injury_risk * READINESS_INJURY_WEIGHT
    return _clamp(raw, 0.0, 1.0)


# =============================================================================
# PRIVATE — probe data for diagnostics
# =============================================================================

def _make_probe_events(n: int) -> list[dict]:
    """
    Return a deterministic synthetic event list of length ``n``.

    Values follow a linear sweep from low-fatigue/low-load to
    high-fatigue/high-load.  No randomness is used.
    """
    events: list[dict] = []
    for i in range(n):
        t = i / max(n - 1, 1)        # 0.0 → 1.0 linear sweep
        events.append({
            "fatigue_index":  round(0.2 + 0.65 * t, 6),
            "sprint_load":    round(2_000.0 + 6_000.0 * t, 2),
            "recovery_hours": round(40.0 - 36.0 * t, 2),
            "injury_flag":    1 if t > 0.65 else 0,
        })
    return events


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to ``[lo, hi]``.

    Returns ``lo`` for any non-finite input (NaN, ±Inf) so corrupt
    intermediates can never propagate out of the module boundary.
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