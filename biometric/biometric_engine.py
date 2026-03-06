"""
A.C.C.E.S.S. — BiometricEngine (Phase 7.10.0 — DATA ENGINE INTEGRATION)

Deterministic biometric signal processor conforming to the DomainEngine contract.

Architecture:
    BiometricEngine
        ├── validate_schema()          — structural + type guards, no side-effects
        ├── clean_data()               — finite-value filter, length enforcement
        ├── normalize()                — min-max scaling per channel, [0, 1]
        ├── compute_core_metrics()     — descriptive statistics over clean signals
        ├── compute_fatigue_index()    — weighted sigmoid fatigue score in [0, 1]
        ├── compute_drift()            — weighted drift score in [0, 1]
        ├── compute_injury_risk()      — weighted injury risk score in [0, 1]
        ├── compute_anomaly_signals()  — normalised anomaly signal decomposition
        ├── compute_anomaly()          — weighted anomaly score in [0, 1]
        ├── compute_training_state()   — priority-rule training state classifier
        └── compute_recommended_load() — load multiplier in [0, 1] for training state
        └── compute_training_advice()  — structured TrainingRecommendation dataclass

    Optional DataEngine integration (Phase 7.10.0)
    -----------------------------------------------
    BiometricEngine(data_engine=<engine>) accepts any object that implements
    ``run({"values": list[float]}) → dict``.  When provided it is called on the
    ``"load"`` channel inside ``process()`` and the extracted features are
    surfaced under the top-level ``"data_features"`` key of the output dict.
    When no engine is provided ``"data_features"`` is always ``{}``.

    Module-level functions (also exposed as engine methods):
        compute_training_state(metrics)          → str
        compute_recommended_load(training_state) → float
        compute_training_advice(metrics)         → TrainingRecommendation

Design invariants (all enforced by tests):
    - Deterministic  : same input -> identical output, always
    - No randomness  : no random / uuid / time calls in the hot path
    - No ML          : pure arithmetic only
    - No global state: BiometricEngine is self-contained; no module-level mutation
    - JSON-safe      : every float in output is finite; ints are plain ints
    - Sandbox-safe   : stdlib only (math) — no numpy, no scipy

Backward compatibility:
    7.3.0 -> 7.3.1 : added "fatigue_index"
    7.3.1 -> 7.4.0 : added "drift_score", "injury_risk"
    7.4.0 -> 7.5.0 : compute_anomaly_signals() + AnomalySignals (no new output key)
    7.5.0 -> 7.6.0 : added "anomaly_score"; AnomalyResult + compute_anomaly()
    7.6.0 -> 7.7.0 : added "training_state"; compute_training_state() + module fn
    7.7.0 -> 7.8.0 : added "recommended_load"; compute_recommended_load() + module fn
    7.8.0 -> 7.9.0  : added "training_advice"; TrainingRecommendation + compute_training_advice()
    7.9.0 -> 7.10.0 : optional DataEngine integration; top-level "data_features" key in output
    All prior keys are unchanged in position and semantics.

Input schema
------------
    {
        "hr":   list[float],   # heart-rate samples (bpm)     — length >= 10
        "hrv":  list[float],   # HRV samples (ms)             — length >= 10
        "load": list[float],   # training-load samples (AU)   — length >= 10
    }

Output schema (v7.10.0)
-----------------------
    {
        "status":  "ok" | "error",
        "metrics": {
            "mean_hr":          float,
            "mean_hrv":         float,
            "load_mean":        float,
            "hr_std":           float,
            "hrv_std":          float,
            "fatigue_index":    float,
            "drift_score":      float,
            "injury_risk":      float,
            "anomaly_score":    float,
            "training_state":   str,
            "recommended_load": float,
            "training_advice":  {"state": str, "recommended_load": float},
        } | null,
        "data_features": dict,   # DataEngine features on load channel  <-- NEW 7.10.0
                                 # always present; {} when no data_engine configured
        "engine":  "BiometricEngine",
        "version": "7.10.0",
    }

    On error: "metrics" is null, "error" carries the message.
    "data_features" is always present (empty dict on error or when no data_engine).

Fatigue model (unchanged from 7.3.1)
--------------------------------------
    fatigue_raw = W1*norm_load_mean + W2*norm_hr_std - W3*norm_hrv_mean
    fatigue     = clamp(sigmoid(fatigue_raw), 0.0, 1.0)
    W1=0.5, W2=0.3, W3=0.2

Drift model (unchanged from 7.4.0)
-------------------------------------
    fatigue_drift = abs(fatigue.value - 0.5)   # deviation from neutral [0,0.5]
    hr_drift      = abs(metrics.hr_std)         # raw-scale HR variability
    hrv_drift     = abs(metrics.hrv_std)        # raw-scale HRV variability

    drift_score = clamp(DW1*fatigue_drift + DW2*hr_drift + DW3*hrv_drift, 0, 1)
    DW1=0.5, DW2=0.3, DW3=0.2

Injury risk model (unchanged from 7.4.0)
------------------------------------------
    fatigue_signal = fatigue.value                         # [0, 1]
    drift_signal   = drift.score                           # [0, 1]
    load_signal    = clamp(metrics.load_mean / 1000, 0, 1) # normalised load

    risk_raw   = RW1*fatigue_signal + RW2*drift_signal + RW3*load_signal
    risk_score = clamp(risk_raw, 0.0, 1.0)
    RW1=0.45, RW2=0.35, RW3=0.20

Anomaly signals model (unchanged from 7.5.0)
----------------------------------------------
    Decomposes the pipeline state into four normalised [0, 1] signals consumed
    by compute_anomaly().  compute_anomaly_signals() is called inside process().

    fatigue_signal = fatigue.value                                  # [0, 1]
    drift_signal   = drift.score                                    # [0, 1]
    hr_spread      = clamp(metrics.hr_std  / ANOMALY_HR_REF, 0, 1) # [0, 1]
    hrv_spread     = clamp(metrics.hrv_std / ANOMALY_HRV_REF, 0, 1)# [0, 1]

    Reference scales (module-level constants — never mutated):
        ANOMALY_HR_REF  = 50.0  bpm
        ANOMALY_HRV_REF = 100.0 ms

Anomaly scoring model (new in 7.6.0)
--------------------------------------
    anomaly_raw = AW_F * fatigue_signal
                + AW_D * drift_signal
                + AW_H * hr_spread
                + AW_V * hrv_spread
    anomaly_score = clamp(anomaly_raw, 0.0, 1.0)

    Weights: AW_F=0.40, AW_D=0.30, AW_H=0.20, AW_V=0.10  (sum to 1.0)

    All four inputs are already in [0, 1]; the weighted sum is inherently in
    [0, 1] before clamping.  The clamp is a belt-and-suspenders float guard.

Training state model (new in 7.7.0)
--------------------------------------
    Priority-ordered rule classifier over the three composite scores.
    Rules are evaluated top-to-bottom; the first match wins.

    if   injury_risk   >= TS_RECOVERY_THRESHOLD (0.65) → "RECOVERY"
    elif fatigue_index >= TS_LIGHT_THRESHOLD    (0.70) → "LIGHT"
    elif anomaly_score >= TS_CAUTION_THRESHOLD  (0.60) → "CAUTION"
    else                                               → "FULL"

    Return type  : str
    Allowed values: {"FULL", "CAUTION", "LIGHT", "RECOVERY"}  (VALID_TRAINING_STATES)
    Guarantee    : always returns exactly one valid state; never raises.

Recommended load model (new in 7.8.0)
--------------------------------------
    Maps each training state to a normalised load multiplier in [0, 1].
    The mapping is a closed lookup — no arithmetic, purely tabular.

    "FULL"     → 1.00   (no restriction)
    "CAUTION"  → 0.75   (moderate restriction)
    "LIGHT"    → 0.50   (significant restriction)
    "RECOVERY" → 0.25   (maximum restriction)

    Invalid state → raises ValueError.
    Result is clamped to [0, 1] as a belt-and-suspenders float guard.
    Input is never mutated.

Training advice model (new in 7.9.0)
--------------------------------------
    Bundles the training state and recommended load into a single immutable
    TrainingRecommendation dataclass.  compute_training_advice() is a pure
    function — it calls compute_training_state() then compute_recommended_load()
    and returns the result with no additional arithmetic.

    TrainingRecommendation fields
    ------------------------------
    state            : str    — one of VALID_TRAINING_STATES
    recommended_load : float  — corresponding load multiplier in [0, 1]

    to_dict() → {"state": str, "recommended_load": float}
        Returns a plain JSON-serialisable dict; does not mutate self.

Usage
-----
    # No DataEngine — data_features is always {}
    engine = BiometricEngine()
    result = engine.process({"hr": [...], "hrv": [...], "load": [...]})
    assert result["status"] == "ok"
    assert result["data_features"] == {}
    print(result["metrics"]["training_advice"])   # {"state": ..., "recommended_load": ...}
    print(result["metrics"]["training_state"])    # str in VALID_TRAINING_STATES

    # With optional DataEngine (duck-typed, must implement run())
    from agent.domain.data_engine import DataEngine
    engine = BiometricEngine(data_engine=DataEngine())
    result = engine.process({"hr": [...], "hrv": [...], "load": [...]})
    print(result["data_features"])                # dict of load-channel features  <-- NEW
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

ENGINE_NAME: str = "BiometricEngine"
ENGINE_VERSION: str = "7.10.0"

#: Minimum number of valid (finite) samples required per channel after cleaning.
MIN_SAMPLES: int = 10

#: Recognised top-level channel keys.
REQUIRED_CHANNELS: tuple[str, ...] = ("hr", "hrv", "load")

# Fatigue model weights (frozen by convention — never reassigned at runtime).
FATIGUE_W1: float = 0.5   # load contribution
FATIGUE_W2: float = 0.3   # HR std contribution
FATIGUE_W3: float = 0.2   # HRV mean contribution (inverted)

# Drift model weights (frozen by convention — never reassigned at runtime).
DRIFT_W_FATIGUE: float = 0.5   # fatigue deviation from neutral (dominant)
DRIFT_W_HR: float      = 0.3   # raw HR std — exertion spread
DRIFT_W_HRV: float     = 0.2   # raw HRV std — recovery spread

# Injury risk model weights (frozen by convention — never reassigned at runtime).
RISK_W_FATIGUE: float = 0.45   # fatigue signal — dominant driver
RISK_W_DRIFT: float   = 0.35   # drift signal   — instability marker
RISK_W_LOAD: float    = 0.20   # normalised load — cumulative stress

# Anomaly signal reference scales (frozen by convention — never reassigned at runtime).
#   hr_spread  = clamp(hr_std  / ANOMALY_HR_REF,  0, 1)   — 50 bpm ceiling
#   hrv_spread = clamp(hrv_std / ANOMALY_HRV_REF, 0, 1)   — 100 ms ceiling
ANOMALY_HR_REF: float  = 50.0    # HR std reference scale  (bpm)
ANOMALY_HRV_REF: float = 100.0   # HRV std reference scale (ms)

# Anomaly scoring weights (frozen by convention — never reassigned at runtime).
# Sum to 1.0; all four input signals are already in [0, 1], so anomaly_raw ∈ [0, 1].
ANOMALY_W_FATIGUE: float = 0.40   # fatigue_signal — dominant driver
ANOMALY_W_DRIFT: float   = 0.30   # drift_signal   — instability contribution
ANOMALY_W_HR: float      = 0.20   # hr_spread      — exertion variability
ANOMALY_W_HRV: float     = 0.10   # hrv_spread     — recovery variability

# Training state thresholds (frozen by convention — never reassigned at runtime).
# Priority order: RECOVERY > LIGHT > CAUTION > FULL  (first match wins).
TS_RECOVERY_THRESHOLD: float = 0.65   # injury_risk   >= this → "RECOVERY"
TS_LIGHT_THRESHOLD:    float = 0.70   # fatigue_index >= this → "LIGHT"
TS_CAUTION_THRESHOLD:  float = 0.60   # anomaly_score >= this → "CAUTION"

#: Complete set of valid training state strings.  Membership is enforced by
#: compute_training_state() — the function always returns one of these values.
VALID_TRAINING_STATES: frozenset[str] = frozenset({"FULL", "CAUTION", "LIGHT", "RECOVERY"})

# Recommended load mapping (frozen by convention — never reassigned or mutated at runtime).
# Maps each valid training state to a normalised load multiplier in [0, 1].
# Lookup table; no arithmetic.  All values are pre-validated members of [0, 1].
RECOMMENDED_LOAD: dict[str, float] = {
    "FULL":     1.00,   # no restriction
    "CAUTION":  0.75,   # moderate restriction
    "LIGHT":    0.50,   # significant restriction
    "RECOVERY": 0.25,   # maximum restriction
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class BiometricConfig:
    """
    All thresholds for the BiometricEngine.

    Frozen — never mutated after construction.
    All fields have safe, physiologically reasonable defaults.
    """

    #: Minimum valid samples per channel (after cleaning).
    min_samples: int = MIN_SAMPLES

    #: Physiological HR range guard (bpm).
    #: Applied only when strict_guards=True.
    hr_min: float = 20.0
    hr_max: float = 300.0

    #: Physiological HRV range guard (ms).
    hrv_min: float = 0.0
    hrv_max: float = 3000.0

    #: Load range guard (arbitrary units).
    load_min: float = 0.0
    load_max: float = 10_000.0

    #: When True, out-of-range values are scrubbed during clean_data().
    #: When False, only non-finite values are removed.
    strict_guards: bool = False

    #: Rounding precision for output floats.
    output_precision: int = 6


# =============================================================================
# INTERNAL DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class CleanedSignals:
    """
    Holds the cleaned (finite-only) channel arrays.
    Immutable — passed through the pipeline without mutation.
    """

    hr: tuple[float, ...]
    hrv: tuple[float, ...]
    load: tuple[float, ...]

    def channel(self, name: str) -> tuple[float, ...]:
        return getattr(self, name)


@dataclass(frozen=True)
class NormalizedSignals:
    """
    Holds min-max normalised channels in [0.0, 1.0].
    When all values in a channel are equal the channel collapses to all-zeros.
    """

    hr: tuple[float, ...]
    hrv: tuple[float, ...]
    load: tuple[float, ...]

    def channel(self, name: str) -> tuple[float, ...]:
        return getattr(self, name)


@dataclass(frozen=True)
class CoreMetrics:
    """
    Five core descriptive statistics computed over cleaned (raw) channels.
    All floats are guaranteed finite by construction.
    """

    mean_hr: float
    mean_hrv: float
    load_mean: float
    hr_std: float
    hrv_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_hr":   self.mean_hr,
            "mean_hrv":  self.mean_hrv,
            "load_mean": self.load_mean,
            "hr_std":    self.hr_std,
            "hrv_std":   self.hrv_std,
        }


@dataclass(frozen=True)
class FatigueResult:
    """
    Single fatigue score in [0, 1] plus the intermediate raw value for
    auditability and future drift tracking.

    value     — clamped sigmoid output, always in [0, 1]
    raw       — linear combination before sigmoid (unbounded, for diagnostics)
    """

    value: float
    raw: float

    def to_dict(self) -> dict[str, float]:
        # Only "value" is exposed in the public output; "raw" is internal.
        return {"fatigue_index": self.value}


@dataclass(frozen=True)
class DriftResult:
    """
    Weighted drift score in [0, 1].

    score      — final clamped drift score, always in [0, 1]
    components — raw component values before weighting (diagnostic):
                 (fatigue_drift, hr_drift, hrv_drift)
                 NOT individually clamped; preserves diagnostic range.
    """

    score: float
    components: tuple[float, float, float]   # (fatigue_drift, hr_drift, hrv_drift)

    def to_dict(self) -> dict[str, float]:
        return {"drift_score": self.score}


@dataclass(frozen=True)
class InjuryRiskResult:
    """
    Weighted injury risk score in [0, 1].

    score   — final clamped risk score, always in [0, 1]
    signals — the three input signals used (diagnostic):
              (fatigue_signal, drift_signal, load_signal)
              All three are already in [0, 1] by construction.
    """

    score: float
    signals: tuple[float, float, float]   # (fatigue_signal, drift_signal, load_signal)

    def to_dict(self) -> dict[str, float]:
        return {"injury_risk": self.score}


@dataclass(frozen=True)
class AnomalySignals:
    """
    Four normalised anomaly signals, each guaranteed in [0, 1].

    This is an intermediate decomposition produced by compute_anomaly_signals()
    for downstream anomaly scoring.  It is NOT emitted in process() output —
    callers invoke compute_anomaly_signals() directly when the breakdown is needed.

    Fields
    ------
    fatigue_signal — direct pass-through of fatigue.value          in [0, 1]
    drift_signal   — direct pass-through of drift.score            in [0, 1]
    hr_spread      — clamp(hr_std  / ANOMALY_HR_REF,  0.0, 1.0)   in [0, 1]
    hrv_spread     — clamp(hrv_std / ANOMALY_HRV_REF, 0.0, 1.0)   in [0, 1]

    All four fields are finite and in [0, 1] by construction; this invariant
    is enforced by compute_anomaly_signals() before the dataclass is created.
    """

    fatigue_signal: float
    drift_signal:   float
    hr_spread:      float
    hrv_spread:     float


@dataclass(frozen=True)
class AnomalyResult:
    """
    Weighted anomaly score in [0, 1] plus the AnomalySignals used to derive it.

    score   — final clamped anomaly score, always in [0, 1]
    signals — the AnomalySignals snapshot consumed by compute_anomaly()

    Formula (weights sum to 1.0):
        anomaly_raw   = AW_F * signals.fatigue_signal
                      + AW_D * signals.drift_signal
                      + AW_H * signals.hr_spread
                      + AW_V * signals.hrv_spread
        anomaly_score = clamp(anomaly_raw, 0.0, 1.0)

    AW_F=0.40, AW_D=0.30, AW_H=0.20, AW_V=0.10  (sum to 1.0)

    All inputs are in [0, 1] → anomaly_raw ∈ [0, 1] before clamping; the
    clamp is a belt-and-suspenders float-rounding guard.
    """

    score:   float
    signals: AnomalySignals

    def to_dict(self) -> dict[str, float]:
        # Only the composite score is exposed in the public output dict.
        return {"anomaly_score": self.score}


@dataclass(frozen=True)
class TrainingRecommendation:
    """
    Immutable structured training recommendation produced by
    ``compute_training_advice()``.

    Bundles the training state and its corresponding load multiplier into
    a single, self-describing, JSON-serialisable object.

    Fields
    ------
    state            : str   — training state; always in ``VALID_TRAINING_STATES``
    recommended_load : float — load multiplier in [0, 1] for ``state``

    Invariants (guaranteed by ``compute_training_advice()`` before construction)
    ---------------------------------------------------------------------------
    - ``state`` is always a member of ``VALID_TRAINING_STATES``.
    - ``recommended_load`` is always finite and in [0.0, 1.0].
    - The dataclass is frozen — no field can be reassigned after construction.
    """

    state:            str
    recommended_load: float

    def to_dict(self) -> dict[str, object]:
        """
        Return a JSON-serialisable plain-dict representation.

        Returns
        -------
        dict
            ``{"state": str, "recommended_load": float}``

        The returned dict is a fresh object; mutating it does not affect
        this frozen instance.
        """
        return {
            "state":            self.state,
            "recommended_load": self.recommended_load,
        }


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class SchemaError(ValueError):
    """Raised (internally) when input fails validate_schema()."""


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]. Deterministic, no branching on NaN."""
    return max(lo, min(hi, value))


def _bounded_sigmoid(z: float) -> float:
    """
    Numerically stable sigmoid: 1 / (1 + exp(-z)).

    Handles extreme z values without overflow:
        z >> 0  -> value approaches 1.0 (but never exceeds it)
        z << 0  -> value approaches 0.0 (but never goes negative)

    Returns a finite float in (0, 1).
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    # For large negative z, exp(-z) is large — use the equivalent form
    # 1/(1+e^-z) = e^z / (1 + e^z) which is numerically safer when z < 0.
    e = math.exp(z)
    return e / (1.0 + e)


def _round(value: float, precision: int) -> float:
    """Round to precision decimal places; return value as plain float."""
    return round(float(value), precision)


def compute_training_state(metrics: dict) -> str:
    """
    Classify the current training state from the three composite scores.

    This is a module-level pure function — it reads ``metrics`` and returns a
    str without mutating anything or touching global state.  It is also exposed
    as ``BiometricEngine.compute_training_state()`` for pipeline convenience.

    Parameters
    ----------
    metrics : dict
        Must contain ``"injury_risk"``, ``"fatigue_index"``, and
        ``"anomaly_score"`` as float values.  Extra keys are silently ignored;
        the dict is never mutated.

    Rules  (priority order — first match wins)
    -------------------------------------------
    1. injury_risk   >= TS_RECOVERY_THRESHOLD (0.65) → ``"RECOVERY"``
    2. fatigue_index >= TS_LIGHT_THRESHOLD    (0.70) → ``"LIGHT"``
    3. anomaly_score >= TS_CAUTION_THRESHOLD  (0.60) → ``"CAUTION"``
    4. (default)                                     → ``"FULL"``

    Returns
    -------
    str
        Always one of ``{"FULL", "CAUTION", "LIGHT", "RECOVERY"}``
        (i.e. always a member of ``VALID_TRAINING_STATES``).

    Invariants
    ----------
    - Return type is always ``str``.
    - Return value is always in ``VALID_TRAINING_STATES``.
    - The function always returns — it never raises.
    - ``metrics`` is never mutated.
    - Deterministic: identical inputs → identical output, always.
    """
    injury_risk:   float = metrics["injury_risk"]
    fatigue_index: float = metrics["fatigue_index"]
    anomaly_score: float = metrics["anomaly_score"]

    if injury_risk >= TS_RECOVERY_THRESHOLD:
        return "RECOVERY"
    if fatigue_index >= TS_LIGHT_THRESHOLD:
        return "LIGHT"
    if anomaly_score >= TS_CAUTION_THRESHOLD:
        return "CAUTION"
    return "FULL"


def compute_recommended_load(training_state: str) -> float:
    """
    Map a training state string to a normalised load multiplier in [0, 1].

    This is a module-level pure function — it reads ``training_state`` and
    returns a ``float`` without mutating anything or touching global state.
    It is also exposed as ``BiometricEngine.compute_recommended_load()`` for
    pipeline convenience.

    Parameters
    ----------
    training_state : str
        Must be one of ``VALID_TRAINING_STATES``
        (``"FULL"``, ``"CAUTION"``, ``"LIGHT"``, ``"RECOVERY"``).

    Returns
    -------
    float
        Load multiplier clamped to ``[0.0, 1.0]``:

        +-----------+---------+
        | State     |  Value  |
        +===========+=========+
        | FULL      |  1.00   |
        | CAUTION   |  0.75   |
        | LIGHT     |  0.50   |
        | RECOVERY  |  0.25   |
        +-----------+---------+

    Raises
    ------
    ValueError
        If ``training_state`` is not a member of ``VALID_TRAINING_STATES``.

    Invariants
    ----------
    - Return type is always ``float``.
    - Return value is always in ``[0.0, 1.0]``.
    - Raises ``ValueError`` for any invalid state — never silently returns.
    - ``training_state`` is never mutated.
    - Deterministic: identical inputs → identical output, always.
    """
    if training_state not in VALID_TRAINING_STATES:
        raise ValueError(
            f"Invalid training_state {training_state!r}. "
            f"Must be one of {sorted(VALID_TRAINING_STATES)}."
        )
    return _clamp(RECOMMENDED_LOAD[training_state], 0.0, 1.0)


def compute_training_advice(metrics: dict) -> "TrainingRecommendation":
    """
    Compute a structured ``TrainingRecommendation`` from a metrics dict.

    Pure module-level function — reads ``metrics``, calls
    ``compute_training_state()`` then ``compute_recommended_load()``, and
    returns the result as a frozen ``TrainingRecommendation`` dataclass.
    Nothing is mutated; no global state is touched.

    Also exposed as ``BiometricEngine.compute_training_advice()`` for
    pipeline convenience.

    Parameters
    ----------
    metrics : dict
        Must contain ``"injury_risk"``, ``"fatigue_index"``, and
        ``"anomaly_score"`` as float values.  Extra keys are silently
        ignored; the dict is never mutated.

    Returns
    -------
    TrainingRecommendation
        Frozen dataclass with:

        - ``.state``            : str   — member of ``VALID_TRAINING_STATES``
        - ``.recommended_load`` : float — in [0.0, 1.0], always finite
        - ``.to_dict()``        : ``{"state": str, "recommended_load": float}``

    Invariants
    ----------
    - ``metrics`` is never mutated.
    - ``result.state`` is always in ``VALID_TRAINING_STATES``.
    - ``result.recommended_load`` is always finite and in [0.0, 1.0].
    - The returned ``TrainingRecommendation`` is frozen (immutable).
    - Deterministic: identical inputs → identical output, always.
    """
    state: str   = compute_training_state(metrics)
    load:  float = compute_recommended_load(state)
    return TrainingRecommendation(state=state, recommended_load=load)


def _safe_mean(values: tuple[float, ...]) -> float:
    """Arithmetic mean; caller guarantees len >= 1 and all finite."""
    return sum(values) / len(values)


def _safe_pstdev(values: tuple[float, ...]) -> float:
    """
    Population standard deviation.
    Returns 0.0 for a single-element series (sigma undefined -> 0 is safe).
    """
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _min_max_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    """
    Min-max normalisation -> [0.0, 1.0].
    When min == max (constant signal), returns all-zeros (no division-by-zero).
    """
    lo = min(values)
    hi = max(values)
    spread = hi - lo
    if spread == 0.0:
        return tuple(0.0 for _ in values)
    return tuple((v - lo) / spread for v in values)


def _is_finite_list(obj: Any) -> bool:
    """Return True iff obj is a list whose every element is a finite float/int."""
    if not isinstance(obj, list):
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in obj)


# =============================================================================
# ENGINE
# =============================================================================

class BiometricEngine:
    """
    Deterministic, stateless biometric signal processor.

    The engine holds NO mutable state beyond the injected ``data_engine``
    reference.  Every public method is a pure function of its arguments
    plus the frozen ``BiometricConfig``.

    Thread-safety: safe to call concurrently — no shared mutable state.

    Optional DataEngine integration
    --------------------------------
    Pass any object that implements ``run({"values": list[float]}) → dict``
    as ``data_engine``.  When provided it is called on the ``"load"`` channel
    inside ``process()`` and the returned ``"features"`` dict is surfaced as
    ``result["data_features"]``.  If the engine raises or returns an
    unexpected value, ``data_features`` silently falls back to ``{}`` so the
    biometric pipeline always succeeds.
    """

    def __init__(
        self,
        config: BiometricConfig | None = None,
        data_engine: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : BiometricConfig | None
            Engine thresholds.  Uses ``BiometricConfig()`` defaults if None.
        data_engine : object | None
            Optional time-series feature extractor.  Must implement::

                run({"values": list[float]}) -> {"features": dict, ...}

            Duck-typed — no ABC coupling with ``agent.domain``.
            Defaults to ``None``; ``data_features`` will be ``{}`` in all
            output dicts when not provided.
        """
        self._cfg         = config or BiometricConfig()
        self._data_engine = data_engine

    # -------------------------------------------------------------------------
    # PUBLIC API — DomainEngine contract
    # -------------------------------------------------------------------------

    def process(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Full pipeline:
            validate -> clean -> normalize -> compute_core_metrics
                                           -> compute_fatigue_index
                                           -> compute_drift
                                           -> compute_injury_risk
                                           -> compute_anomaly_signals
                                           -> compute_anomaly
                                           -> compute_training_state
                                           -> compute_recommended_load
                                           -> compute_training_advice
                                      [opt: data_engine.run() on load channel]

        Returns a JSON-serialisable dict conforming to the output schema.
        Never raises — all errors are captured in the returned dict.

        ``"data_features"`` is always present in the output (top-level key,
        separate from ``"metrics"``).  It is ``{}`` when no ``data_engine``
        is configured.
        """
        # Run the optional DataEngine *before* the biometric pipeline so
        # that any exception it raises is fully isolated from the core path.
        data_features: dict = self._extract_data_features(raw)

        try:
            self.validate_schema(raw)
            cleaned    = self.clean_data(raw)
            normalized = self.normalize(cleaned)
            metrics    = self.compute_core_metrics(cleaned)
            fatigue    = self.compute_fatigue_index(metrics, normalized)
            drift      = self.compute_drift(metrics, fatigue)
            risk       = self.compute_injury_risk(metrics, fatigue, drift)
            signals    = self.compute_anomaly_signals(metrics, fatigue, drift)
            anomaly    = self.compute_anomaly(signals)
            state      = self.compute_training_state(
                self._build_score_dict(fatigue, risk, anomaly)
            )
            rec_load   = self.compute_recommended_load(state)
            advice     = self.compute_training_advice(
                self._build_score_dict(fatigue, risk, anomaly)
            )
            return self._ok(
                metrics, fatigue, drift, risk, anomaly,
                state, rec_load, advice, data_features,
            )
        except SchemaError as exc:
            return self._error(str(exc), data_features)
        except Exception as exc:  # pragma: no cover — safety net
            return self._error(f"internal: {exc}", data_features)

    def validate_schema(self, raw: dict[str, Any]) -> None:
        """
        Structural validation.  Raises SchemaError on any violation.

        Checks:
            1. raw is a dict
            2. All required channels present
            3. Each channel is a list
            4. Each element is numeric (int or float)
            5. Each channel has at least min_samples items

        Does NOT modify raw.  Does NOT filter non-finite values here —
        that is clean_data()'s responsibility.
        """
        if not isinstance(raw, dict):
            raise SchemaError(
                f"Input must be a dict, got {type(raw).__name__!r}."
            )

        for ch in REQUIRED_CHANNELS:
            if ch not in raw:
                raise SchemaError(f"Missing required channel: {ch!r}.")

            channel = raw[ch]

            if not isinstance(channel, list):
                raise SchemaError(
                    f"Channel {ch!r} must be a list, "
                    f"got {type(channel).__name__!r}."
                )

            if len(channel) == 0:
                raise SchemaError(f"Channel {ch!r} is empty.")

            for idx, val in enumerate(channel):
                if not isinstance(val, (int, float)):
                    raise SchemaError(
                        f"Channel {ch!r}[{idx}] must be numeric, "
                        f"got {type(val).__name__!r}."
                    )

            if len(channel) < self._cfg.min_samples:
                raise SchemaError(
                    f"Channel {ch!r} has {len(channel)} sample(s); "
                    f"minimum required is {self._cfg.min_samples}."
                )

    def clean_data(self, raw: dict[str, Any]) -> CleanedSignals:
        """
        Remove non-finite values (NaN, +/-Inf) from every channel.

        When strict_guards=True, also removes physiologically implausible
        values based on per-channel bounds in BiometricConfig.

        Raises SchemaError if any channel drops below min_samples after
        cleaning (caller should validate_schema first).

        Returns an immutable CleanedSignals.
        """
        guards: dict[str, tuple[float, float]] = {
            "hr":   (self._cfg.hr_min,   self._cfg.hr_max),
            "hrv":  (self._cfg.hrv_min,  self._cfg.hrv_max),
            "load": (self._cfg.load_min, self._cfg.load_max),
        }

        cleaned: dict[str, tuple[float, ...]] = {}

        for ch in REQUIRED_CHANNELS:
            values: list[float] = [float(v) for v in raw[ch]]

            # Step 1 — remove non-finite values
            values = [v for v in values if math.isfinite(v)]

            # Step 2 (optional) — physiological range guard
            if self._cfg.strict_guards:
                lo, hi = guards[ch]
                values = [v for v in values if lo <= v <= hi]

            if len(values) < self._cfg.min_samples:
                raise SchemaError(
                    f"Channel {ch!r} has only {len(values)} finite sample(s) "
                    f"after cleaning; minimum required is {self._cfg.min_samples}."
                )

            cleaned[ch] = tuple(values)

        return CleanedSignals(**cleaned)

    def normalize(self, cleaned: CleanedSignals) -> NormalizedSignals:
        """
        Min-max normalise each channel independently to [0.0, 1.0].

        A channel where all values are equal (zero spread) maps to all-zeros.
        This avoids division-by-zero and is deterministic.

        Returns an immutable NormalizedSignals.
        """
        return NormalizedSignals(
            hr=_min_max_normalize(cleaned.hr),
            hrv=_min_max_normalize(cleaned.hrv),
            load=_min_max_normalize(cleaned.load),
        )

    def compute_core_metrics(self, cleaned: CleanedSignals) -> CoreMetrics:
        """
        Compute five descriptive statistics over the cleaned (raw-scale) channels.

        Metrics
        -------
        mean_hr   — arithmetic mean of HR channel (bpm)
        mean_hrv  — arithmetic mean of HRV channel (ms)
        load_mean — arithmetic mean of load channel (AU)
        hr_std    — population std-dev of HR channel
        hrv_std   — population std-dev of HRV channel

        All returned floats are finite by construction (inputs are clean).
        Values are rounded to BiometricConfig.output_precision decimal places.
        """
        p = self._cfg.output_precision
        return CoreMetrics(
            mean_hr=_round(_safe_mean(cleaned.hr),     p),
            mean_hrv=_round(_safe_mean(cleaned.hrv),   p),
            load_mean=_round(_safe_mean(cleaned.load), p),
            hr_std=_round(_safe_pstdev(cleaned.hr),    p),
            hrv_std=_round(_safe_pstdev(cleaned.hrv),  p),
        )

    def compute_fatigue_index(
        self,
        metrics: CoreMetrics,           # reserved for future phases (7.5+)
        normalized: NormalizedSignals,
    ) -> FatigueResult:
        """
        Compute the fatigue index from normalised signal statistics.

        The three fatigue inputs are all derived from NormalizedSignals so
        they share the same [0, 1] scale, making the weight interpretation
        transparent and the output inherently bounded.

        Inputs
        ------
        norm_load_mean  = mean(normalized.load)    — mean normalised load
        norm_hr_std     = pstdev(normalized.hr)    — normalised HR variability
        norm_hrv_mean   = mean(normalized.hrv)     — mean normalised HRV

        Formula
        -------
        fatigue_raw = W1 * norm_load_mean
                    + W2 * norm_hr_std
                    - W3 * norm_hrv_mean

        fatigue = sigmoid(fatigue_raw)
        fatigue = clamp(fatigue, 0.0, 1.0)

        The `metrics` argument (CoreMetrics) is accepted but not used here;
        it is present in the signature to keep the interface open for
        Phase 7.5 risk scoring which will combine raw-scale metrics with the
        fatigue score.

        Returns
        -------
        FatigueResult with:
            .value  — final fatigue score in [0, 1]
            .raw    — pre-sigmoid linear combination (diagnostic)
        """
        norm_load_mean: float = _safe_mean(normalized.load)
        norm_hr_std: float    = _safe_pstdev(normalized.hr)
        norm_hrv_mean: float  = _safe_mean(normalized.hrv)

        fatigue_raw: float = (
            FATIGUE_W1 * norm_load_mean
            + FATIGUE_W2 * norm_hr_std
            - FATIGUE_W3 * norm_hrv_mean
        )

        fatigue_sigmoid: float = _bounded_sigmoid(fatigue_raw)
        fatigue_final: float   = _clamp(fatigue_sigmoid, 0.0, 1.0)
        fatigue_rounded: float = _round(fatigue_final, self._cfg.output_precision)

        return FatigueResult(
            value=fatigue_rounded,
            raw=_round(fatigue_raw, self._cfg.output_precision),
        )

    def compute_drift(
        self,
        metrics: CoreMetrics,
        fatigue: FatigueResult,
    ) -> DriftResult:
        """
        Compute the drift score from fatigue deviation and raw signal variability.

        All three component signals are non-negative by construction.

        Components
        ----------
        fatigue_drift = abs(fatigue.value - 0.5)
            Deviation of fatigue from the neutral midpoint [0, 0.5].
            abs() makes the formula symmetric: extreme fatigue in either
            direction (very low or very high) both register as drift.

        hr_drift = abs(metrics.hr_std)
            Raw-scale HR population std-dev.  abs() is a semantic no-op
            (std >= 0) but is included per spec for explicit clarity.
            Range: [0, +inf) — raw-scale; clamped downstream.

        hrv_drift = abs(metrics.hrv_std)
            Same rationale as hr_drift.

        Formula
        -------
        drift_raw   = DW1 * fatigue_drift + DW2 * hr_drift + DW3 * hrv_drift
        drift_score = clamp(drift_raw, 0.0, 1.0)

        The clamp is essential: hr_drift and hrv_drift are raw-scale and can
        be arbitrarily large, making the weighted sum potentially >> 1.

        Returns
        -------
        DriftResult with:
            .score       — drift score in [0, 1]
            .components  — (fatigue_drift, hr_drift, hrv_drift) for diagnostics
        """
        p = self._cfg.output_precision

        fatigue_drift: float = abs(fatigue.value - 0.5)
        hr_drift: float      = abs(metrics.hr_std)
        hrv_drift: float     = abs(metrics.hrv_std)

        drift_raw: float = (
            DRIFT_W_FATIGUE * fatigue_drift
            + DRIFT_W_HR    * hr_drift
            + DRIFT_W_HRV   * hrv_drift
        )

        drift_score: float = _clamp(drift_raw, 0.0, 1.0)

        return DriftResult(
            score=_round(drift_score, p),
            components=(
                _round(fatigue_drift, p),
                _round(hr_drift,      p),
                _round(hrv_drift,     p),
            ),
        )

    def compute_injury_risk(
        self,
        metrics: CoreMetrics,
        fatigue: FatigueResult,
        drift: DriftResult,
    ) -> InjuryRiskResult:
        """
        Compute the injury risk score from fatigue, drift, and normalised load.

        All three input signals are in [0, 1] by construction, so the weighted
        sum is inherently in [0, 1] before clamping.  The final clamp is a
        belt-and-suspenders guard against float rounding at the boundaries.

        Signals
        -------
        fatigue_signal = fatigue.value
            Already in [0, 1] — direct pass-through.

        drift_signal = drift.score
            Already in [0, 1] — direct pass-through.

        load_signal = clamp(metrics.load_mean / 1000, 0.0, 1.0)
            Normalises raw-scale load_mean (AU) to [0, 1] using 1000 AU as
            the reference ceiling.  Values above 1000 AU are clamped to 1.0.
            This is the only component that requires explicit normalisation.

        Formula
        -------
        risk_raw   = RW1 * fatigue_signal
                   + RW2 * drift_signal
                   + RW3 * load_signal
        risk_score = clamp(risk_raw, 0.0, 1.0)

        Weights: RW1=0.45, RW2=0.35, RW3=0.20  (sum to 1.0)

        Returns
        -------
        InjuryRiskResult with:
            .score    — injury risk score in [0, 1]
            .signals  — (fatigue_signal, drift_signal, load_signal) for diagnostics
        """
        p = self._cfg.output_precision

        fatigue_signal: float = fatigue.value
        drift_signal: float   = drift.score
        load_signal: float    = _clamp(metrics.load_mean / 1000.0, 0.0, 1.0)

        risk_raw: float = (
            RISK_W_FATIGUE * fatigue_signal
            + RISK_W_DRIFT * drift_signal
            + RISK_W_LOAD  * load_signal
        )

        risk_score: float = _clamp(risk_raw, 0.0, 1.0)

        return InjuryRiskResult(
            score=_round(risk_score, p),
            signals=(
                _round(fatigue_signal, p),
                _round(drift_signal,   p),
                _round(load_signal,    p),
            ),
        )

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS — output construction
    # -------------------------------------------------------------------------

    def compute_anomaly_signals(
        self,
        metrics: CoreMetrics,
        fatigue: FatigueResult,
        drift: DriftResult,
    ) -> AnomalySignals:
        """
        Decompose the current pipeline state into four normalised [0, 1] signals.

        This is a pure function — it reads its arguments and returns a new
        AnomalySignals without mutating anything or touching global state.
        It is NOT wired into process() and adds no key to the output dict.

        Signal definitions
        ------------------
        fatigue_signal = fatigue.value
            Direct pass-through of the clamped sigmoid fatigue index; already
            in [0, 1] — no further transformation needed.

        drift_signal = drift.score
            Direct pass-through of the clamped drift score; already in [0, 1].

        hr_spread = clamp(metrics.hr_std / ANOMALY_HR_REF, 0.0, 1.0)
            Normalises raw HR population std-dev to [0, 1] using 50.0 bpm as
            the reference ceiling.  hr_std >= 50 bpm → hr_spread = 1.0.
            hr_std == 0  (constant channel) → hr_spread = 0.0.

        hrv_spread = clamp(metrics.hrv_std / ANOMALY_HRV_REF, 0.0, 1.0)
            Normalises raw HRV population std-dev to [0, 1] using 100.0 ms as
            the reference ceiling.  hrv_std >= 100 ms → hrv_spread = 1.0.
            hrv_std == 0 (constant channel) → hrv_spread = 0.0.

        Invariants (enforced here, not delegated to the dataclass)
        ----------------------------------------------------------
        - All four output values are finite (inputs guaranteed finite by pipeline).
        - All four output values are in [0, 1] by construction.
        - metrics, fatigue, and drift are never mutated (all are frozen dataclasses).

        Returns
        -------
        AnomalySignals with four float fields, all in [0, 1].
        """
        p = self._cfg.output_precision

        fatigue_signal: float = fatigue.value
        drift_signal: float   = drift.score
        hr_spread: float      = _clamp(metrics.hr_std  / ANOMALY_HR_REF,  0.0, 1.0)
        hrv_spread: float     = _clamp(metrics.hrv_std / ANOMALY_HRV_REF, 0.0, 1.0)

        return AnomalySignals(
            fatigue_signal=_round(fatigue_signal, p),
            drift_signal=  _round(drift_signal,   p),
            hr_spread=     _round(hr_spread,       p),
            hrv_spread=    _round(hrv_spread,      p),
        )

    def compute_anomaly(self, signals: AnomalySignals) -> AnomalyResult:
        """
        Compute the anomaly score from a fully-populated AnomalySignals snapshot.

        This is a pure function — it reads `signals` and returns a new
        AnomalyResult without mutating anything or touching global state.
        It is wired into process() immediately after compute_anomaly_signals().

        Formula
        -------
        anomaly_raw = AW_F * signals.fatigue_signal
                    + AW_D * signals.drift_signal
                    + AW_H * signals.hr_spread
                    + AW_V * signals.hrv_spread

        anomaly_score = clamp(anomaly_raw, 0.0, 1.0)

        All four input signals are already in [0, 1] and the weights sum to
        1.0, so anomaly_raw is inherently in [0, 1] before clamping.  The
        clamp is a belt-and-suspenders guard against float rounding.

        Weights
        -------
        AW_F = ANOMALY_W_FATIGUE = 0.40   dominant driver
        AW_D = ANOMALY_W_DRIFT   = 0.30   instability contribution
        AW_H = ANOMALY_W_HR      = 0.20   exertion variability
        AW_V = ANOMALY_W_HRV     = 0.10   recovery variability

        Returns
        -------
        AnomalyResult with:
            .score   — anomaly score in [0, 1]
            .signals — the AnomalySignals snapshot consumed (for diagnostics)
        """
        p = self._cfg.output_precision

        anomaly_raw: float = (
            ANOMALY_W_FATIGUE * signals.fatigue_signal
            + ANOMALY_W_DRIFT * signals.drift_signal
            + ANOMALY_W_HR    * signals.hr_spread
            + ANOMALY_W_HRV   * signals.hrv_spread
        )

        anomaly_score: float = _clamp(anomaly_raw, 0.0, 1.0)

        return AnomalyResult(
            score=_round(anomaly_score, p),
            signals=signals,
        )

    def compute_training_state(self, metrics: dict) -> str:
        """
        Classify the current training state from the three composite scores.

        Delegates to the module-level ``compute_training_state()`` function,
        making the method a pure pass-through with no added logic.  Accepts
        the same ``metrics`` dict that ``process()`` emits, so callers can
        drive it from live output without manual field extraction.

        Parameters
        ----------
        metrics : dict
            Must contain ``"injury_risk"``, ``"fatigue_index"``, and
            ``"anomaly_score"`` as float values.  Extra keys are ignored;
            the dict is never mutated.

        Returns
        -------
        str
            One of ``{"FULL", "CAUTION", "LIGHT", "RECOVERY"}``.
            Always a member of ``VALID_TRAINING_STATES``.

        Rules (priority order — first match wins)
        ------------------------------------------
        1. injury_risk   >= TS_RECOVERY_THRESHOLD (0.65) → ``"RECOVERY"``
        2. fatigue_index >= TS_LIGHT_THRESHOLD    (0.70) → ``"LIGHT"``
        3. anomaly_score >= TS_CAUTION_THRESHOLD  (0.60) → ``"CAUTION"``
        4. (default)                                     → ``"FULL"``
        """
        return compute_training_state(metrics)

    def compute_recommended_load(self, training_state: str) -> float:
        """
        Map a training state string to a normalised load multiplier in [0, 1].

        Delegates to the module-level ``compute_recommended_load()`` function,
        making the method a pure pass-through with no added logic.

        Parameters
        ----------
        training_state : str
            Must be one of ``VALID_TRAINING_STATES``.

        Returns
        -------
        float
            Always in ``[0.0, 1.0]``:
            ``"FULL"`` → 1.0, ``"CAUTION"`` → 0.75,
            ``"LIGHT"`` → 0.50, ``"RECOVERY"`` → 0.25.

        Raises
        ------
        ValueError
            If ``training_state`` is not a member of ``VALID_TRAINING_STATES``.
        """
        return compute_recommended_load(training_state)

    def compute_training_advice(self, metrics: dict) -> TrainingRecommendation:
        """
        Compute a structured ``TrainingRecommendation`` from a metrics dict.

        Delegates to the module-level ``compute_training_advice()`` function,
        making the method a pure pass-through with no added logic.

        Parameters
        ----------
        metrics : dict
            Must contain ``"injury_risk"``, ``"fatigue_index"``, and
            ``"anomaly_score"``.  Extra keys are ignored; dict never mutated.

        Returns
        -------
        TrainingRecommendation
            Frozen dataclass with ``.state`` (str) and
            ``.recommended_load`` (float in [0, 1]).
        """
        return compute_training_advice(metrics)

    # ── DataEngine integration ────────────────────────────────────────────────

    def _extract_data_features(self, raw: Any) -> dict:
        """
        Call the optional DataEngine on the ``"load"`` channel of ``raw``.

        Returns
        -------
        dict
            The ``"features"`` sub-dict from the DataEngine result, or ``{}``
            when:

            - no ``data_engine`` is configured,
            - ``raw`` is not a dict or has no ``"load"`` key,
            - the DataEngine call raises any exception, or
            - the DataEngine result is not a dict.

        Guarantees
        ----------
        - ``raw`` is never mutated.
        - All exceptions from the DataEngine are silently swallowed; a
          faulty engine can never break the biometric pipeline.
        - The returned dict is always a shallow copy so mutations by the
          caller cannot affect the DataEngine's internal state.
        """
        if self._data_engine is None:
            return {}
        try:
            if not isinstance(raw, dict) or "load" not in raw:
                return {}
            # Build a fresh one-key dict — never pass raw directly to avoid
            # the DataEngine seeing (and potentially mutating) extra channels.
            de_result = self._data_engine.run({"values": list(raw["load"])})
            if not isinstance(de_result, dict):
                return {}
            features = de_result.get("features", {})
            return dict(features) if isinstance(features, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _build_score_dict(
        fatigue: FatigueResult,
        risk: InjuryRiskResult,
        anomaly: AnomalyResult,
    ) -> dict:
        """Build the minimal dict required by compute_training_state()."""
        return {
            "fatigue_index": fatigue.value,
            "injury_risk":   risk.score,
            "anomaly_score": anomaly.score,
        }

    def _ok(
        self,
        metrics: CoreMetrics,
        fatigue: FatigueResult,
        drift: DriftResult,
        risk: InjuryRiskResult,
        anomaly: AnomalyResult,
        state: str,
        rec_load: float,
        advice: TrainingRecommendation,
        data_features: dict,
    ) -> dict[str, Any]:
        output = metrics.to_dict()
        output.update(fatigue.to_dict())          # adds "fatigue_index"
        output.update(drift.to_dict())            # adds "drift_score"
        output.update(risk.to_dict())             # adds "injury_risk"
        output.update(anomaly.to_dict())          # adds "anomaly_score"
        output["training_state"]   = state              # adds "training_state"
        output["recommended_load"] = rec_load           # adds "recommended_load"
        output["training_advice"]  = advice.to_dict()  # adds "training_advice"
        return {
            "status":        "ok",
            "metrics":       output,
            "data_features": data_features,       # {} when no data_engine
            "engine":        ENGINE_NAME,
            "version":       ENGINE_VERSION,
        }

    def _error(self, message: str, data_features: dict | None = None) -> dict[str, Any]:
        return {
            "status":        "error",
            "metrics":       None,
            "error":         message,
            "data_features": data_features if data_features is not None else {},
            "engine":        ENGINE_NAME,
            "version":       ENGINE_VERSION,
        }