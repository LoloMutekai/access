"""
A.C.C.E.S.S. — Recovery Prediction Engine (Phase 7.9)
biometric/recovery_prediction.py

Estimates the number of hours an athlete should rest before their next
optimal training session, based on five normalised biometric signals produced
by earlier stages of the BiometricEngine pipeline.

───────────────────────────────────────────────────────────────────────────────
Model overview
───────────────────────────────────────────────────────────────────────────────

Inputs (all ∈ [0, 1])
─────────────────────
    fatigue_index       — physiological fatigue burden (dominant signal)
    injury_risk         — current injury probability signal
    anomaly_score       — statistical anomaly in biometric channels
    baseline_deviation  — how far today's metrics sit from personal baseline
    recommended_load    — fraction of full training load that is advised
                          (inverted: low recommended load → high burden)

Recovery burden score
─────────────────────
    recovery_score =
          W_FATIGUE           * fatigue_index
        + W_INJURY            * injury_risk
        + W_ANOMALY           * anomaly_score
        + W_BASELINE_DEV      * baseline_deviation
        + W_LOAD              * (1 − recommended_load)

    score = clamp(recovery_score, 0.0, 1.0)

    Weights
    ───────
        W_FATIGUE        = 0.35   fatigue is the primary driver
        W_INJURY         = 0.30   injury risk is the next most critical
        W_ANOMALY        = 0.15   anomalous readings raise caution
        W_BASELINE_DEV   = 0.10   deviation from personal norm contributes
        W_LOAD           = 0.10   light recommended load signals more rest
        Sum              = 1.00 ✓

Recovery hours model
────────────────────
    recovery_hours =
        MIN_RECOVERY_HOURS
        + score * (MAX_RECOVERY_HOURS − MIN_RECOVERY_HOURS)

    MIN_RECOVERY_HOURS = 4.0    (score ≈ 0  → athlete near-recovered)
    MAX_RECOVERY_HOURS = 72.0   (score ≈ 1  → severe load, needs 3 days)

    The linear model is intentionally simple and auditable.  Non-linearity
    can be introduced in a later phase without changing the public API.

Interpretation table
────────────────────
    score 0.00–0.25  hours  4– 21  Athlete is well-recovered; light training OK
    score 0.25–0.50  hours 21– 38  Moderate fatigue; consider intensity reduction
    score 0.50–0.75  hours 38– 55  Significant fatigue; active recovery advised
    score 0.75–1.00  hours 55– 72  High burden; full rest strongly recommended

Mathematical guarantees
───────────────────────
    G1: recovery_hours ∈ [MIN_RECOVERY_HOURS, MAX_RECOVERY_HOURS] = [4, 72]
    G2: recovery_score ∈ [0.0, 1.0]
    G3: Both fields are always finite floats
    G4: Fully deterministic — no randomness, no external state
    G5: All inputs are read-only; no mutation occurs
    G6: RecoveryPrediction is JSON-serialisable via to_dict()

Integration position in the BiometricEngine pipeline
─────────────────────────────────────────────────────
    raw signals
        → metrics          (CoreMetrics)
        → fatigue_index    (FatigueResult.value)
        → drift_score      (DriftResult)
        → injury_risk      (InjuryRiskResult.score)
        → anomaly_score    (AnomalyResult.score)
        → training_state   (str)
        → recommended_load (float via RECOMMENDED_LOAD mapping)
        → baseline_deviation (float via compute_baseline_deviation())
        → recovery_prediction ← THIS MODULE                         ← NEW
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Union


# =============================================================================
# CONSTANTS
# =============================================================================

#: Minimum predicted recovery time (score = 0 → fully recovered).
MIN_RECOVERY_HOURS: float = 4.0

#: Maximum predicted recovery time (score = 1 → maximum burden).
MAX_RECOVERY_HOURS: float = 72.0

# ── Recovery burden weights ───────────────────────────────────────────────────

#: Fatigue index weight — dominant signal in the recovery model.
W_FATIGUE: float = 0.35

#: Injury risk weight — second most critical signal.
W_INJURY: float = 0.30

#: Anomaly score weight — statistical anomaly in biometric channels.
W_ANOMALY: float = 0.15

#: Baseline deviation weight — distance from personal norm.
W_BASELINE_DEV: float = 0.10

#: Inverted recommended-load weight — low load → higher recovery burden.
W_LOAD: float = 0.10

# Sanity assertion: weights must sum to exactly 1.0 (verified at import time).
_WEIGHT_SUM: float = W_FATIGUE + W_INJURY + W_ANOMALY + W_BASELINE_DEV + W_LOAD
assert abs(_WEIGHT_SUM - 1.0) < 1e-12, (
    f"Recovery burden weights must sum to 1.0, got {_WEIGHT_SUM}"
)

# Derived range — precomputed to avoid repeated arithmetic in the hot path.
_HOUR_RANGE: float = MAX_RECOVERY_HOURS - MIN_RECOVERY_HOURS  # 68.0


# =============================================================================
# RECOVERY PREDICTION DATACLASS
# =============================================================================

@dataclass(frozen=True)
class RecoveryPrediction:
    """
    Immutable snapshot of a single recovery prediction.

    Fields
    ──────
    hours : float
        Predicted recovery time in hours, always ∈ [MIN_RECOVERY_HOURS,
        MAX_RECOVERY_HOURS] = [4.0, 72.0].

    score : float
        Aggregate recovery burden ∈ [0.0, 1.0].
        0.0 → athlete is fully recovered.
        1.0 → maximum fatigue burden detected.

    The dataclass is frozen so snapshots can be safely shared, cached, or
    stored without risk of accidental mutation.

    Invariants
    ──────────
    - Both fields are always finite.
    - ``score`` ∈ [0.0, 1.0].
    - ``hours`` ∈ [MIN_RECOVERY_HOURS, MAX_RECOVERY_HOURS].

    Example
    ───────
        pred = compute_recovery_prediction(
            fatigue_index=0.6,
            injury_risk=0.4,
            anomaly_score=0.2,
            baseline_deviation=0.15,
            recommended_load=0.5,
        )
        assert 4.0 <= pred.hours <= 72.0
        assert 0.0 <= pred.score <= 1.0
        print(pred.to_dict())
        # {"recovery_hours": ..., "recovery_score": ...}
    """

    hours: float
    score: float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, float]:
        """
        Return a JSON-serialisable plain-dict representation.

        Returns
        ───────
        dict
            ``{"recovery_hours": float, "recovery_score": float}``

        The returned dict is a fresh object — mutating it does not affect
        this frozen instance.
        """
        return {
            "recovery_hours": self.hours,
            "recovery_score": self.score,
        }

    def is_valid(self) -> bool:
        """
        Return ``True`` when both fields satisfy all documented invariants.

        Checks
        ──────
        - ``hours`` finite and ∈ [MIN_RECOVERY_HOURS, MAX_RECOVERY_HOURS]
        - ``score`` finite and ∈ [0.0, 1.0]
        """
        return (
            math.isfinite(self.hours)
            and MIN_RECOVERY_HOURS <= self.hours <= MAX_RECOVERY_HOURS
            and math.isfinite(self.score)
            and 0.0 <= self.score <= 1.0
        )


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_recovery_prediction(
    fatigue_index:      float,
    injury_risk:        float,
    anomaly_score:      float,
    baseline_deviation: float,
    recommended_load:   float,
) -> RecoveryPrediction:
    """
    Compute a deterministic recovery prediction from five normalised signals.

    All inputs are expected to be finite floats in [0, 1].  Non-finite
    values are treated as 0.0 (safe fallback) so a single corrupt upstream
    signal cannot produce an invalid prediction.

    Parameters
    ──────────
    fatigue_index : float
        Physiological fatigue score ∈ [0, 1].  Produced by
        ``BiometricEngine.compute_fatigue_index()``.

    injury_risk : float
        Injury risk score ∈ [0, 1].  Produced by
        ``BiometricEngine.compute_injury_risk()``.

    anomaly_score : float
        Statistical anomaly score ∈ [0, 1].  Produced by
        ``BiometricEngine.compute_anomaly()``.

    baseline_deviation : float
        Deviation from personal baseline ∈ [0, 1].  Produced by
        ``biometric.baseline_deviation.compute_baseline_deviation()``.

    recommended_load : float
        Advised training load fraction ∈ [0, 1].  Produced by
        ``BiometricEngine.compute_recommended_load()``.
        Inverted inside this function: ``1 − recommended_load``.

    Returns
    ───────
    RecoveryPrediction
        Frozen dataclass with ``hours`` ∈ [4.0, 72.0] and
        ``score`` ∈ [0.0, 1.0].

    Examples
    ────────
        # Healthy athlete — low burden
        pred = compute_recovery_prediction(0.1, 0.05, 0.0, 0.05, 0.9)
        assert pred.hours < 20.0

        # Heavily fatigued athlete — high burden
        pred = compute_recovery_prediction(0.9, 0.8, 0.6, 0.7, 0.25)
        assert pred.hours > 50.0
    """
    # ── Guard inputs: replace any non-finite value with 0.0 ──────────────────
    fi  = _finite_or_zero(fatigue_index)
    ir  = _finite_or_zero(injury_risk)
    ans = _finite_or_zero(anomaly_score)
    bd  = _finite_or_zero(baseline_deviation)
    rl  = _finite_or_zero(recommended_load)

    # ── Compute weighted recovery burden ─────────────────────────────────────
    raw_score = (
        W_FATIGUE      * fi
        + W_INJURY     * ir
        + W_ANOMALY    * ans
        + W_BASELINE_DEV * bd
        + W_LOAD       * (1.0 - rl)
    )

    score = _clamp(raw_score, 0.0, 1.0)

    # ── Map score → hours via bounded linear model ────────────────────────────
    hours = MIN_RECOVERY_HOURS + score * _HOUR_RANGE

    # Defensive clamp on hours (guards against any future floating-point edge)
    hours = _clamp(hours, MIN_RECOVERY_HOURS, MAX_RECOVERY_HOURS)

    return RecoveryPrediction(hours=hours, score=score)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to the closed interval ``[lo, hi]``.

    Returns ``lo`` for any non-finite input (NaN, ±Inf) so a corrupt
    intermediate can never propagate out of the prediction boundary.

    Parameters
    ──────────
    value : float   Value to clamp.
    lo    : float   Lower bound (inclusive).
    hi    : float   Upper bound (inclusive).

    Returns
    ───────
    float
        Clamped value, always finite and in ``[lo, hi]``.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _finite_or_zero(value: float) -> float:
    """
    Return ``value`` if it is a finite float, otherwise return ``0.0``.

    Used to sanitise upstream signal inputs before they enter the weighted
    summation, ensuring a single corrupt sensor reading cannot produce an
    invalid or undefined recovery prediction.

    Parameters
    ──────────
    value : float   Candidate input signal.

    Returns
    ───────
    float
        ``value`` when finite, ``0.0`` otherwise.
    """
    return value if math.isfinite(value) else 0.0