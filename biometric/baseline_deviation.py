"""
A.C.C.E.S.S. — Baseline Deviation Engine (Phase 7.8B)
biometric/baseline_deviation.py

Measures how far a current biometric observation deviates from the user's
personal baseline produced by BaselineEngine (Phase 7.8A).

Design
------
The deviation score is a weighted sum of four normalised channel deviations,
clamped to [0, 1].  A score of 0 means the current reading is identical to
the stored baseline; 1 (or close to it) signals a large departure.

Channel deviations
------------------
    dev_hr      = |metrics.mean_hr   − baseline.mean_hr|   / max(baseline.mean_hr,   1)
    dev_hrv     = |metrics.mean_hrv  − baseline.mean_hrv|  / max(baseline.mean_hrv,  1)
    dev_load    = |metrics.load_mean − baseline.mean_load| / max(baseline.mean_load,  1)
    dev_fatigue = |fatigue.value     − baseline.baseline_fatigue|

Combination
-----------
    score = 0.35·dev_hr + 0.25·dev_hrv + 0.20·dev_load + 0.20·dev_fatigue
    score = clamp(score, 0.0, 1.0)

The denominator guard ``max(baseline_field, 1)`` prevents division-by-zero
when a baseline channel has not yet accumulated a meaningful magnitude (e.g.
on the very first session). ``dev_fatigue`` needs no such guard because both
sides already live in [0, 1].

Weights
-------
    WEIGHT_HR      = 0.35   — heart rate deviation (dominant)
    WEIGHT_HRV     = 0.25   — HRV deviation
    WEIGHT_LOAD    = 0.20   — load deviation
    WEIGHT_FATIGUE = 0.20   — fatigue deviation
    Sum            = 1.00   ✓

Mathematical guarantees
-----------------------
    G1: score ∈ [0.0, 1.0]  — enforced by final clamp
    G2: All intermediate floats are finite — guarded by _safe_abs_norm()
    G3: Deterministic — no randomness, no mutable state
    G4: Inputs are never mutated
    G5: Return value is JSON-serialisable (plain Python float)

Typical usage
-------------
    from biometric.baseline_deviation import compute_baseline_deviation
    from biometric.baseline_engine    import BaselineEngine
    from biometric.biometric_engine   import BiometricEngine

    be  = BiometricEngine()
    bln = BaselineEngine()
    raw = {"hr": [...], "hrv": [...], "load": [...]}

    result = be.process(raw)
    if result["status"] == "ok":
        # reconstruct typed objects from the pipeline result …
        score = compute_baseline_deviation(metrics, fatigue, baseline)
        assert 0.0 <= score <= 1.0
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from biometric.biometric_engine  import CoreMetrics, FatigueResult
from biometric.baseline_engine   import BaselineState


# =============================================================================
# CONSTANTS
# =============================================================================

#: Weight applied to the HR channel deviation.
WEIGHT_HR: float = 0.35

#: Weight applied to the HRV channel deviation.
WEIGHT_HRV: float = 0.25

#: Weight applied to the load channel deviation.
WEIGHT_LOAD: float = 0.20

#: Weight applied to the fatigue deviation.
WEIGHT_FATIGUE: float = 0.20

# Sanity assertion — weights must sum to 1.0 exactly (checked at import time).
_WEIGHT_SUM: float = WEIGHT_HR + WEIGHT_HRV + WEIGHT_LOAD + WEIGHT_FATIGUE
assert abs(_WEIGHT_SUM - 1.0) < 1e-12, (
    f"Deviation weights must sum to 1.0, got {_WEIGHT_SUM}"
)


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_baseline_deviation(
    metrics:  CoreMetrics,
    fatigue:  FatigueResult,
    baseline: BaselineState,
) -> float:
    """
    Compute a scalar deviation score ∈ [0, 1] measuring how far the current
    biometric observation sits from the user's personal baseline.

    Parameters
    ----------
    metrics : CoreMetrics
        Core biometric statistics from the current session / window.
        Never mutated.
    fatigue : FatigueResult
        Fatigue result from the current session / window.
        ``fatigue.value`` is used; ``fatigue.raw`` is ignored.
        Never mutated.
    baseline : BaselineState
        The user's current personal baseline from ``BaselineEngine``.
        Never mutated.

    Returns
    -------
    float
        Deviation score ∈ [0.0, 1.0].  A score of 0.0 means the current
        observation matches the baseline exactly.  Scores close to 1.0
        indicate a large departure across one or more channels.

    Notes
    -----
    The denominator guard ``max(baseline_field, 1)`` prevents division-by-zero
    for cold baselines where a channel value may be near zero.

    ``dev_fatigue`` requires no denominator guard because both
    ``fatigue.value`` and ``baseline.baseline_fatigue`` are guaranteed to
    lie in [0, 1], so their absolute difference is already in [0, 1].

    All arithmetic is deterministic, finite, and does not mutate any argument.
    """
    dev_hr      = _safe_abs_norm(metrics.mean_hr,   baseline.mean_hr)
    dev_hrv     = _safe_abs_norm(metrics.mean_hrv,  baseline.mean_hrv)
    dev_load    = _safe_abs_norm(metrics.load_mean, baseline.mean_load)
    dev_fatigue = _safe_abs_diff(fatigue.value,     baseline.baseline_fatigue)

    raw_score = (
        WEIGHT_HR      * dev_hr
        + WEIGHT_HRV     * dev_hrv
        + WEIGHT_LOAD    * dev_load
        + WEIGHT_FATIGUE * dev_fatigue
    )

    return _clamp(raw_score, 0.0, 1.0)


# =============================================================================
# NAMED DEVIATION BREAKDOWN  (optional — useful for debugging / display)
# =============================================================================

def deviation_components(
    metrics:  CoreMetrics,
    fatigue:  FatigueResult,
    baseline: BaselineState,
) -> dict[str, float]:
    """
    Return a dict of all four individual (pre-weight) deviations alongside the
    final combined score.

    This is a *diagnostic* helper — the canonical value for downstream logic
    is always ``compute_baseline_deviation()``.

    Returns
    -------
    dict
        Keys: ``"dev_hr"``, ``"dev_hrv"``, ``"dev_load"``, ``"dev_fatigue"``,
        ``"score"``.  All values are finite floats; the dict is
        JSON-serialisable.
    """
    dev_hr      = _safe_abs_norm(metrics.mean_hr,   baseline.mean_hr)
    dev_hrv     = _safe_abs_norm(metrics.mean_hrv,  baseline.mean_hrv)
    dev_load    = _safe_abs_norm(metrics.load_mean, baseline.mean_load)
    dev_fatigue = _safe_abs_diff(fatigue.value,     baseline.baseline_fatigue)

    raw_score = (
        WEIGHT_HR      * dev_hr
        + WEIGHT_HRV     * dev_hrv
        + WEIGHT_LOAD    * dev_load
        + WEIGHT_FATIGUE * dev_fatigue
    )

    return {
        "dev_hr":      dev_hr,
        "dev_hrv":     dev_hrv,
        "dev_load":    dev_load,
        "dev_fatigue": dev_fatigue,
        "score":       _clamp(raw_score, 0.0, 1.0),
    }


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _safe_abs_norm(current: float, baseline_val: float) -> float:
    """
    Compute ``|current − baseline_val| / max(baseline_val, 1)``.

    Both inputs must be finite; non-finite values return 0.0 (safe fallback).
    The denominator guard ``max(baseline_val, 1)`` prevents division-by-zero
    for small baseline values while preserving proportional scaling for
    physiologically meaningful magnitudes (HR ≥ 20 bpm, HRV ≥ 1 ms, etc.).
    """
    if not (math.isfinite(current) and math.isfinite(baseline_val)):
        return 0.0
    denom = max(baseline_val, 1.0)
    return abs(current - baseline_val) / denom


def _safe_abs_diff(a: float, b: float) -> float:
    """
    Compute ``|a − b|`` with a non-finite guard (returns 0.0 on bad input).

    Used for the fatigue channel where both sides are in [0, 1] so no
    denominator scaling is needed.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0
    return abs(a - b)


def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to ``[lo, hi]``.

    Returns ``lo`` for non-finite input so a corrupt intermediate never
    leaks out of the function boundary.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))