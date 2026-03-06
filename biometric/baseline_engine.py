"""
A.C.C.E.S.S. — BaselineEngine (Phase 7.8A — PERSONAL BASELINE)
biometric/baseline_engine.py

Deterministic personal-baseline tracker for BiometricEngine output.

Purpose
-------
Learns the user's *individual* physiological baseline over time using
Exponential Moving Average (EMA) smoothing.  Each call to ``update()``
consumes one (CoreMetrics, FatigueResult) observation and returns a fresh
``BaselineState`` snapshot that can be persisted, compared, or used
downstream to personalise training recommendations.

Architecture
------------
    BaselineEngine
        └── update(CoreMetrics, FatigueResult) → BaselineState
                │
                ├── cold start (sample_count == 0)
                │       baseline = observation (no distortion)
                └── warm path  (sample_count >= 1)
                        baseline_new = (1-α)*baseline_old + α*observation

Mathematical guarantees
-----------------------
    M1: EMA_ALPHA ∈ [0.01, 0.05]  (constant, never mutated at runtime)
    M2: All BaselineState floats are always finite — guaranteed by clamp
        guards applied before the EMA step.
    M3: sample_count is strictly monotone-increasing (by +1 per valid call).
    M4: Deterministic — identical call sequences produce identical states.
    M5: input CoreMetrics / FatigueResult are never mutated.

Output schema
-------------
    BaselineState.to_dict() →
    {
        "mean_hr":          float,   # EMA-smoothed HR mean (bpm)
        "mean_hrv":         float,   # EMA-smoothed HRV mean (ms)
        "mean_load":        float,   # EMA-smoothed load mean (AU)
        "baseline_fatigue": float,   # EMA-smoothed fatigue index ∈ [0, 1]
        "sample_count":     int,     # number of observations processed
    }

State rollback
--------------
    engine.get_state() → dict   (JSON-serialisable snapshot)
    engine.set_state(dict)      (restore from snapshot)

Usage
-----
    from biometric.baseline_engine import BaselineEngine
    from biometric.biometric_engine import BiometricEngine

    be  = BiometricEngine()
    bln = BaselineEngine()

    result = be.process({"hr": [...], "hrv": [...], "load": [...]})
    if result["status"] == "ok":
        # (reconstruct typed objects from the result dict for the update call)
        pass  # see test suite for full wiring example

Backward compatibility
----------------------
    Phase 7.8A adds this module.  biometric_engine.py is untouched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from biometric.biometric_engine import CoreMetrics, FatigueResult


# =============================================================================
# CONSTANTS
# =============================================================================

ENGINE_NAME:    str   = "BaselineEngine"
ENGINE_VERSION: str   = "7.8A.0"

#: EMA smoothing factor.  Kept deliberately small so the baseline tracks
#: genuine long-term trends without over-reacting to single sessions.
#: Constraint: EMA_ALPHA ∈ [0.01, 0.05].
EMA_ALPHA: float = 0.05

#: Physiological sanity clamps applied to observations before the EMA step.
#: These prevent a single corrupt reading from permanently skewing the baseline.
_HR_MIN:    float = 20.0
_HR_MAX:    float = 300.0
_HRV_MIN:   float = 0.0
_HRV_MAX:   float = 3_000.0
_LOAD_MIN:  float = 0.0
_LOAD_MAX:  float = 10_000.0
_FATIGUE_MIN: float = 0.0
_FATIGUE_MAX: float = 1.0


# =============================================================================
# BASELINE STATE
# =============================================================================

@dataclass(frozen=True)
class BaselineState:
    """
    Immutable snapshot of the user's personal physiological baseline.

    All float fields are guaranteed finite by ``BaselineEngine.update()``.
    The dataclass is frozen so it can be safely shared across callers without
    the risk of accidental mutation.

    Fields
    ------
    mean_hr          : float — EMA-smoothed HR mean (bpm)
    mean_hrv         : float — EMA-smoothed HRV mean (ms)
    mean_load        : float — EMA-smoothed load mean (AU)
    baseline_fatigue : float — EMA-smoothed fatigue index ∈ [0, 1]
    sample_count     : int   — number of valid observations processed so far

    Invariants
    ----------
    - All floats are finite.
    - ``sample_count >= 1`` on any state returned by ``update()``.
    - ``baseline_fatigue`` ∈ [0.0, 1.0].
    """

    mean_hr:          float
    mean_hrv:         float
    mean_load:        float
    baseline_fatigue: float
    sample_count:     int

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable plain-dict representation.

        Returns
        -------
        dict
            Keys: ``"mean_hr"``, ``"mean_hrv"``, ``"mean_load"``,
            ``"baseline_fatigue"``, ``"sample_count"``.

        The returned dict is a fresh object — mutating it does not affect
        this frozen instance.
        """
        return {
            "mean_hr":          self.mean_hr,
            "mean_hrv":         self.mean_hrv,
            "mean_load":        self.mean_load,
            "baseline_fatigue": self.baseline_fatigue,
            "sample_count":     self.sample_count,
        }

    @staticmethod
    def from_dict(d: dict) -> "BaselineState":
        """
        Reconstruct a ``BaselineState`` from a ``to_dict()`` snapshot.

        Parameters
        ----------
        d : dict
            Must contain all five keys produced by ``to_dict()``.

        Returns
        -------
        BaselineState
        """
        return BaselineState(
            mean_hr=          float(d["mean_hr"]),
            mean_hrv=         float(d["mean_hrv"]),
            mean_load=        float(d["mean_load"]),
            baseline_fatigue= float(d["baseline_fatigue"]),
            sample_count=     int(d["sample_count"]),
        )

    def is_valid(self) -> bool:
        """
        Return ``True`` when all float fields are finite and
        ``baseline_fatigue`` is in [0, 1].
        """
        return (
            math.isfinite(self.mean_hr)
            and math.isfinite(self.mean_hrv)
            and math.isfinite(self.mean_load)
            and math.isfinite(self.baseline_fatigue)
            and 0.0 <= self.baseline_fatigue <= 1.0
            and self.sample_count >= 0
        )


# =============================================================================
# BASELINE ENGINE
# =============================================================================

class BaselineEngine:
    """
    Deterministic personal-baseline tracker.

    Accepts a sequence of (``CoreMetrics``, ``FatigueResult``) observations
    and maintains a running EMA-smoothed baseline.  Every call to
    ``update()`` returns an immutable ``BaselineState`` snapshot.

    The engine holds exactly one piece of mutable state: the current
    ``BaselineState`` (plus a cold-start sentinel flag handled via
    ``_state``).  No I/O, no randomness, no global state.

    Thread-safety
    -------------
    Not thread-safe — ``update()`` mutates ``_state`` and ``_sample_count``.
    Use a separate instance per thread or protect with a lock.

    Example
    -------
        engine = BaselineEngine()
        for metrics, fatigue in session_observations:
            state = engine.update(metrics, fatigue)
        print(state.to_dict())
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        """
        Parameters
        ----------
        alpha : float
            EMA smoothing factor.  Must be in ``[0.01, 0.05]``.
            Defaults to ``EMA_ALPHA = 0.05``.

        Raises
        ------
        ValueError
            If ``alpha`` is not in ``[0.01, 0.05]``.
        """
        if not (0.01 <= alpha <= 0.05):
            raise ValueError(
                f"alpha must be in [0.01, 0.05], got {alpha!r}."
            )
        self._alpha: float = alpha
        # Current baseline; None means cold-start (no observations yet).
        self._state: Optional[BaselineState] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Read-only EMA smoothing factor."""
        return self._alpha

    @property
    def sample_count(self) -> int:
        """Number of valid observations processed so far."""
        return self._state.sample_count if self._state is not None else 0

    @property
    def current_state(self) -> Optional[BaselineState]:
        """
        The most recently returned ``BaselineState``, or ``None`` if
        ``update()`` has not been called yet.
        """
        return self._state

    # ── Core API ──────────────────────────────────────────────────────────────

    def update(
        self,
        metrics: CoreMetrics,
        fatigue: FatigueResult,
    ) -> BaselineState:
        """
        Ingest one observation and return an updated ``BaselineState``.

        Algorithm
        ---------
        Cold start (first call):
            ``baseline = observation``   (pure initialisation, no blending)

        Warm path (subsequent calls):
            ``baseline_new = (1 - α) * baseline_old + α * observation``

        Clamp guards are applied to each raw observation *before* the EMA
        step to prevent a single corrupt value from corrupting the baseline.

        Parameters
        ----------
        metrics : CoreMetrics
            Core biometric statistics from ``BiometricEngine``.  Never mutated.
        fatigue : FatigueResult
            Fatigue score from ``BiometricEngine``.  Never mutated.

        Returns
        -------
        BaselineState
            Fresh frozen snapshot.  ``sample_count`` is one greater than the
            previous state's count.

        Invariants
        ----------
        - ``metrics`` and ``fatigue`` are never mutated.
        - The returned state always passes ``BaselineState.is_valid()``.
        - ``sample_count`` is exactly ``previous_count + 1``.
        """
        # ── Clamp-guard the raw observation values ─────────────────────────
        obs_hr      = _clamp(metrics.mean_hr,    _HR_MIN,   _HR_MAX)
        obs_hrv     = _clamp(metrics.mean_hrv,   _HRV_MIN,  _HRV_MAX)
        obs_load    = _clamp(metrics.load_mean,  _LOAD_MIN, _LOAD_MAX)
        obs_fatigue = _clamp(fatigue.value,      _FATIGUE_MIN, _FATIGUE_MAX)

        if self._state is None:
            # ── Cold start: first observation becomes the baseline ──────────
            new_state = BaselineState(
                mean_hr=          obs_hr,
                mean_hrv=         obs_hrv,
                mean_load=        obs_load,
                baseline_fatigue= obs_fatigue,
                sample_count=     1,
            )
        else:
            # ── Warm path: EMA blend ────────────────────────────────────────
            α  = self._alpha
            β  = 1.0 - α
            prev = self._state
            new_state = BaselineState(
                mean_hr=          _ema(β, prev.mean_hr,          obs_hr),
                mean_hrv=         _ema(β, prev.mean_hrv,         obs_hrv),
                mean_load=        _ema(β, prev.mean_load,        obs_load),
                baseline_fatigue= _clamp(
                    _ema(β, prev.baseline_fatigue, obs_fatigue),
                    _FATIGUE_MIN, _FATIGUE_MAX,
                ),
                sample_count=     prev.sample_count + 1,
            )

        self._state = new_state
        return new_state

    def reset(self) -> None:
        """
        Reset the engine to its cold-start state (``sample_count == 0``).

        Useful for unit tests and rollback scenarios.
        """
        self._state = None

    # ── State rollback (mirrors DataEngine protocol) ──────────────────────────

    def get_state(self) -> dict:
        """
        Return a JSON-serialisable snapshot of all mutable internal state.

        The snapshot can be passed to ``set_state()`` to restore the engine
        to an earlier point in time.

        Returns
        -------
        dict
            ``{"alpha": float, "baseline": dict | None}``
        """
        return {
            "alpha":    self._alpha,
            "baseline": self._state.to_dict() if self._state is not None else None,
        }

    def set_state(self, state: dict) -> None:
        """
        Restore mutable state from a previously captured ``get_state()``
        snapshot.

        Parameters
        ----------
        state : dict
            Must contain ``"alpha"`` (float) and ``"baseline"``
            (``to_dict()`` output or ``None``).
        """
        alpha = float(state.get("alpha", EMA_ALPHA))
        if not (0.01 <= alpha <= 0.05):
            raise ValueError(
                f"alpha in snapshot must be in [0.01, 0.05], got {alpha!r}."
            )
        self._alpha = alpha
        baseline = state.get("baseline")
        self._state = BaselineState.from_dict(baseline) if baseline is not None else None

    # ── Self-diagnostics ──────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run five named internal invariant checks on a *fresh* engine instance
        so the live baseline state is never polluted.

        Checks
        ------
        1. ``cold_start_equals_observation``
               First update → baseline == observation (no EMA distortion).
        2. ``ema_smoothing_correct``
               Second update blends via EMA formula exactly.
        3. ``sample_count_increments``
               sample_count grows by 1 per call for 10 successive updates.
        4. ``all_floats_finite``
               All BaselineState floats are finite after 10 updates.
        5. ``json_serialisable``
               ``to_dict()`` output round-trips through ``json.dumps`` /
               ``json.loads`` without loss.

        Returns
        -------
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        checks: list[dict] = []

        # 1 — cold start
        def _cold_start() -> bool:
            probe = BaselineEngine(alpha=self._alpha)
            m, f  = _canonical_pair()
            state = probe.update(m, f)
            return (
                state.mean_hr          == _clamp(m.mean_hr,   _HR_MIN,   _HR_MAX)
                and state.mean_hrv     == _clamp(m.mean_hrv,  _HRV_MIN,  _HRV_MAX)
                and state.mean_load    == _clamp(m.load_mean, _LOAD_MIN, _LOAD_MAX)
                and state.baseline_fatigue == _clamp(f.value, _FATIGUE_MIN, _FATIGUE_MAX)
                and state.sample_count == 1
            )
        checks.append(_run_check("cold_start_equals_observation", _cold_start))

        # 2 — EMA formula exactness
        def _ema_correct() -> bool:
            α = self._alpha
            probe = BaselineEngine(alpha=α)
            m1, f1 = _canonical_pair(hr=70.0, hrv=40.0, load=100.0, fatigue=0.3)
            m2, f2 = _canonical_pair(hr=80.0, hrv=50.0, load=200.0, fatigue=0.5)
            probe.update(m1, f1)
            s2 = probe.update(m2, f2)
            expected_hr = (1 - α) * 70.0 + α * 80.0
            return abs(s2.mean_hr - expected_hr) < 1e-9
        checks.append(_run_check("ema_smoothing_correct", _ema_correct))

        # 3 — sample_count monotone
        def _count_increments() -> bool:
            probe = BaselineEngine(alpha=self._alpha)
            m, f  = _canonical_pair()
            for i in range(1, 11):
                s = probe.update(m, f)
                if s.sample_count != i:
                    return False
            return True
        checks.append(_run_check("sample_count_increments", _count_increments))

        # 4 — all floats finite
        def _floats_finite() -> bool:
            probe = BaselineEngine(alpha=self._alpha)
            m, f  = _canonical_pair()
            for _ in range(10):
                s = probe.update(m, f)
            return s.is_valid()
        checks.append(_run_check("all_floats_finite", _floats_finite))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            probe = BaselineEngine(alpha=self._alpha)
            m, f  = _canonical_pair()
            probe.update(m, f)
            s = probe.update(m, f)
            raw   = s.to_dict()
            back  = json.loads(json.dumps(raw))
            restored = BaselineState.from_dict(back)
            return (
                abs(restored.mean_hr          - s.mean_hr)          < 1e-9
                and abs(restored.mean_hrv     - s.mean_hrv)         < 1e-9
                and abs(restored.mean_load    - s.mean_load)        < 1e-9
                and abs(restored.baseline_fatigue - s.baseline_fatigue) < 1e-9
                and restored.sample_count     == s.sample_count
            )
        checks.append(_run_check("json_serialisable", _json_rt))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  ENGINE_NAME,
            "version": ENGINE_VERSION,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run the canonical observation sequence twice on independent fresh
        engine instances and verify the results are bit-for-bit identical.

        Canonical sequence: five updates with the same fixed
        (CoreMetrics, FatigueResult) pair.

        Returns
        -------
        bool
            ``True`` if both sequences produce identical ``BaselineState``
            snapshots for every step.
        """
        def _run_sequence() -> list[BaselineState]:
            probe = BaselineEngine(alpha=self._alpha)
            m, f  = _canonical_pair()
            return [probe.update(m, f) for _ in range(5)]

        seq1 = _run_sequence()
        seq2 = _run_sequence()
        return all(
            s1.to_dict() == s2.to_dict()
            for s1, s2 in zip(seq1, seq2)
        )


# =============================================================================
# MODULE-LEVEL HELPERS  (not exported as public API)
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to ``[lo, hi]``.  Returns ``lo`` for non-finite input."""
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _ema(beta: float, old: float, new: float) -> float:
    """
    One step of EMA: ``beta * old + (1 - beta) * new``.

    ``beta = 1 - alpha``; kept as a pre-computed argument to avoid
    re-subtraction inside the tight update loop.
    """
    return beta * old + (1.0 - beta) * new


def _canonical_pair(
    hr:      float = 70.0,
    hrv:     float = 40.0,
    load:    float = 100.0,
    fatigue: float = 0.3,
) -> tuple[CoreMetrics, FatigueResult]:
    """
    Build a fixed (CoreMetrics, FatigueResult) pair for internal tests.

    Used by ``self_test()`` and ``deterministic_check()`` so the same
    canonical values are consumed each time.
    """
    metrics = CoreMetrics(
        mean_hr=hr,
        mean_hrv=hrv,
        load_mean=load,
        hr_std=5.0,
        hrv_std=3.0,
    )
    fat = FatigueResult(value=fatigue, raw=fatigue * 2.0)
    return metrics, fat


def _run_check(name: str, fn) -> dict:
    """Execute a named check function and catch all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}