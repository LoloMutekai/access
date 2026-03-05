"""
A.C.C.E.S.S. — DataEngine / Data Backbone Layer (Phase 7.1)
agent/domain/data_engine.py

Concrete DomainEngine implementation for deterministic data processing.
No machine learning. No randomness. No external libraries beyond stdlib.

Pipeline (inside run())
───────────────────────
    1. validate_schema(input_data)     → list[str]  (errors)
    2. clean_data(values)              → list[float] (finite only)
    3. normalize(values)               → list[float] (z-score)
    4. extract_features(values, norm)  → dict
    5. compute_signals(features, norm) → dict
    6. detect_drift(features)          → float  (updates internal baseline)
    7. build_output(...)               → dict   (final JSON-safe payload)

Mathematical guarantees
───────────────────────
    P1: run(x) == run(x) excluding baseline state mutation
        (deterministic_check() validates this by comparing output
         with a fixed canonical input on back-to-back calls, excluding
         the drift_score field which reflects accumulated baseline)
    P2: input_data is never mutated
    P3: all floats in the output are finite
    P4: all outputs are JSON-serializable

Baseline state
──────────────
    The engine holds a running baseline (mean + std) for drift detection.
    It is updated via a slow EMA (alpha=0.05) on each successful run.
    This is the ONLY intentional mutation inside run().

    For rollback compatibility the baseline is exposed via get_state() /
    set_state() using the same protocol as DummyDomainEngine.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from .base import DomainEngine


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ROLLING_WINDOW   = 3
_EMA_ALPHA        = 0.2       # feature EMA
_DRIFT_ALPHA      = 0.05      # baseline update speed (slow EMA)
_DRIFT_MAX_NORM   = 5.0       # denominator clamp for normalised drift
_ANOMALY_SIGMA    = 3.0       # z-score at which anomaly_score clips to 1.0


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL BENCHMARK DATASETS
# ─────────────────────────────────────────────────────────────────────────────

_BENCHMARK_SERIES: tuple[tuple[str, list[float]], ...] = (
    # (name, values)
    ("increasing",   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    ("flat",         [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    ("oscillating",  [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0]),
    ("noisy_det",    [1.0, 1.5, 2.1, 1.9, 2.5, 3.0, 2.8]),
    ("decreasing",   [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
)


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DataEngine(DomainEngine):
    """
    Deterministic data backbone engine.

    Accepts a dict with a "values" key containing a list of floats and
    returns a richly annotated processing result — features, signals, drift
    score, and confidence — without performing any I/O or ML.

    All computations are pure arithmetic.  The only mutable state is the
    internal drift baseline, which is updated slowly via EMA after each
    successful run.

    Example usage
    ─────────────
        engine = DataEngine()
        result = engine.run({"values": [1, 2, 3, 4, 5, 6, 7]})
        assert result["status"] == "ok"
        assert 0.0 <= result["drift_score"] <= 1.0
    """

    _NAME    = "data_backbone_engine"
    _VERSION = "0.1.0"
    _REQUIRED_PERMISSIONS: frozenset = frozenset({"filesystem_read"})

    def __init__(self) -> None:
        # Drift baseline — initialised to sentinel None (cold start)
        self._baseline_mean: Optional[float] = None
        self._baseline_std:  Optional[float] = None
        # Monotonic run counter for rollback support
        self.run_count: int = 0

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def required_permissions(self) -> frozenset:
        return self._REQUIRED_PERMISSIONS

    # ── Core contract ─────────────────────────────────────────────────────────

    def run(
        self,
        input_data: dict,
        sandbox_context: Optional[Any] = None,
    ) -> dict:
        """
        Execute the data processing pipeline on input_data["values"].

        Never mutates input_data.
        Updates internal baseline state on each successful run.

        Returns
        ───────
        {
            "status":     "ok" | "error",
            "features":   {...},
            "signals":    {...},
            "drift_score": float,   # ∈ [0.0, 1.0]
            "confidence":  float,   # ∈ [0.0, 1.0]
            "engine":     self.name,
            "version":    self.version,
            "errors":     [...],    # present only on "error"
        }
        """
        # ── Step 1: schema validation ─────────────────────────────────────────
        errors = self.validate_schema(input_data)
        if errors:
            return self._error_output(errors)

        # Work on a copy — never touch input_data after this point
        raw_values: list[float] = [float(v) for v in input_data["values"]]

        # ── Step 2: clean ─────────────────────────────────────────────────────
        clean = self.clean_data(raw_values)
        if len(clean) < 5:
            return self._error_output(
                ["After cleaning, fewer than 5 finite values remain."]
            )

        # ── Step 3: normalize ─────────────────────────────────────────────────
        normalized = self.normalize(clean)

        # ── Step 4: extract features ──────────────────────────────────────────
        features = self.extract_features(clean, normalized)

        # ── Step 5: compute signals ───────────────────────────────────────────
        signals = self.compute_signals(features, normalized)

        # ── Step 6: detect drift (may update baseline) ────────────────────────
        drift_score = self.detect_drift(features)

        # ── Step 7: confidence ────────────────────────────────────────────────
        confidence = _clamp(
            signals["stability_score"] * (1.0 - drift_score), 0.0, 1.0
        )

        # ── Step 8: state bookkeeping ─────────────────────────────────────────
        self.run_count += 1

        # ── Sandbox log ───────────────────────────────────────────────────────
        if sandbox_context is not None:
            try:
                sandbox_context.log_action(
                    "filesystem_read",
                    detail="DataEngine.run — no actual I/O",
                )
            except Exception:
                pass

        output = {
            "status":      "ok",
            "features":    features,
            "signals":     signals,
            "drift_score": _safe_round(drift_score),
            "confidence":  _safe_round(confidence),
            "engine":      self.name,
            "version":     self.version,
        }
        self._assert_json_serializable(output, "run output")
        return output

    # ── Schema validation ─────────────────────────────────────────────────────

    def validate_schema(self, data: Any) -> list[str]:
        """
        Return a list of error messages; empty list means valid.

        Rules
        ─────
        - data must be a dict
        - must contain key "values"
        - "values" must be a list
        - list length >= 5
        - every element must be a real number (int or float, not None, not str)
        - no nested structures (lists / dicts inside the list)
        - all elements must be finite (no NaN, no Inf)
        """
        errors: list[str] = []

        if not isinstance(data, dict):
            return [f"input_data must be dict, got {type(data).__name__}"]

        if "values" not in data:
            errors.append("Missing required key 'values'.")
            return errors

        values = data["values"]
        if not isinstance(values, list):
            errors.append(
                f"'values' must be a list, got {type(values).__name__}."
            )
            return errors

        if len(values) < 5:
            errors.append(
                f"'values' must contain at least 5 elements, got {len(values)}."
            )

        for i, v in enumerate(values):
            if v is None:
                errors.append(f"values[{i}] is None.")
            elif isinstance(v, (list, dict)):
                errors.append(
                    f"values[{i}] is a nested structure ({type(v).__name__}); "
                    f"only scalars are allowed."
                )
            elif not isinstance(v, (int, float)):
                errors.append(
                    f"values[{i}] is {type(v).__name__!r}, expected int or float."
                )
            elif not math.isfinite(float(v)):
                errors.append(
                    f"values[{i}] = {v} is not finite (NaN or Inf)."
                )

        return errors

    # ── Deterministic transformations ─────────────────────────────────────────

    @staticmethod
    def clean_data(values: list[float]) -> list[float]:
        """
        Remove any NaN or Inf values and return the remaining elements in
        their original order.  The input list is never modified.
        """
        return [v for v in values if math.isfinite(v)]

    @staticmethod
    def normalize(values: list[float]) -> list[float]:
        """
        Apply population z-score normalization.

        z_i = (v_i - mean) / std

        Zero-std guard: if std == 0.0 (constant series), every element maps
        to 0.0 which is both correct and finite.
        """
        if not values:
            return []
        n    = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std  = math.sqrt(variance)
        if std == 0.0:
            return [0.0] * n
        return [(v - mean) / std for v in values]

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_features(
        clean: list[float],
        normalized: list[float],
    ) -> dict:
        """
        Compute a deterministic feature dict from the cleaned and normalised
        value series.

        Features produced
        ─────────────────
        mean, std             — population statistics of the raw clean series
        rolling_mean (w=3)    — centred-left rolling mean of normalised series
        rolling_std  (w=3)    — centred-left rolling std  of normalised series
        ema          (α=0.2)  — exponential moving average of normalised series
        volatility            — population std of log-returns of the clean series
        max, min              — of the raw clean series
        count                 — number of values
        """
        n    = len(clean)
        mean = sum(clean) / n
        variance = sum((v - mean) ** 2 for v in clean) / n
        std  = math.sqrt(variance)

        # Rolling statistics on the normalised series
        rolling_mean: list[float] = []
        rolling_std:  list[float] = []
        for i in range(n):
            window = normalized[max(0, i - _ROLLING_WINDOW + 1): i + 1]
            wm = sum(window) / len(window)
            rolling_mean.append(wm)
            if len(window) >= 2:
                wvar = sum((v - wm) ** 2 for v in window) / len(window)
                rolling_std.append(math.sqrt(wvar))
            else:
                rolling_std.append(0.0)

        # EMA of normalised series
        ema: list[float] = [normalized[0]]
        for v in normalized[1:]:
            ema.append(_EMA_ALPHA * v + (1.0 - _EMA_ALPHA) * ema[-1])

        # Volatility: population std of returns on the *clean* series
        #   return_i = (v_i - v_{i-1}) / |v_{i-1}|   (skips zero denominators)
        returns: list[float] = []
        for i in range(1, n):
            if clean[i - 1] != 0.0:
                returns.append((clean[i] - clean[i - 1]) / abs(clean[i - 1]))

        if returns:
            ret_mean   = sum(returns) / len(returns)
            ret_var    = sum((r - ret_mean) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(ret_var)
        else:
            volatility = 0.0

        return {
            "mean":         _safe_round(mean),
            "std":          _safe_round(std),
            "rolling_mean": [_safe_round(v) for v in rolling_mean],
            "rolling_std":  [_safe_round(v) for v in rolling_std],
            "ema":          [_safe_round(v) for v in ema],
            "volatility":   _safe_round(volatility),
            "max":          _safe_round(max(clean)),
            "min":          _safe_round(min(clean)),
            "count":        n,
        }

    # ── Signal computation ────────────────────────────────────────────────────

    @staticmethod
    def compute_signals(features: dict, normalized: list[float]) -> dict:
        """
        Derive bounded signals from pre-computed features.

        Signals
        ───────
        anomaly_score   ∈ [0, 1]
            |z| / ANOMALY_SIGMA, clipped.  Measures how many standard
            deviations the last observation sits from the series mean.

        momentum_score  ∈ [-1, 1]
            (last_z) / ANOMALY_SIGMA, clipped.  Positive = series trending
            above average at the tail; negative = below average.

        stability_score ∈ (0, 1]
            1 / (1 + volatility).  Higher volatility → lower stability.
        """
        last_z = normalized[-1] if normalized else 0.0
        volatility: float = features["volatility"]

        anomaly_score   = _clamp(abs(last_z) / _ANOMALY_SIGMA,  0.0, 1.0)
        momentum_score  = _clamp(last_z      / _ANOMALY_SIGMA, -1.0, 1.0)
        stability_score = 1.0 / (1.0 + volatility)   # volatility >= 0 → in (0,1]

        return {
            "anomaly_score":   _safe_round(anomaly_score),
            "momentum_score":  _safe_round(momentum_score),
            "stability_score": _safe_round(stability_score),
        }

    # ── Drift detection ───────────────────────────────────────────────────────

    def detect_drift(self, features: dict) -> float:
        """
        Compare current statistics against the stored baseline and return a
        drift score ∈ [0.0, 1.0].

        Cold start (no baseline yet): drift_score = 0.0; baseline is
        initialised to the current statistics.

        Baseline update: slow EMA (alpha = DRIFT_ALPHA = 0.05) so the
        baseline tracks gradual shifts without being sensitive to transients.

        Drift score formula
        ───────────────────
            mean_shift = |μ_curr − μ_base| / (|μ_base| + ε)
            std_shift  = |σ_curr − σ_base| / (σ_base + ε)
            raw        = (mean_shift + std_shift) / 2
            score      = clamp(raw / DRIFT_MAX_NORM, 0, 1)

        Using ε = 1e-9 to guard against exact-zero denominators.
        """
        curr_mean: float = features["mean"]
        curr_std:  float = features["std"]

        if self._baseline_mean is None:
            # Cold start — no drift on the first observation
            self._baseline_mean = curr_mean
            self._baseline_std  = max(curr_std, 0.0)
            return 0.0

        eps = 1e-9
        mean_shift = abs(curr_mean - self._baseline_mean) / (
            abs(self._baseline_mean) + eps
        )
        std_shift  = abs(curr_std  - self._baseline_std)  / (
            self._baseline_std + eps
        )
        raw_drift   = (mean_shift + std_shift) / 2.0
        drift_score = _clamp(raw_drift / _DRIFT_MAX_NORM, 0.0, 1.0)

        # Slow EMA update so the baseline tracks gradual regime changes
        self._baseline_mean = (
            _DRIFT_ALPHA * curr_mean
            + (1.0 - _DRIFT_ALPHA) * self._baseline_mean
        )
        self._baseline_std = max(
            _DRIFT_ALPHA * curr_std
            + (1.0 - _DRIFT_ALPHA) * self._baseline_std,
            0.0,
        )

        return drift_score

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self) -> dict:
        """
        Run the engine against five deterministic internal series.

        Checks per series
        ─────────────────
        - run() returns status == "ok"
        - drift_score ∈ [0.0, 1.0]
        - confidence  ∈ [0.0, 1.0]
        - all signal values are finite

        Score = fraction of series that pass all checks.
        """
        passed  = 0
        total   = len(_BENCHMARK_SERIES)
        details: list[dict] = []

        # Use a fresh engine instance so benchmark never pollutes the live
        # baseline state and remains fully deterministic across calls.
        probe = DataEngine()

        for name, series in _BENCHMARK_SERIES:
            result = probe.run({"values": series})
            ok, detail = self._check_result(result)
            details.append({"series": name, "passed": ok, "detail": detail})
            if ok:
                passed += 1

        score = passed / total if total > 0 else 0.0

        return {
            "engine":  self.name,
            "version": self.version,
            "score":   round(score, 6),
            "metrics": {"series_results": details},
            "passed":  passed == total,
        }

    # ── Self-test ─────────────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Five named internal invariant checks.

        1. determinism_invariant     run() twice → same output (excl. drift)
        2. no_input_mutation         input dict unchanged after run()
        3. all_floats_finite         every float in the output is finite
        4. benchmark_score_stable    benchmark score == 1.0 across 3 fresh runs
        5. drift_bounded             drift_score ∈ [0.0, 1.0] after 10 runs
        """
        checks: list[dict] = []

        # 1 — determinism (drift_score excluded because baseline evolves)
        def _check_determinism() -> bool:
            inp = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
            probe = DataEngine()
            o1 = probe.run(inp)
            probe2 = DataEngine()
            o2 = probe2.run(inp)
            cmp_keys = ("status", "features", "signals", "engine", "version")
            return all(o1.get(k) == o2.get(k) for k in cmp_keys)

        checks.append(_run_check("determinism_invariant", _check_determinism))

        # 2 — no mutation
        def _check_no_mutation() -> bool:
            original = {"values": [1.0, 2.0, 3.0, 4.0, 5.0]}
            snapshot  = {"values": list(original["values"])}
            probe = DataEngine()
            probe.run(original)
            return original == snapshot

        checks.append(_run_check("no_input_mutation", _check_no_mutation))

        # 3 — finite floats
        def _check_finite() -> bool:
            probe = DataEngine()
            result = probe.run({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})
            return probe._check_floats_finite(result)

        checks.append(_run_check("all_floats_finite", _check_finite))

        # 4 — benchmark score stable
        def _check_benchmark_stable() -> bool:
            scores = [DataEngine().benchmark()["score"] for _ in range(3)]
            return all(s == scores[0] == 1.0 for s in scores)

        checks.append(_run_check("benchmark_score_stable", _check_benchmark_stable))

        # 5 — drift bounded after repeated runs
        def _check_drift_bounded() -> bool:
            probe = DataEngine()
            inp   = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
            for _ in range(10):
                r = probe.run(inp)
                if not (0.0 <= r.get("drift_score", -1) <= 1.0):
                    return False
            return True

        checks.append(_run_check("drift_bounded", _check_drift_bounded))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  self.name,
            "version": self.version,
            "checks":  checks,
            "passed":  all_passed,
        }

    # ── Deterministic check ───────────────────────────────────────────────────

    def deterministic_check(self) -> bool:
        """
        Run the canonical input twice (on fresh engine instances) and compare
        all output fields except drift_score (which is baseline-dependent).

        Canonical input: {"values": [1, 2, 3, 4, 5, 6, 7]}
        """
        canonical = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        compared_keys = ("status", "features", "signals", "confidence",
                         "engine", "version")

        probe1 = DataEngine()
        probe2 = DataEngine()
        o1 = probe1.run(canonical)
        o2 = probe2.run(canonical)

        return all(o1.get(k) == o2.get(k) for k in compared_keys)

    # ── State (rollback support) ───────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of all mutable internal state."""
        return {
            "run_count":      self.run_count,
            "baseline_mean":  self._baseline_mean,
            "baseline_std":   self._baseline_std,
        }

    def set_state(self, state: dict) -> None:
        """Restore mutable state from a previously captured snapshot."""
        self.run_count      = int(state.get("run_count", 0))
        bm = state.get("baseline_mean")
        bs = state.get("baseline_std")
        self._baseline_mean = float(bm) if bm is not None else None
        self._baseline_std  = float(bs) if bs is not None else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _error_output(errors: list[str]) -> dict:
        return {
            "status":      "error",
            "features":    {},
            "signals":     {},
            "drift_score": 0.0,
            "confidence":  0.0,
            "engine":      DataEngine._NAME,
            "version":     DataEngine._VERSION,
            "errors":      errors,
        }

    @staticmethod
    def _check_result(result: dict) -> tuple[bool, str]:
        """Validate a single benchmark result dict."""
        if result.get("status") != "ok":
            return False, f"status={result.get('status')!r}"

        drift = result.get("drift_score", -1)
        if not (isinstance(drift, float) and 0.0 <= drift <= 1.0):
            return False, f"drift_score={drift!r} out of [0,1]"

        conf = result.get("confidence", -1)
        if not (isinstance(conf, float) and 0.0 <= conf <= 1.0):
            return False, f"confidence={conf!r} out of [0,1]"

        signals = result.get("signals", {})
        for key, val in signals.items():
            if not isinstance(val, float) or not math.isfinite(val):
                return False, f"signals.{key}={val!r} is not finite"

        return True, "pass"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi].  Handles NaN by returning lo."""
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _safe_round(value: float, digits: int = 8) -> float:
    """Round to digits decimal places.  Returns 0.0 for non-finite input."""
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _run_check(name: str, fn) -> dict:
    """Execute a named check function and catch all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}