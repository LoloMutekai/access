"""
A.C.C.E.S.S. — Benchmark Harness (Phase 7.0)

Provides deterministic, versioned benchmarking for DomainEngine instances.

Design:
    - BenchmarkRunner accepts a DomainEngine and a list of BenchmarkDataset
    - Each dataset is a frozen (input, expected_output) pair
    - Score is computed as the fraction of datasets whose engine output
      matches the expected output on a configurable set of keys
    - Results are stored in-memory in a versioned log (no file I/O)
    - Regression detection compares the current score against the most
      recent stored result for the same engine/version pair

Output guarantee:
    All BenchmarkResult fields are JSON-serializable.
    score ∈ [0.0, 1.0].
    regression_detected is deterministic given stored history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BenchmarkDataset:
    """
    A single deterministic input/output pair for benchmarking.

    name            : Unique identifier for this dataset within a suite
    input_data      : Input dict passed to engine.run()
    expected_keys   : Set of result keys to check in the engine output
    expected_values : Dict of key → expected value (subset of output)
    weight          : Relative weight of this dataset in the final score
    """
    name: str
    input_data: dict
    expected_values: dict
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("BenchmarkDataset.name must be non-empty.")
        if self.weight <= 0.0 or not math.isfinite(self.weight):
            raise ValueError(
                f"BenchmarkDataset.weight must be a positive finite float, "
                f"got {self.weight}"
            )

    def to_dict(self) -> dict:
        return {
            "name":            self.name,
            "input_data":      self.input_data,
            "expected_values": self.expected_values,
            "weight":          self.weight,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BenchmarkResult:
    """
    Frozen, JSON-serializable output of a benchmark run.

    engine              : Engine name
    version             : Engine version
    score               : Weighted fraction of passing datasets ∈ [0.0, 1.0]
    metrics             : Per-dataset pass/fail and detail
    regression_detected : True iff score < previous score for same engine+version
    previous_score      : Score of the most recent prior run, or None
    run_at              : ISO-8601 UTC timestamp
    dataset_count       : Number of datasets evaluated
    passed_count        : Number of datasets that passed
    """
    engine: str
    version: str
    score: float
    metrics: dict
    regression_detected: bool
    previous_score: Optional[float]
    run_at: str
    dataset_count: int
    passed_count: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0) or not math.isfinite(self.score):
            raise ValueError(
                f"BenchmarkResult.score must be in [0.0, 1.0], got {self.score}"
            )

    def to_dict(self) -> dict:
        return {
            "engine":             self.engine,
            "version":            self.version,
            "score":              round(self.score, 6),
            "metrics":            self.metrics,
            "regression_detected": self.regression_detected,
            "previous_score":     self.previous_score,
            "run_at":             self.run_at,
            "dataset_count":      self.dataset_count,
            "passed_count":       self.passed_count,
        }

    def __repr__(self) -> str:
        reg = " ⚠ REGRESSION" if self.regression_detected else ""
        return (
            f"BenchmarkResult("
            f"engine={self.engine!r}, "
            f"version={self.version!r}, "
            f"score={self.score:.4f}, "
            f"passed={self.passed_count}/{self.dataset_count}"
            f"{reg})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs deterministic benchmarks against a DomainEngine.

    State:
        _history : dict keyed by (engine_name, version) → list[BenchmarkResult]
        This is intentionally non-persistent. For durable storage, serialise
        the to_dict() output of each result to your persistence layer.

    Usage:
        runner  = BenchmarkRunner()
        dataset = BenchmarkDataset(
            name="trivial",
            input_data={"value": 1},
            expected_values={"status": "ok"},
        )
        result = runner.run(engine, datasets=[dataset])
        assert result.score == 1.0
    """

    def __init__(self) -> None:
        # (engine_name, version) → ordered list of BenchmarkResult
        self._history: dict[tuple[str, str], list[BenchmarkResult]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        engine: Any,
        datasets: Optional[list[BenchmarkDataset]] = None,
        regression_threshold: float = 0.0,
    ) -> BenchmarkResult:
        """
        Benchmark engine against datasets and store the result.

        Args:
            engine               : DomainEngine instance to benchmark.
            datasets             : List of BenchmarkDataset. If None, calls
                                   engine.benchmark() and wraps its output.
            regression_threshold : Score drop below which regression is flagged.
                                   0.0 means any decrease triggers a regression.

        Returns:
            BenchmarkResult — frozen, JSON-serializable.
        """
        engine_name = getattr(engine, "name",    "unknown")
        version     = getattr(engine, "version", "unknown")

        if datasets is not None and len(datasets) > 0:
            result = self._run_datasets(
                engine, engine_name, version, datasets, regression_threshold
            )
        else:
            result = self._run_engine_benchmark(
                engine, engine_name, version, regression_threshold
            )

        key = (engine_name, version)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(result)

        return result

    def get_history(
        self,
        engine_name: str,
        version: str,
    ) -> list[BenchmarkResult]:
        """Return all stored results for an engine+version pair (oldest first)."""
        return list(self._history.get((engine_name, version), []))

    def clear_history(
        self,
        engine_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Clear stored history.
        If engine_name and version are given, clears only that entry.
        If neither is given, clears all history.
        """
        if engine_name is not None and version is not None:
            self._history.pop((engine_name, version), None)
        else:
            self._history.clear()

    def compare_last_two(
        self,
        engine_name: str,
        version: str,
    ) -> Optional[dict]:
        """
        Compare the two most recent runs for an engine+version pair.

        Returns:
            dict with keys: score_delta, improved, regressed, stable
            or None if fewer than two results exist.
        """
        history = self.get_history(engine_name, version)
        if len(history) < 2:
            return None
        prev, curr = history[-2], history[-1]
        delta = curr.score - prev.score
        return {
            "previous_score": prev.score,
            "current_score":  curr.score,
            "score_delta":    round(delta, 6),
            "improved":       delta > 0,
            "regressed":      delta < 0,
            "stable":         delta == 0.0,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_datasets(
        self,
        engine: Any,
        engine_name: str,
        version: str,
        datasets: list[BenchmarkDataset],
        regression_threshold: float,
    ) -> BenchmarkResult:
        """Evaluate engine against an explicit list of BenchmarkDataset."""
        per_dataset_metrics: list[dict] = []
        total_weight = 0.0
        passed_weight = 0.0
        passed_count = 0

        for ds in datasets:
            try:
                output = engine.run(ds.input_data)
                passed, detail = self._check_expected(output, ds.expected_values)
            except Exception as exc:
                passed = False
                detail = f"run() raised: {type(exc).__name__}: {exc}"

            per_dataset_metrics.append({
                "dataset":  ds.name,
                "passed":   passed,
                "weight":   ds.weight,
                "detail":   detail,
            })
            total_weight  += ds.weight
            if passed:
                passed_weight += ds.weight
                passed_count  += 1

        score = (passed_weight / total_weight) if total_weight > 0.0 else 0.0
        score = max(0.0, min(1.0, score))

        previous_score = self._last_score(engine_name, version)
        regression = self._detect_regression(score, previous_score, regression_threshold)

        return BenchmarkResult(
            engine=engine_name,
            version=version,
            score=score,
            metrics={"datasets": per_dataset_metrics},
            regression_detected=regression,
            previous_score=previous_score,
            run_at=datetime.now(UTC).isoformat(),
            dataset_count=len(datasets),
            passed_count=passed_count,
        )

    def _run_engine_benchmark(
        self,
        engine: Any,
        engine_name: str,
        version: str,
        regression_threshold: float,
    ) -> BenchmarkResult:
        """
        Delegate to engine.benchmark() and wrap its output as a BenchmarkResult.
        Used when no external datasets are provided.
        """
        try:
            raw = engine.benchmark()
            score = float(raw.get("score", 0.0))
            score = max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
            passed_flag = bool(raw.get("passed", score >= 0.5))
        except Exception as exc:
            raw = {"error": f"{type(exc).__name__}: {exc}"}
            score = 0.0
            passed_flag = False

        previous_score = self._last_score(engine_name, version)
        regression = self._detect_regression(score, previous_score, regression_threshold)

        return BenchmarkResult(
            engine=engine_name,
            version=version,
            score=score,
            metrics=raw,
            regression_detected=regression,
            previous_score=previous_score,
            run_at=datetime.now(UTC).isoformat(),
            dataset_count=1,
            passed_count=1 if passed_flag else 0,
        )

    def _check_expected(
        self,
        output: dict,
        expected_values: dict,
    ) -> tuple[bool, str]:
        """
        Check whether engine output matches all expected_values entries.

        Returns:
            (True, "pass") if all keys match,
            (False, detail_string) if any key mismatches or is missing.
        """
        if not isinstance(output, dict):
            return False, f"output is not a dict: {type(output).__name__}"

        for key, expected in expected_values.items():
            if key not in output:
                return False, f"missing key {key!r} in output"
            actual = output[key]
            if actual != expected:
                return False, (
                    f"key {key!r}: expected {expected!r}, got {actual!r}"
                )
        return True, "pass"

    def _last_score(self, engine_name: str, version: str) -> Optional[float]:
        history = self._history.get((engine_name, version), [])
        return history[-1].score if history else None

    def _detect_regression(
        self,
        score: float,
        previous_score: Optional[float],
        threshold: float,
    ) -> bool:
        if previous_score is None:
            return False
        return (previous_score - score) > threshold