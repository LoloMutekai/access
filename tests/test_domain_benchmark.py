"""
Tests for agent/domain/benchmark.py

Coverage:
    BenchmarkDataset  — construction and validation
    BenchmarkResult   — structure, score bounds, serialization
    BenchmarkRunner   — run, history, regression, compare
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.domain.benchmark import BenchmarkDataset, BenchmarkResult, BenchmarkRunner
from agent.domain.dummy import DummyDomainEngine


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_datasets(n: int = 3) -> list[BenchmarkDataset]:
    return [
        BenchmarkDataset(
            name=f"ds_{i}",
            input_data={"value": i},
            expected_values={"status": "ok"},
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK DATASET TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkDataset:

    def test_construction_valid(self):
        ds = BenchmarkDataset(
            name="test_dataset",
            input_data={"x": 1},
            expected_values={"status": "ok"},
        )
        assert ds.name == "test_dataset"
        assert ds.weight == 1.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            BenchmarkDataset(name="", input_data={}, expected_values={})

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            BenchmarkDataset(
                name="ds",
                input_data={},
                expected_values={},
                weight=0.0,
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            BenchmarkDataset(
                name="ds",
                input_data={},
                expected_values={},
                weight=-1.0,
            )

    def test_nan_weight_raises(self):
        with pytest.raises(ValueError):
            BenchmarkDataset(
                name="ds",
                input_data={},
                expected_values={},
                weight=float("nan"),
            )

    def test_to_dict_json_serializable(self):
        ds = BenchmarkDataset(
            name="ds",
            input_data={"x": 1},
            expected_values={"status": "ok"},
        )
        json.dumps(ds.to_dict())

    def test_frozen_immutable(self):
        ds = BenchmarkDataset(name="ds", input_data={}, expected_values={})
        with pytest.raises((AttributeError, TypeError)):
            ds.name = "hacked"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RESULT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkResult:

    def _make_result(self, score: float = 1.0) -> BenchmarkResult:
        return BenchmarkResult(
            engine="dummy",
            version="1.0.0",
            score=score,
            metrics={},
            regression_detected=False,
            previous_score=None,
            run_at="2026-01-01T00:00:00+00:00",
            dataset_count=1,
            passed_count=1,
        )

    def test_valid_score_construction(self):
        r = self._make_result(0.75)
        assert r.score == 0.75

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            self._make_result(1.1)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            self._make_result(-0.1)

    def test_nan_score_raises(self):
        with pytest.raises(ValueError):
            self._make_result(float("nan"))

    def test_to_dict_json_serializable(self):
        r = self._make_result()
        json.dumps(r.to_dict())

    def test_to_dict_contains_required_keys(self):
        r = self._make_result()
        d = r.to_dict()
        for key in ("engine", "version", "score", "metrics",
                    "regression_detected", "run_at"):
            assert key in d

    def test_repr_contains_engine(self):
        r = self._make_result()
        assert "dummy" in repr(r)

    def test_repr_regression_marker(self):
        r = BenchmarkResult(
            engine="e",
            version="1.0",
            score=0.5,
            metrics={},
            regression_detected=True,
            previous_score=0.9,
            run_at="2026-01-01T00:00:00+00:00",
            dataset_count=1,
            passed_count=0,
        )
        assert "REGRESSION" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkRunner:

    def test_run_returns_benchmark_result(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine, _make_datasets())
        assert isinstance(result, BenchmarkResult)

    def test_run_score_is_one_for_passing_engine(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine, _make_datasets(5))
        assert result.score == 1.0

    def test_run_score_in_range(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine, _make_datasets())
        assert 0.0 <= result.score <= 1.0

    def test_run_no_regression_on_first_run(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine, _make_datasets())
        assert result.regression_detected is False
        assert result.previous_score is None

    def test_run_failing_dataset_reduces_score(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        bad_dataset = BenchmarkDataset(
            name="bad",
            input_data={"x": 1},
            expected_values={"status": "impossible_value"},
        )
        result = runner.run(engine, [bad_dataset])
        assert result.score == 0.0

    def test_regression_detected_on_score_drop(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()

        passing = _make_datasets(3)
        runner.run(engine, passing)  # first run: score=1.0

        failing = [
            BenchmarkDataset(
                name="fail",
                input_data={},
                expected_values={"status": "impossible"},
            )
        ]
        result2 = runner.run(engine, failing)  # second run: score=0.0
        assert result2.regression_detected is True
        assert result2.previous_score == 1.0

    def test_no_regression_when_score_improves(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()

        failing = [
            BenchmarkDataset(
                name="fail",
                input_data={},
                expected_values={"status": "impossible"},
            )
        ]
        runner.run(engine, failing)   # score=0.0

        passing = _make_datasets(3)
        result2 = runner.run(engine, passing)  # score=1.0
        assert result2.regression_detected is False

    def test_history_stored_after_run(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        runner.run(engine, _make_datasets())
        history = runner.get_history(engine.name, engine.version)
        assert len(history) == 1

    def test_history_grows_per_run(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        for _ in range(3):
            runner.run(engine, _make_datasets())
        history = runner.get_history(engine.name, engine.version)
        assert len(history) == 3

    def test_clear_history_single_engine(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        runner.run(engine, _make_datasets())
        runner.clear_history(engine.name, engine.version)
        history = runner.get_history(engine.name, engine.version)
        assert len(history) == 0

    def test_clear_all_history(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        runner.run(engine, _make_datasets())
        runner.clear_history()
        history = runner.get_history(engine.name, engine.version)
        assert len(history) == 0

    def test_compare_last_two_none_on_single_run(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        runner.run(engine, _make_datasets())
        comparison = runner.compare_last_two(engine.name, engine.version)
        assert comparison is None

    def test_compare_last_two_returns_dict(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        runner.run(engine, _make_datasets())
        runner.run(engine, _make_datasets())
        comparison = runner.compare_last_two(engine.name, engine.version)
        assert isinstance(comparison, dict)
        for key in ("score_delta", "improved", "regressed", "stable"):
            assert key in comparison

    def test_run_without_datasets_uses_engine_benchmark(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine)
        assert isinstance(result, BenchmarkResult)
        assert result.score == 1.0

    def test_result_is_json_serializable(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        result = runner.run(engine, _make_datasets())
        json.dumps(result.to_dict())

    def test_weighted_datasets_affect_score(self):
        runner = BenchmarkRunner()
        engine = DummyDomainEngine()
        heavy_pass = BenchmarkDataset(
            name="heavy_pass",
            input_data={"x": 1},
            expected_values={"status": "ok"},
            weight=9.0,
        )
        light_fail = BenchmarkDataset(
            name="light_fail",
            input_data={},
            expected_values={"status": "impossible"},
            weight=1.0,
        )
        result = runner.run(engine, [heavy_pass, light_fail])
        # 9/(9+1) = 0.9 expected
        assert abs(result.score - 0.9) < 1e-6