"""
A.C.C.E.S.S. — DataEngine Test Suite (Phase 7.1)
tests/test_data_engine.py

Full coverage of agent/domain/data_engine.py

Test sections
─────────────
     1. Schema validation
     2. clean_data
     3. normalize
     4. extract_features
     5. compute_signals
     6. detect_drift
     7. run() — success path
     8. run() — error paths
     9. run() — output contract
    10. Determinism
    11. Input immutability
    12. Benchmark
    13. Self-test
    14. get_state / set_state
    15. SandboxContext integration
    16. DomainEngine ABC compliance
    17. Edge cases
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from agent.domain.data_engine import DataEngine, _clamp, _safe_round
from agent.domain.sandbox import SandboxConfig, SandboxContext


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES / HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_INPUT  = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
CANONICAL_MEAN   = 4.0
CANONICAL_STD    = 2.0

FLAT_INPUT       = {"values": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]}
MINIMUM_INPUT    = {"values": [1.0, 2.0, 3.0, 4.0, 5.0]}   # exactly 5


def _fresh() -> DataEngine:
    """Return a brand-new DataEngine with no baseline state."""
    return DataEngine()


def _run(values: list) -> dict:
    return DataEngine().run({"values": values})


# ─────────────────────────────────────────────────────────────────────────────
# 1 — SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidation:

    def test_valid_input_returns_no_errors(self):
        e = _fresh()
        assert e.validate_schema(CANONICAL_INPUT) == []

    def test_non_dict_returns_error(self):
        e = _fresh()
        errs = e.validate_schema([1, 2, 3, 4, 5])
        assert len(errs) >= 1
        assert any("dict" in err.lower() for err in errs)

    def test_missing_values_key(self):
        e = _fresh()
        errs = e.validate_schema({"other": 42})
        assert any("values" in err for err in errs)

    def test_values_not_a_list(self):
        e = _fresh()
        errs = e.validate_schema({"values": 42})
        assert len(errs) >= 1
        assert any("list" in err.lower() for err in errs)

    def test_too_few_elements(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0]})
        assert any("5" in err or "least" in err for err in errs)

    def test_exactly_five_elements_valid(self):
        e = _fresh()
        assert e.validate_schema(MINIMUM_INPUT) == []

    def test_none_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, None]})
        assert any("None" in err for err in errs)

    def test_nested_list_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, [5.0]]})
        assert any("nested" in err.lower() or "list" in err for err in errs)

    def test_nested_dict_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, {"x": 1}]})
        assert any("nested" in err.lower() or "dict" in err for err in errs)

    def test_string_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, "5.0"]})
        assert len(errs) >= 1

    def test_nan_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, float("nan")]})
        assert any("finite" in err.lower() or "nan" in err.lower() for err in errs)

    def test_inf_element_rejected(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0, 3.0, 4.0, float("inf")]})
        assert any("finite" in err.lower() or "inf" in err.lower() for err in errs)

    def test_integer_elements_accepted(self):
        # int is a valid numeric type
        e = _fresh()
        assert e.validate_schema({"values": [1, 2, 3, 4, 5]}) == []

    def test_mixed_int_float_accepted(self):
        e = _fresh()
        assert e.validate_schema({"values": [1, 2.0, 3, 4.0, 5]}) == []

    def test_error_list_is_list_of_strings(self):
        e = _fresh()
        errs = e.validate_schema({"values": [1.0, 2.0]})
        assert all(isinstance(msg, str) for msg in errs)


# ─────────────────────────────────────────────────────────────────────────────
# 2 — CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanData:

    def test_all_finite_unchanged(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert DataEngine.clean_data(vals) == vals

    def test_nan_removed(self):
        vals = [1.0, float("nan"), 2.0, 3.0, 4.0, 5.0]
        result = DataEngine.clean_data(vals)
        assert all(math.isfinite(v) for v in result)
        assert 1.0 in result and 2.0 in result

    def test_inf_removed(self):
        vals = [1.0, float("inf"), 2.0, 3.0, 4.0, 5.0]
        result = DataEngine.clean_data(vals)
        assert all(math.isfinite(v) for v in result)

    def test_neg_inf_removed(self):
        vals = [float("-inf"), 1.0, 2.0, 3.0, 4.0]
        result = DataEngine.clean_data(vals)
        assert all(math.isfinite(v) for v in result)

    def test_order_preserved(self):
        vals = [5.0, float("nan"), 3.0, float("inf"), 1.0]
        assert DataEngine.clean_data(vals) == [5.0, 3.0, 1.0]

    def test_empty_input_returns_empty(self):
        assert DataEngine.clean_data([]) == []

    def test_does_not_mutate_input(self):
        vals = [1.0, float("nan"), 2.0]
        original = list(vals)
        DataEngine.clean_data(vals)
        assert vals == original


# ─────────────────────────────────────────────────────────────────────────────
# 3 — NORMALIZE
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:

    def test_output_length_matches_input(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(DataEngine.normalize(vals)) == len(vals)

    def test_mean_zero_after_normalization(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        norm = DataEngine.normalize(vals)
        mean = sum(norm) / len(norm)
        assert abs(mean) < 1e-9

    def test_std_one_after_normalization(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        norm = DataEngine.normalize(vals)
        mean = sum(norm) / len(norm)
        var  = sum((v - mean) ** 2 for v in norm) / len(norm)
        assert abs(math.sqrt(var) - 1.0) < 1e-9

    def test_flat_series_gives_all_zeros(self):
        vals = [7.0] * 7
        norm = DataEngine.normalize(vals)
        assert all(v == 0.0 for v in norm)

    def test_canonical_extreme_values(self):
        norm = DataEngine.normalize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert abs(norm[0] - (-1.5)) < 1e-9
        assert abs(norm[-1] - 1.5)   < 1e-9

    def test_all_outputs_finite(self):
        norm = DataEngine.normalize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert all(math.isfinite(v) for v in norm)

    def test_empty_input_returns_empty(self):
        assert DataEngine.normalize([]) == []

    def test_single_value_returns_zero(self):
        # single value → std=0 → 0.0
        assert DataEngine.normalize([42.0]) == [0.0]

    def test_deterministic_same_input_same_output(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        assert DataEngine.normalize(vals) == DataEngine.normalize(vals)


# ─────────────────────────────────────────────────────────────────────────────
# 4 — EXTRACT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFeatures:

    def _feats(self, vals=None):
        vals = vals or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        norm = DataEngine.normalize(vals)
        return DataEngine.extract_features(vals, norm)

    def test_required_keys_present(self):
        f = self._feats()
        for key in ("mean", "std", "rolling_mean", "rolling_std",
                    "ema", "volatility", "max", "min", "count"):
            assert key in f, f"Missing feature key: {key!r}"

    def test_mean_correct(self):
        assert self._feats()["mean"] == CANONICAL_MEAN

    def test_std_correct(self):
        assert self._feats()["std"] == CANONICAL_STD

    def test_max_correct(self):
        assert self._feats()["max"] == 7.0

    def test_min_correct(self):
        assert self._feats()["min"] == 1.0

    def test_count_correct(self):
        assert self._feats()["count"] == 7

    def test_rolling_mean_length(self):
        f = self._feats()
        assert len(f["rolling_mean"]) == 7

    def test_rolling_std_length(self):
        f = self._feats()
        assert len(f["rolling_std"]) == 7

    def test_ema_length(self):
        f = self._feats()
        assert len(f["ema"]) == 7

    def test_all_scalars_finite(self):
        f = self._feats()
        for key in ("mean", "std", "volatility", "max", "min"):
            assert math.isfinite(f[key]), f"features.{key} is not finite"

    def test_all_list_elements_finite(self):
        f = self._feats()
        for key in ("rolling_mean", "rolling_std", "ema"):
            assert all(math.isfinite(v) for v in f[key]), \
                f"Non-finite value in features.{key}"

    def test_flat_series_volatility_zero(self):
        f = self._feats([5.0] * 7)
        assert f["volatility"] == 0.0

    def test_flat_series_std_zero(self):
        f = self._feats([5.0] * 7)
        assert f["std"] == 0.0

    def test_json_serializable(self):
        json.dumps(self._feats())

    def test_deterministic(self):
        vals = [2.0, 4.0, 1.0, 8.0, 5.0, 7.0, 3.0]
        norm = DataEngine.normalize(vals)
        f1 = DataEngine.extract_features(vals, norm)
        f2 = DataEngine.extract_features(vals, norm)
        assert f1 == f2

    def test_volatility_finite_with_zero_in_series(self):
        # Series containing zero — should not divide by zero
        vals = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        norm = DataEngine.normalize(vals)
        f = DataEngine.extract_features(vals, norm)
        assert math.isfinite(f["volatility"])


# ─────────────────────────────────────────────────────────────────────────────
# 5 — COMPUTE SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSignals:

    def _signals(self, vals=None):
        vals = vals or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        norm = DataEngine.normalize(vals)
        feat = DataEngine.extract_features(vals, norm)
        return DataEngine.compute_signals(feat, norm)

    def test_required_keys_present(self):
        s = self._signals()
        for key in ("anomaly_score", "momentum_score", "stability_score"):
            assert key in s

    def test_anomaly_score_in_range(self):
        s = self._signals()
        assert 0.0 <= s["anomaly_score"] <= 1.0

    def test_momentum_score_in_range(self):
        s = self._signals()
        assert -1.0 <= s["momentum_score"] <= 1.0

    def test_stability_score_in_range(self):
        s = self._signals()
        assert 0.0 < s["stability_score"] <= 1.0

    def test_flat_series_anomaly_zero(self):
        s = self._signals([5.0] * 7)
        assert s["anomaly_score"] == 0.0

    def test_flat_series_momentum_zero(self):
        s = self._signals([5.0] * 7)
        assert s["momentum_score"] == 0.0

    def test_flat_series_stability_one(self):
        # zero volatility → stability = 1/(1+0) = 1.0
        s = self._signals([5.0] * 7)
        assert s["stability_score"] == 1.0

    def test_all_signals_finite(self):
        s = self._signals()
        for key, val in s.items():
            assert math.isfinite(val), f"signals.{key} is not finite"

    def test_canonical_anomaly_score(self):
        # last z = +1.5, ANOMALY_SIGMA=3, so anomaly = 1.5/3 = 0.5
        s = self._signals()
        assert abs(s["anomaly_score"] - 0.5) < 1e-6

    def test_canonical_momentum_positive(self):
        # increasing series: last value > mean → positive momentum
        s = self._signals()
        assert s["momentum_score"] > 0.0

    def test_decreasing_series_negative_momentum(self):
        s = self._signals([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        assert s["momentum_score"] < 0.0

    def test_json_serializable(self):
        json.dumps(self._signals())

    def test_deterministic(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        assert self._signals(vals) == self._signals(vals)


# ─────────────────────────────────────────────────────────────────────────────
# 6 — DETECT DRIFT
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDrift:

    def _feats(self, vals):
        norm = DataEngine.normalize(vals)
        return DataEngine.extract_features(vals, norm)

    def test_cold_start_returns_zero(self):
        e = _fresh()
        f = self._feats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert e.detect_drift(f) == 0.0

    def test_cold_start_sets_baseline(self):
        e = _fresh()
        assert e._baseline_mean is None
        f = self._feats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        e.detect_drift(f)
        assert e._baseline_mean is not None

    def test_same_data_drift_near_zero(self):
        e = _fresh()
        f = self._feats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        e.detect_drift(f)          # prime baseline
        drift = e.detect_drift(f)  # same data again
        assert 0.0 <= drift <= 1.0
        assert drift < 0.1         # no meaningful drift

    def test_large_shift_produces_high_drift(self):
        e = _fresh()
        small = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        large = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
        e.detect_drift(self._feats(small))
        drift = e.detect_drift(self._feats(large))
        assert drift > 0.5

    def test_drift_always_in_unit_interval(self):
        e = _fresh()
        series = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [1000.0] * 7,
        ]
        for vals in series:
            drift = e.detect_drift(self._feats(vals))
            assert 0.0 <= drift <= 1.0, f"drift={drift} out of [0,1] for {vals[:3]}…"

    def test_baseline_updated_after_run(self):
        e = _fresh()
        f = self._feats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        e.detect_drift(f)
        baseline_before = e._baseline_mean
        e.detect_drift(f)
        # Baseline should shift slightly (slow EMA update)
        assert e._baseline_mean is not None
        # Same data → baseline converges toward current mean (≈ no change but EMA moves)
        assert math.isfinite(e._baseline_mean)

    def test_drift_score_finite(self):
        e = _fresh()
        f = self._feats([5.0] * 7)   # flat series — std=0
        e.detect_drift(f)
        drift = e.detect_drift(f)
        assert math.isfinite(drift)


# ─────────────────────────────────────────────────────────────────────────────
# 7 — run() SUCCESS PATH
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSuccess:

    def test_status_ok(self):
        assert _fresh().run(CANONICAL_INPUT)["status"] == "ok"

    def test_engine_field(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert r["engine"] == "data_backbone_engine"

    def test_version_field(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert r["version"] == "0.1.0"

    def test_features_present(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert isinstance(r["features"], dict)
        assert len(r["features"]) > 0

    def test_signals_present(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert isinstance(r["signals"], dict)
        assert len(r["signals"]) > 0

    def test_drift_score_present_and_float(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert "drift_score" in r
        assert isinstance(r["drift_score"], float)

    def test_confidence_present_and_float(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert "confidence" in r
        assert isinstance(r["confidence"], float)

    def test_drift_score_in_unit_interval(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert 0.0 <= r["drift_score"] <= 1.0

    def test_confidence_in_unit_interval(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_run_count_increments(self):
        e = _fresh()
        assert e.run_count == 0
        e.run(CANONICAL_INPUT)
        assert e.run_count == 1
        e.run(CANONICAL_INPUT)
        assert e.run_count == 2

    def test_cold_start_drift_zero(self):
        assert _fresh().run(CANONICAL_INPUT)["drift_score"] == 0.0

    def test_canonical_mean(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert r["features"]["mean"] == CANONICAL_MEAN

    def test_canonical_std(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert r["features"]["std"] == CANONICAL_STD

    def test_flat_series_ok(self):
        r = _fresh().run(FLAT_INPUT)
        assert r["status"] == "ok"

    def test_minimum_length_ok(self):
        r = _fresh().run(MINIMUM_INPUT)
        assert r["status"] == "ok"

    def test_integer_values_accepted(self):
        r = _fresh().run({"values": [1, 2, 3, 4, 5, 6, 7]})
        assert r["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 8 — run() ERROR PATHS
# ─────────────────────────────────────────────────────────────────────────────

class TestRunErrors:

    def test_missing_values_key(self):
        r = _fresh().run({})
        assert r["status"] == "error"

    def test_too_few_values(self):
        r = _fresh().run({"values": [1.0, 2.0]})
        assert r["status"] == "error"

    def test_nan_value(self):
        r = _fresh().run({"values": [1.0, 2.0, 3.0, 4.0, float("nan")]})
        assert r["status"] == "error"

    def test_inf_value(self):
        r = _fresh().run({"values": [1.0, 2.0, 3.0, 4.0, float("inf")]})
        assert r["status"] == "error"

    def test_nested_value(self):
        r = _fresh().run({"values": [1.0, 2.0, 3.0, 4.0, [5.0]]})
        assert r["status"] == "error"

    def test_error_output_has_errors_key(self):
        r = _fresh().run({"values": [1.0]})
        assert "errors" in r
        assert isinstance(r["errors"], list)

    def test_error_output_has_required_keys(self):
        r = _fresh().run({})
        for key in ("status", "features", "signals", "drift_score",
                    "confidence", "engine", "version"):
            assert key in r

    def test_error_does_not_increment_run_count(self):
        e = _fresh()
        e.run({"values": [1.0]})   # error
        assert e.run_count == 0

    def test_error_does_not_mutate_baseline(self):
        e = _fresh()
        e.run({"values": [1.0]})
        assert e._baseline_mean is None

    def test_error_output_json_serializable(self):
        r = _fresh().run({"values": [1.0]})
        json.dumps(r)


# ─────────────────────────────────────────────────────────────────────────────
# 9 — run() OUTPUT CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

class TestRunOutputContract:

    def test_output_json_serializable(self):
        r = _fresh().run(CANONICAL_INPUT)
        json.dumps(r)     # must not raise

    def test_no_nan_in_output(self):
        r = _fresh().run(CANONICAL_INPUT)
        self._assert_no_nan(r)

    def test_no_inf_in_output(self):
        r = _fresh().run(CANONICAL_INPUT)
        self._assert_no_inf(r)

    def test_all_floats_finite(self):
        e = _fresh()
        r = e.run(CANONICAL_INPUT)
        assert e._check_floats_finite(r)

    def test_required_top_level_keys(self):
        r = _fresh().run(CANONICAL_INPUT)
        for key in ("status", "features", "signals", "drift_score",
                    "confidence", "engine", "version"):
            assert key in r

    def test_feature_values_in_dict(self):
        r = _fresh().run(CANONICAL_INPUT)
        for key in ("mean", "std", "rolling_mean", "rolling_std",
                    "ema", "volatility", "max", "min", "count"):
            assert key in r["features"]

    def test_signal_values_in_dict(self):
        r = _fresh().run(CANONICAL_INPUT)
        for key in ("anomaly_score", "momentum_score", "stability_score"):
            assert key in r["signals"]

    @staticmethod
    def _assert_no_nan(obj):
        if isinstance(obj, float):
            assert not math.isnan(obj), f"NaN found in output"
        elif isinstance(obj, dict):
            for v in obj.values():
                TestRunOutputContract._assert_no_nan(v)
        elif isinstance(obj, list):
            for v in obj:
                TestRunOutputContract._assert_no_nan(v)

    @staticmethod
    def _assert_no_inf(obj):
        if isinstance(obj, float):
            assert math.isfinite(obj) or obj != obj, f"Inf found: {obj}"
            assert not math.isinf(obj), f"Inf found in output"
        elif isinstance(obj, dict):
            for v in obj.values():
                TestRunOutputContract._assert_no_inf(v)
        elif isinstance(obj, list):
            for v in obj:
                TestRunOutputContract._assert_no_inf(v)


# ─────────────────────────────────────────────────────────────────────────────
# 10 — DETERMINISM
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_two_fresh_engines_same_output(self):
        """Two brand-new engines on the same input must produce identical results."""
        r1 = DataEngine().run(CANONICAL_INPUT)
        r2 = DataEngine().run(CANONICAL_INPUT)
        # Exclude drift_score: depends on baseline state (cold start → 0.0 both)
        for key in ("status", "features", "signals", "confidence", "engine", "version"):
            assert r1[key] == r2[key], f"Mismatch on key {key!r}"

    def test_deterministic_check_returns_true(self):
        e = _fresh()
        assert e.deterministic_check() is True

    def test_deterministic_check_returns_bool(self):
        assert isinstance(_fresh().deterministic_check(), bool)

    def test_features_deterministic(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        norm = DataEngine.normalize(vals)
        f1 = DataEngine.extract_features(vals, norm)
        f2 = DataEngine.extract_features(vals, norm)
        assert f1 == f2

    def test_signals_deterministic(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        norm = DataEngine.normalize(vals)
        feat = DataEngine.extract_features(vals, norm)
        s1 = DataEngine.compute_signals(feat, norm)
        s2 = DataEngine.compute_signals(feat, norm)
        assert s1 == s2

    def test_normalize_deterministic(self):
        vals = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0]
        assert DataEngine.normalize(vals) == DataEngine.normalize(vals)

    def test_clean_data_deterministic(self):
        vals = [1.0, float("nan"), 2.0, float("inf"), 3.0]
        assert DataEngine.clean_data(vals) == DataEngine.clean_data(vals)

    def test_100_runs_same_features(self):
        """100 runs on identical inputs from identical state must yield identical features."""
        results = [DataEngine().run(CANONICAL_INPUT)["features"] for _ in range(100)]
        assert all(r == results[0] for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# 11 — INPUT IMMUTABILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestInputImmutability:

    def test_run_does_not_mutate_input_dict(self):
        original = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
        snapshot  = {"values": list(original["values"])}
        _fresh().run(original)
        assert original == snapshot

    def test_run_does_not_mutate_values_list(self):
        values   = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        snapshot = list(values)
        _fresh().run({"values": values})
        assert values == snapshot

    def test_clean_data_does_not_mutate_input(self):
        vals     = [1.0, float("nan"), 2.0]
        snapshot = list(vals)
        DataEngine.clean_data(vals)
        assert vals == snapshot

    def test_normalize_does_not_mutate_input(self):
        vals     = [1.0, 2.0, 3.0, 4.0, 5.0]
        snapshot = list(vals)
        DataEngine.normalize(vals)
        assert vals == snapshot

    def test_error_path_does_not_mutate_input(self):
        bad = {"values": [1.0]}
        snapshot = {"values": [1.0]}
        _fresh().run(bad)
        assert bad == snapshot


# ─────────────────────────────────────────────────────────────────────────────
# 12 — BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:

    def test_returns_dict(self):
        assert isinstance(_fresh().benchmark(), dict)

    def test_required_keys(self):
        b = _fresh().benchmark()
        for key in ("engine", "version", "score", "metrics", "passed"):
            assert key in b

    def test_score_is_float_in_range(self):
        b = _fresh().benchmark()
        assert isinstance(b["score"], float)
        assert 0.0 <= b["score"] <= 1.0

    def test_passed_is_true(self):
        assert _fresh().benchmark()["passed"] is True

    def test_score_is_one(self):
        assert _fresh().benchmark()["score"] == 1.0

    def test_json_serializable(self):
        json.dumps(_fresh().benchmark())

    def test_does_not_mutate_live_baseline(self):
        """benchmark() must use an isolated probe engine, not self."""
        e = _fresh()
        e.run(CANONICAL_INPUT)                    # set a live baseline
        bm_before = e._baseline_mean
        e.benchmark()
        assert e._baseline_mean == bm_before      # live baseline unchanged

    def test_stable_across_multiple_calls(self):
        scores = [_fresh().benchmark()["score"] for _ in range(3)]
        assert len(set(scores)) == 1              # all identical

    def test_engine_name_correct(self):
        b = _fresh().benchmark()
        assert b["engine"] == "data_backbone_engine"

    def test_metrics_contains_series_results(self):
        b = _fresh().benchmark()
        assert "series_results" in b["metrics"]
        assert isinstance(b["metrics"]["series_results"], list)
        assert len(b["metrics"]["series_results"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 13 — SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfTest:

    def test_returns_dict(self):
        assert isinstance(_fresh().self_test(), dict)

    def test_required_keys(self):
        st = _fresh().self_test()
        for key in ("engine", "version", "checks", "passed"):
            assert key in st

    def test_passed_is_true(self):
        assert _fresh().self_test()["passed"] is True

    def test_checks_is_list(self):
        assert isinstance(_fresh().self_test()["checks"], list)

    def test_exactly_five_checks(self):
        assert len(_fresh().self_test()["checks"]) == 5

    def test_all_checks_pass(self):
        for c in _fresh().self_test()["checks"]:
            assert c["passed"] is True, (
                f"Self-test check {c['name']!r} failed: {c['detail']}"
            )

    def test_each_check_has_required_keys(self):
        for c in _fresh().self_test()["checks"]:
            for key in ("name", "passed", "detail"):
                assert key in c

    def test_json_serializable(self):
        json.dumps(_fresh().self_test())

    def test_check_names_unique(self):
        names = [c["name"] for c in _fresh().self_test()["checks"]]
        assert len(names) == len(set(names))

    def test_determinism_check_passes(self):
        checks = {c["name"]: c for c in _fresh().self_test()["checks"]}
        assert checks["determinism_invariant"]["passed"] is True

    def test_benchmark_stable_check_passes(self):
        checks = {c["name"]: c for c in _fresh().self_test()["checks"]}
        assert checks["benchmark_score_stable"]["passed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 14 — get_state / set_state
# ─────────────────────────────────────────────────────────────────────────────

class TestStateManagement:

    def test_initial_state_json_serializable(self):
        json.dumps(_fresh().get_state())

    def test_initial_state_has_required_keys(self):
        s = _fresh().get_state()
        for key in ("run_count", "baseline_mean", "baseline_std"):
            assert key in s

    def test_initial_run_count_zero(self):
        assert _fresh().get_state()["run_count"] == 0

    def test_initial_baseline_none(self):
        s = _fresh().get_state()
        assert s["baseline_mean"] is None
        assert s["baseline_std"]  is None

    def test_state_after_run_has_correct_run_count(self):
        e = _fresh()
        e.run(CANONICAL_INPUT)
        assert e.get_state()["run_count"] == 1

    def test_state_after_run_has_baseline(self):
        e = _fresh()
        e.run(CANONICAL_INPUT)
        s = e.get_state()
        assert s["baseline_mean"] is not None
        assert s["baseline_std"]  is not None

    def test_set_state_restores_run_count(self):
        e1 = _fresh()
        e1.run(CANONICAL_INPUT)
        e1.run(CANONICAL_INPUT)
        state = e1.get_state()

        e2 = _fresh()
        e2.set_state(state)
        assert e2.run_count == 2

    def test_set_state_restores_baseline(self):
        e1 = _fresh()
        e1.run(CANONICAL_INPUT)
        state = e1.get_state()

        e2 = _fresh()
        e2.set_state(state)
        assert e2._baseline_mean == e1._baseline_mean
        assert e2._baseline_std  == e1._baseline_std

    def test_set_state_none_baseline_accepted(self):
        e = _fresh()
        e.set_state({"run_count": 0, "baseline_mean": None, "baseline_std": None})
        assert e._baseline_mean is None

    def test_state_roundtrip_via_json(self):
        """State can be serialized and restored through JSON."""
        e1 = _fresh()
        e1.run(CANONICAL_INPUT)
        raw_state = json.loads(json.dumps(e1.get_state()))

        e2 = _fresh()
        e2.set_state(raw_state)
        assert e2.run_count     == e1.run_count
        assert e2._baseline_mean == e1._baseline_mean


# ─────────────────────────────────────────────────────────────────────────────
# 15 — SandboxContext integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxContextIntegration:

    def test_run_with_sandbox_context_succeeds(self):
        e   = _fresh()
        ctx = SandboxContext(SandboxConfig())
        r   = e.run(CANONICAL_INPUT, sandbox_context=ctx)
        assert r["status"] == "ok"

    def test_sandbox_action_logged(self):
        e   = _fresh()
        ctx = SandboxContext(SandboxConfig())
        e.run(CANONICAL_INPUT, sandbox_context=ctx)
        assert len(ctx.actions) >= 1

    def test_logged_action_is_filesystem_read(self):
        e   = _fresh()
        ctx = SandboxContext(SandboxConfig())
        e.run(CANONICAL_INPUT, sandbox_context=ctx)
        actions = [a.action for a in ctx.actions]
        assert "filesystem_read" in actions

    def test_run_without_context_succeeds(self):
        e = _fresh()
        r = e.run(CANONICAL_INPUT, sandbox_context=None)
        assert r["status"] == "ok"

    def test_output_identical_with_and_without_context(self):
        """Context must not alter the engine output."""
        ctx = SandboxContext(SandboxConfig())
        r1  = DataEngine().run(CANONICAL_INPUT, sandbox_context=ctx)
        r2  = DataEngine().run(CANONICAL_INPUT, sandbox_context=None)
        for key in ("status", "features", "signals", "drift_score",
                    "confidence", "engine", "version"):
            assert r1[key] == r2[key]


# ─────────────────────────────────────────────────────────────────────────────
# 16 — DomainEngine ABC compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainEngineCompliance:

    def test_name_is_string(self):
        assert isinstance(_fresh().name, str)

    def test_name_value(self):
        assert _fresh().name == "data_backbone_engine"

    def test_version_is_string(self):
        assert isinstance(_fresh().version, str)

    def test_version_value(self):
        assert _fresh().version == "0.1.0"

    def test_required_permissions_is_frozenset(self):
        assert isinstance(_fresh().required_permissions, frozenset)

    def test_required_permissions_contains_filesystem_read(self):
        assert "filesystem_read" in _fresh().required_permissions

    def test_run_returns_dict(self):
        assert isinstance(_fresh().run(CANONICAL_INPUT), dict)

    def test_run_contains_status(self):
        r = _fresh().run(CANONICAL_INPUT)
        assert "status" in r

    def test_benchmark_returns_dict(self):
        assert isinstance(_fresh().benchmark(), dict)

    def test_benchmark_score_finite(self):
        assert math.isfinite(_fresh().benchmark()["score"])

    def test_self_test_returns_dict(self):
        assert isinstance(_fresh().self_test(), dict)

    def test_self_test_has_passed(self):
        assert "passed" in _fresh().self_test()

    def test_deterministic_check_returns_bool(self):
        assert isinstance(_fresh().deterministic_check(), bool)

    def test_metadata_returns_metadata_object(self):
        from agent.domain.base import DomainEngineMetadata
        meta = _fresh().metadata()
        assert isinstance(meta, DomainEngineMetadata)
        assert meta.name == "data_backbone_engine"

    def test_validate_input_non_dict_returns_errors(self):
        errors = _fresh().validate_input("not_a_dict")
        assert len(errors) > 0

    def test_repr_contains_class_name(self):
        assert "DataEngine" in repr(_fresh())


# ─────────────────────────────────────────────────────────────────────────────
# 17 — EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_same_value(self):
        """Constant series: std=0, all normalized=0, stability=1."""
        r = _run([3.14] * 10)
        assert r["status"] == "ok"
        assert r["features"]["std"] == 0.0
        assert r["signals"]["stability_score"] == 1.0
        assert r["signals"]["anomaly_score"]    == 0.0

    def test_all_zeros_with_drift(self):
        """Series of zeros — returns volatility should be 0."""
        r = _run([0.0] * 7)
        assert r["status"] == "ok"
        assert r["features"]["volatility"] == 0.0

    def test_large_values_finite(self):
        vals = [1e12, 2e12, 3e12, 4e12, 5e12, 6e12, 7e12]
        r = _run(vals)
        assert r["status"] == "ok"
        assert math.isfinite(r["confidence"])

    def test_negative_values(self):
        vals = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]
        r = _run(vals)
        assert r["status"] == "ok"
        assert r["features"]["max"] == -1.0
        assert r["features"]["min"] == -7.0

    def test_mixed_sign_values(self):
        vals = [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0]
        r = _run(vals)
        assert r["status"] == "ok"
        assert math.isfinite(r["drift_score"])

    def test_series_with_interior_zeros(self):
        """Zeros in the middle must not cause division-by-zero in returns."""
        vals = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]
        r = _run(vals)
        assert r["status"] == "ok"
        assert math.isfinite(r["features"]["volatility"])

    def test_run_many_times_baseline_converges(self):
        """Running 50 times with the same series: baseline stabilises."""
        e   = _fresh()
        inp = CANONICAL_INPUT
        drifts = [e.run(inp)["drift_score"] for _ in range(50)]
        # After convergence, drift on unchanged data must be very small
        assert drifts[-1] < 0.1
        # All drift scores finite and bounded
        assert all(math.isfinite(d) and 0.0 <= d <= 1.0 for d in drifts)

    def test_confidence_decreases_with_high_drift(self):
        """High drift should pull confidence below the cold-start value."""
        e = _fresh()
        r_baseline = e.run(CANONICAL_INPUT)          # cold start: drift=0
        conf_cold  = r_baseline["confidence"]

        big = {"values": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0]}
        r_shifted = e.run(big)                        # big shift: drift > 0
        assert r_shifted["drift_score"] > 0.0
        # confidence must be ≤ cold-start confidence (drift reduces it)
        assert r_shifted["confidence"] <= conf_cold + 1e-9

    def test_clamp_helper_finite(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5
        assert _clamp(-1.0, 0.0, 1.0) == 0.0
        assert _clamp(2.0, 0.0, 1.0) == 1.0
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_safe_round_non_finite_returns_zero(self):
        assert _safe_round(float("nan")) == 0.0
        assert _safe_round(float("inf")) == 0.0