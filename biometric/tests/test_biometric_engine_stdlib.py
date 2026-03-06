"""
A.C.C.E.S.S. — Phase 7.3 Test Suite: BiometricEngine (Foundation)

Coverage map
────────────
SECTION 1 — validate_schema()
    1.1  Non-dict input rejected
    1.2  Missing channel raises SchemaError
    1.3  Channel that is not a list raises SchemaError
    1.4  Non-numeric element raises SchemaError
    1.5  Channel shorter than min_samples raises SchemaError
    1.6  Exactly min_samples passes
    1.7  int elements accepted (coerced to float downstream)
    1.8  Mixed int/float elements accepted
    1.9  Empty list rejected before min_samples check

SECTION 2 — clean_data()
    2.1  All-finite input passes through unchanged
    2.2  NaN values are removed
    2.3  +Inf values are removed
    2.4  -Inf values are removed
    2.5  Channel that drops below min_samples post-clean raises SchemaError
    2.6  strict_guards=True removes out-of-range HR
    2.7  strict_guards=False keeps out-of-range HR
    2.8  Returns CleanedSignals (immutable)
    2.9  Original dict is not mutated

SECTION 3 — normalize()
    3.1  All values in [0.0, 1.0]
    3.2  Min value maps to 0.0
    3.3  Max value maps to 1.0
    3.4  Constant channel maps to all-zeros (no division-by-zero)
    3.5  Two-element channel normalised correctly
    3.6  Monotone increasing sequence: result is strictly increasing
    3.7  Returns NormalizedSignals (immutable)

SECTION 4 — compute_core_metrics()
    4.1  mean_hr is arithmetic mean of HR
    4.2  mean_hrv is arithmetic mean of HRV
    4.3  load_mean is arithmetic mean of load
    4.4  hr_std is population std-dev of HR
    4.5  hrv_std is population std-dev of HRV
    4.6  Single-element channel → std = 0.0
    4.7  Constant channel → std = 0.0
    4.8  All metrics are finite
    4.9  to_dict() returns correct keys
    4.10 Rounding applied to output_precision

SECTION 5 — process() integration
    5.1  Valid input → status == "ok"
    5.2  Valid input → all five metrics present
    5.3  Valid input → engine == "BiometricEngine"
    5.4  Valid input → version == "7.3.0"
    5.5  Invalid input → status == "error"
    5.6  Invalid input → metrics is None
    5.7  Invalid input → "error" key present
    5.8  process() never raises (error capture)
    5.9  Output is JSON-serialisable
    5.10 Output metrics are all finite floats

SECTION 6 — determinism
    6.1  Identical inputs → identical outputs
    6.2  Multiple BiometricEngine instances → identical outputs
    6.3  No randomness introduced across 100 calls

SECTION 7 — edge cases
    7.1  Very small channel (exactly 10 samples)
    7.2  Large channel (10 000 samples)
    7.3  All-zero channel
    7.4  Negative values (e.g., load can theoretically be negative)
    7.5  Very large values
    7.6  Very small positive values (near zero)
    7.7  Mixed sign values
"""

from __future__ import annotations

import json
import math
import sys
import os

import pytest

# ── path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.biometric_engine import (
    ANOMALY_HR_REF,
    ANOMALY_HRV_REF,
    ANOMALY_W_DRIFT,
    ANOMALY_W_FATIGUE,
    ANOMALY_W_HR,
    ANOMALY_W_HRV,
    AnomalyResult,
    AnomalySignals,
    BiometricConfig,
    BiometricEngine,
    CleanedSignals,
    CoreMetrics,
    DRIFT_W_FATIGUE,
    DRIFT_W_HR,
    DRIFT_W_HRV,
    DriftResult,
    ENGINE_NAME,
    ENGINE_VERSION,
    FATIGUE_W1,
    FATIGUE_W2,
    FATIGUE_W3,
    FatigueResult,
    InjuryRiskResult,
    MIN_SAMPLES,
    NormalizedSignals,
    RISK_W_DRIFT,
    RISK_W_FATIGUE,
    RISK_W_LOAD,
    SchemaError,
    _bounded_sigmoid,
    _clamp,
    _min_max_normalize,
    _safe_mean,
    _safe_pstdev,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_input(n: int = 10) -> dict:
    """Return a minimal valid input dict of length n."""
    hr   = [float(60 + i) for i in range(n)]
    hrv  = [float(40 + i) for i in range(n)]
    load = [float(100 + i * 10) for i in range(n)]
    return {"hr": hr, "hrv": hrv, "load": load}


def _make_engine(strict: bool = False) -> BiometricEngine:
    return BiometricEngine(BiometricConfig(strict_guards=strict))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — validate_schema()
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSchema:

    def test_1_1_non_dict_input_rejected(self):
        eng = _make_engine()
        with pytest.raises(SchemaError, match="must be a dict"):
            eng.validate_schema([1, 2, 3])

    def test_1_1b_none_rejected(self):
        eng = _make_engine()
        with pytest.raises(SchemaError):
            eng.validate_schema(None)

    def test_1_2_missing_hr_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        del raw["hr"]
        with pytest.raises(SchemaError, match="Missing required channel"):
            eng.validate_schema(raw)

    def test_1_2_missing_hrv_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        del raw["hrv"]
        with pytest.raises(SchemaError, match="Missing required channel"):
            eng.validate_schema(raw)

    def test_1_2_missing_load_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        del raw["load"]
        with pytest.raises(SchemaError, match="Missing required channel"):
            eng.validate_schema(raw)

    def test_1_3_channel_not_list_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["hr"] = (60.0, 61.0)  # tuple, not list
        with pytest.raises(SchemaError, match="must be a list"):
            eng.validate_schema(raw)

    def test_1_4_non_numeric_element_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["hr"][3] = "bad"
        with pytest.raises(SchemaError, match="must be numeric"):
            eng.validate_schema(raw)

    def test_1_5_too_short_raises(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["hr"] = [60.0] * (MIN_SAMPLES - 1)
        with pytest.raises(SchemaError, match="minimum required"):
            eng.validate_schema(raw)

    def test_1_6_exactly_min_samples_passes(self):
        eng = _make_engine()
        raw = _make_valid_input(n=MIN_SAMPLES)
        eng.validate_schema(raw)  # must not raise

    def test_1_7_int_elements_accepted(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["hr"] = list(range(60, 60 + MIN_SAMPLES))  # all ints
        eng.validate_schema(raw)  # must not raise

    def test_1_8_mixed_int_float_accepted(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["hrv"] = [40, 41.5, 42, 43.0, 44, 45, 46, 47.5, 48, 49]
        eng.validate_schema(raw)  # must not raise

    def test_1_9_empty_list_rejected(self):
        eng = _make_engine()
        raw = _make_valid_input()
        raw["load"] = []
        with pytest.raises(SchemaError):
            eng.validate_schema(raw)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — clean_data()
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanData:

    def test_2_1_all_finite_unchanged(self):
        eng = _make_engine()
        raw = _make_valid_input(n=12)
        cleaned = eng.clean_data(raw)
        assert list(cleaned.hr) == [float(v) for v in raw["hr"]]

    def test_2_2_nan_removed(self):
        eng = _make_engine()
        raw = _make_valid_input(n=15)
        raw["hr"].insert(5, float("nan"))
        cleaned = eng.clean_data(raw)
        assert all(math.isfinite(v) for v in cleaned.hr)
        assert len(cleaned.hr) == 15  # 15 good + 1 nan removed; 15 remain

    def test_2_3_pos_inf_removed(self):
        eng = _make_engine()
        raw = _make_valid_input(n=15)
        raw["hrv"][0] = float("inf")
        cleaned = eng.clean_data(raw)
        assert all(math.isfinite(v) for v in cleaned.hrv)
        assert len(cleaned.hrv) == 14

    def test_2_4_neg_inf_removed(self):
        eng = _make_engine()
        raw = _make_valid_input(n=15)
        raw["load"][2] = float("-inf")
        cleaned = eng.clean_data(raw)
        assert all(math.isfinite(v) for v in cleaned.load)
        assert len(cleaned.load) == 14

    def test_2_5_drops_below_min_raises(self):
        eng = _make_engine()
        # Start with exactly min_samples; inject NaN to push below threshold.
        raw = _make_valid_input(n=MIN_SAMPLES)
        raw["hr"][0] = float("nan")
        with pytest.raises(SchemaError, match="after cleaning"):
            eng.clean_data(raw)

    def test_2_6_strict_guards_removes_out_of_range_hr(self):
        eng = _make_engine(strict=True)
        raw = _make_valid_input(n=15)
        raw["hr"].extend([500.0, -5.0])  # physiologically impossible
        cleaned = eng.clean_data(raw)
        assert all(eng._cfg.hr_min <= v <= eng._cfg.hr_max for v in cleaned.hr)

    def test_2_7_lenient_mode_keeps_out_of_range_hr(self):
        eng = _make_engine(strict=False)
        raw = _make_valid_input(n=15)
        raw["hr"].append(500.0)          # out of range but finite
        cleaned = eng.clean_data(raw)
        assert 500.0 in cleaned.hr       # kept in lenient mode

    def test_2_8_returns_cleaned_signals(self):
        eng = _make_engine()
        raw = _make_valid_input()
        cleaned = eng.clean_data(raw)
        assert isinstance(cleaned, CleanedSignals)

    def test_2_9_original_dict_not_mutated(self):
        eng = _make_engine()
        raw = _make_valid_input(n=15)
        original_hr = list(raw["hr"])
        raw["hr"].insert(3, float("nan"))
        _ = eng.clean_data(raw)
        # The nan we manually injected is still there — we didn't touch raw
        assert float("nan") != float("nan")  # sanity: nan != nan
        assert len(raw["hr"]) == 16          # original list not shrunk


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — normalize()
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:

    def _cleaned(self, hr=None, hrv=None, load=None) -> CleanedSignals:
        default = tuple(float(i) for i in range(10, 20))
        return CleanedSignals(
            hr=hr or default,
            hrv=hrv or default,
            load=load or default,
        )

    def test_3_1_all_values_in_unit_interval(self):
        eng = _make_engine()
        cleaned = self._cleaned()
        normed = eng.normalize(cleaned)
        for ch in ("hr", "hrv", "load"):
            assert all(0.0 <= v <= 1.0 for v in normed.channel(ch))

    def test_3_2_min_maps_to_zero(self):
        eng = _make_engine()
        cleaned = self._cleaned(hr=tuple(range(50, 61)))
        normed = eng.normalize(cleaned)
        assert normed.hr[0] == pytest.approx(0.0)

    def test_3_3_max_maps_to_one(self):
        eng = _make_engine()
        cleaned = self._cleaned(hr=tuple(range(50, 61)))
        normed = eng.normalize(cleaned)
        assert normed.hr[-1] == pytest.approx(1.0)

    def test_3_4_constant_channel_all_zeros(self):
        eng = _make_engine()
        cleaned = self._cleaned(hr=tuple(75.0 for _ in range(10)))
        normed = eng.normalize(cleaned)
        assert all(v == 0.0 for v in normed.hr)

    def test_3_5_two_element_channel(self):
        # Build a minimal CleanedSignals with len=10 but known boundary values
        vals = (0.0,) * 8 + (0.0, 10.0)
        eng = _make_engine()
        cleaned = self._cleaned(hr=vals)
        normed = eng.normalize(cleaned)
        assert normed.hr[-1] == pytest.approx(1.0)
        assert normed.hr[0] == pytest.approx(0.0)

    def test_3_6_monotone_increasing_stays_increasing(self):
        eng = _make_engine()
        cleaned = self._cleaned(hrv=tuple(float(i) for i in range(10, 20)))
        normed = eng.normalize(cleaned)
        for a, b in zip(normed.hrv, normed.hrv[1:]):
            assert a < b

    def test_3_7_returns_normalized_signals(self):
        eng = _make_engine()
        cleaned = self._cleaned()
        normed = eng.normalize(cleaned)
        assert isinstance(normed, NormalizedSignals)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — compute_core_metrics()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCoreMetrics:

    def _make_cleaned(
        self,
        hr=(70.0, 72.0, 68.0, 74.0, 71.0, 73.0, 69.0, 75.0, 70.0, 72.0),
        hrv=(40.0, 42.0, 38.0, 44.0, 41.0, 43.0, 39.0, 45.0, 40.0, 42.0),
        load=(100.0,) * 10,
    ) -> CleanedSignals:
        return CleanedSignals(hr=tuple(hr), hrv=tuple(hrv), load=tuple(load))

    def test_4_1_mean_hr_correct(self):
        eng = _make_engine()
        hr = (70.0, 80.0, 90.0, 60.0, 75.0, 85.0, 65.0, 78.0, 82.0, 71.0)
        cleaned = self._make_cleaned(hr=hr)
        m = eng.compute_core_metrics(cleaned)
        expected = sum(hr) / len(hr)
        assert m.mean_hr == pytest.approx(expected, abs=1e-5)

    def test_4_2_mean_hrv_correct(self):
        eng = _make_engine()
        hrv = tuple(float(i) for i in range(30, 40))
        cleaned = self._make_cleaned(hrv=hrv)
        m = eng.compute_core_metrics(cleaned)
        expected = sum(hrv) / len(hrv)
        assert m.mean_hrv == pytest.approx(expected, abs=1e-5)

    def test_4_3_load_mean_correct(self):
        eng = _make_engine()
        load = (200.0, 210.0, 190.0, 205.0, 215.0, 195.0, 208.0, 202.0, 198.0, 207.0)
        cleaned = self._make_cleaned(load=load)
        m = eng.compute_core_metrics(cleaned)
        expected = sum(load) / len(load)
        assert m.load_mean == pytest.approx(expected, abs=1e-5)

    def test_4_4_hr_std_correct(self):
        eng = _make_engine()
        hr = (70.0, 80.0, 60.0, 75.0, 65.0, 85.0, 72.0, 78.0, 68.0, 77.0)
        cleaned = self._make_cleaned(hr=hr)
        m = eng.compute_core_metrics(cleaned)
        mean = sum(hr) / len(hr)
        expected_std = math.sqrt(sum((x - mean) ** 2 for x in hr) / len(hr))
        assert m.hr_std == pytest.approx(expected_std, abs=1e-5)

    def test_4_5_hrv_std_correct(self):
        eng = _make_engine()
        hrv = (40.0, 50.0, 30.0, 45.0, 35.0, 55.0, 42.0, 48.0, 38.0, 47.0)
        cleaned = self._make_cleaned(hrv=hrv)
        m = eng.compute_core_metrics(cleaned)
        mean = sum(hrv) / len(hrv)
        expected_std = math.sqrt(sum((x - mean) ** 2 for x in hrv) / len(hrv))
        assert m.hrv_std == pytest.approx(expected_std, abs=1e-5)

    def test_4_6_single_element_std_is_zero(self):
        # Extend to 10 but all same value
        eng = _make_engine()
        cleaned = CleanedSignals(
            hr=(75.0,) * 10,
            hrv=(40.0,) * 10,
            load=(100.0,) * 10,
        )
        m = eng.compute_core_metrics(cleaned)
        assert m.hr_std == pytest.approx(0.0)

    def test_4_7_constant_channel_std_is_zero(self):
        eng = _make_engine()
        cleaned = CleanedSignals(
            hr=(72.0,) * 10,
            hrv=(45.0,) * 10,
            load=(150.0,) * 10,
        )
        m = eng.compute_core_metrics(cleaned)
        assert m.hr_std == pytest.approx(0.0)
        assert m.hrv_std == pytest.approx(0.0)

    def test_4_8_all_metrics_finite(self):
        eng = _make_engine()
        cleaned = self._make_cleaned()
        m = eng.compute_core_metrics(cleaned)
        for field_name in ("mean_hr", "mean_hrv", "load_mean", "hr_std", "hrv_std"):
            val = getattr(m, field_name)
            assert math.isfinite(val), f"{field_name} is not finite: {val}"

    def test_4_9_to_dict_keys(self):
        eng = _make_engine()
        cleaned = self._make_cleaned()
        m = eng.compute_core_metrics(cleaned)
        d = m.to_dict()
        assert set(d.keys()) == {"mean_hr", "mean_hrv", "load_mean", "hr_std", "hrv_std"}

    def test_4_10_rounding_applied(self):
        eng = BiometricEngine(BiometricConfig(output_precision=2))
        hr = (70.123456789,) * 10
        cleaned = CleanedSignals(hr=hr, hrv=(40.0,) * 10, load=(100.0,) * 10)
        m = eng.compute_core_metrics(cleaned)
        # 2-decimal precision: value should be 70.12
        assert m.mean_hr == pytest.approx(70.12, abs=0.005)
        # Ensure no more than 2 decimal places in the float representation
        s = f"{m.mean_hr:.10f}".rstrip("0")
        decimal_places = len(s.split(".")[1]) if "." in s else 0
        assert decimal_places <= 2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — process() integration
# ─────────────────────────────────────────────────────────────────────────────

class TestProcess:

    def test_5_1_valid_input_status_ok(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert result["status"] == "ok"

    def test_5_2_valid_input_all_five_metrics_present(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert result["metrics"] is not None
        assert set(result["metrics"].keys()) == {
            "mean_hr", "mean_hrv", "load_mean", "hr_std", "hrv_std",
            "fatigue_index",     # added Phase 7.3.1
            "drift_score",       # added Phase 7.4.0
            "injury_risk",       # added Phase 7.4.0
            "anomaly_score",     # added Phase 7.6.0
            "training_state",    # added Phase 7.7.0
            "recommended_load",  # added Phase 7.8.0
            "training_advice",   # added Phase 7.9.0
        }
        # data_features lives at the top level, not inside metrics
        assert "data_features" in result                    # added Phase 7.10.0
        assert result["data_features"] == {}                # no data_engine → always {}

    def test_5_3_engine_name_correct(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert result["engine"] == ENGINE_NAME

    def test_5_4_version_correct(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert result["version"] == ENGINE_VERSION

    def test_5_5_invalid_input_status_error(self):
        eng = _make_engine()
        result = eng.process({"hr": [], "hrv": [], "load": []})
        assert result["status"] == "error"

    def test_5_6_invalid_input_metrics_is_none(self):
        eng = _make_engine()
        result = eng.process("not a dict")
        assert result["metrics"] is None

    def test_5_7_invalid_input_error_key_present(self):
        eng = _make_engine()
        result = eng.process({"hr": [1] * 5, "hrv": [1] * 10, "load": [1] * 10})
        assert result["status"] == "error"
        assert "error" in result

    def test_5_8_process_never_raises(self):
        eng = _make_engine()
        # Throw the most pathological inputs conceivable.
        bad_inputs = [
            None,
            42,
            [],
            {},
            {"hr": None, "hrv": None, "load": None},
            {"hr": [float("nan")] * 10, "hrv": [1] * 10, "load": [1] * 10},
            {"hr": [1] * 10, "hrv": [float("inf")] * 10, "load": [1] * 10},
        ]
        for bad in bad_inputs:
            try:
                result = eng.process(bad)
                assert result["status"] in ("ok", "error")
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"process() raised {type(exc).__name__}: {exc} for input {bad!r}")

    def test_5_9_output_is_json_serialisable(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input(n=20))
        try:
            json.dumps(result)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Output not JSON-serialisable: {exc}")

    def test_5_10_output_metrics_are_finite_floats(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input(n=20))
        assert result["status"] == "ok"
        # top-level data_features must be present and be a dict
        assert isinstance(result["data_features"], dict)
        for key, val in result["metrics"].items():
            if key == "training_state":
                assert isinstance(val, str), f"{key} should be str, got {type(val)}"
            elif key == "training_advice":
                assert isinstance(val, dict), f"{key} should be dict, got {type(val)}"
                assert isinstance(val["state"], str)
                assert isinstance(val["recommended_load"], float)
                assert math.isfinite(val["recommended_load"])
            else:
                assert isinstance(val, float), f"{key} is not float: {type(val)}"
                assert math.isfinite(val), f"{key} is not finite: {val}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_6_1_same_input_same_output(self):
        eng = _make_engine()
        raw = _make_valid_input(n=25)
        r1 = eng.process(raw)
        r2 = eng.process(raw)
        assert r1 == r2

    def test_6_2_different_instances_same_output(self):
        raw = _make_valid_input(n=25)
        r1 = BiometricEngine().process(raw)
        r2 = BiometricEngine().process(raw)
        assert r1 == r2

    def test_6_3_100_calls_identical(self):
        eng = _make_engine()
        raw = _make_valid_input(n=50)
        first = eng.process(raw)
        for _ in range(99):
            assert eng.process(raw) == first


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_7_1_exactly_10_samples(self):
        eng = _make_engine()
        raw = _make_valid_input(n=10)
        result = eng.process(raw)
        assert result["status"] == "ok"

    def test_7_2_large_channel(self):
        eng = _make_engine()
        n = 10_000
        raw = {
            "hr":   [float(60 + (i % 40)) for i in range(n)],
            "hrv":  [float(40 + (i % 20)) for i in range(n)],
            "load": [float(100 + (i % 200)) for i in range(n)],
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        assert all(math.isfinite(v) for v in result["metrics"].values() if isinstance(v, float))
        eng = _make_engine()
        raw = {
            "hr":   [0.0] * 10,
            "hrv":  [0.0] * 10,
            "load": [0.0] * 10,
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        assert result["metrics"]["mean_hr"] == pytest.approx(0.0)
        assert result["metrics"]["hr_std"] == pytest.approx(0.0)

    def test_7_4_negative_load_values(self):
        eng = _make_engine()
        raw = {
            "hr":   [70.0] * 10,
            "hrv":  [40.0] * 10,
            "load": [float(-i) for i in range(10)],  # 0, -1, ..., -9
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        # mean_load should be -(0+1+...+9)/10 = -4.5
        assert result["metrics"]["load_mean"] == pytest.approx(-4.5, abs=1e-5)

    def test_7_5_very_large_values(self):
        eng = _make_engine()
        raw = {
            "hr":   [1e6 + i for i in range(10)],
            "hrv":  [1e6 + i for i in range(10)],
            "load": [1e6 + i for i in range(10)],
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        assert all(math.isfinite(v) for v in result["metrics"].values() if isinstance(v, float))

    def test_7_6_very_small_positive_values(self):
        eng = _make_engine()
        raw = {
            "hr":   [1e-9 + i * 1e-11 for i in range(10)],
            "hrv":  [1e-9 + i * 1e-11 for i in range(10)],
            "load": [1e-9 + i * 1e-11 for i in range(10)],
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        assert all(math.isfinite(v) for v in result["metrics"].values() if isinstance(v, float))

    def test_7_7_mixed_sign_values(self):
        eng = _make_engine()
        raw = {
            "hr":   [float(i - 5) for i in range(10)],   # -5 … 4
            "hrv":  [float(i - 5) for i in range(10)],
            "load": [float(i - 5) for i in range(10)],
        }
        result = eng.process(raw)
        assert result["status"] == "ok"
        # mean of -5,-4,...,4 = -0.5
        assert result["metrics"]["mean_hr"] == pytest.approx(-0.5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — helper unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_safe_mean_single_element(self):
        assert _safe_mean((42.0,)) == pytest.approx(42.0)

    def test_safe_mean_multiple(self):
        assert _safe_mean((1.0, 2.0, 3.0, 4.0)) == pytest.approx(2.5)

    def test_safe_pstdev_identical_values(self):
        assert _safe_pstdev((5.0,) * 10) == pytest.approx(0.0)

    def test_safe_pstdev_known_result(self):
        # pstdev([2, 4, 4, 4, 5, 5, 7, 9]) = 2.0  (textbook example)
        vals = (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        assert _safe_pstdev(vals) == pytest.approx(2.0)

    def test_min_max_normalize_known(self):
        vals = (0.0, 5.0, 10.0)
        normed = _min_max_normalize(vals)
        assert normed == pytest.approx((0.0, 0.5, 1.0))

    def test_min_max_normalize_constant(self):
        vals = (7.0,) * 5
        normed = _min_max_normalize(vals)
        assert all(v == 0.0 for v in normed)



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — fatigue index (Phase 7.3.1)
# ─────────────────────────────────────────────────────────────────────────────

class TestFatigueIndex:
    """
    Tests for compute_fatigue_index() and its integration into process().

    Normalised-signal construction note
    ------------------------------------
    min-max of a constant channel → all-zeros regardless of magnitude.
    To steer the normalised mean we need asymmetric distributions:

        Low  pattern : [lo]*9 + [hi]   → norm mean ≈ 0.1
        High pattern : [lo]  + [hi]*9  → norm mean ≈ 0.9
    """

    def _c(self, v: float, n: int = 10) -> list[float]:
        """Constant channel of length n."""
        return [v] * n

    # ── 9.1  present in output ───────────────────────────────────────────────

    def test_9_1_fatigue_index_present(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert "fatigue_index" in result["metrics"]

    # ── 9.2  type ────────────────────────────────────────────────────────────

    def test_9_2_fatigue_index_is_float(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert isinstance(result["metrics"]["fatigue_index"], float)

    # ── 9.3  bounded [0, 1] ──────────────────────────────────────────────────

    def test_9_3_fatigue_index_in_unit_interval(self):
        eng = _make_engine()
        for n in (10, 25, 100):
            fi = eng.process(_make_valid_input(n=n))["metrics"]["fatigue_index"]
            assert 0.0 <= fi <= 1.0, f"fatigue_index={fi} out of [0,1] for n={n}"

    # ── 9.4  finite ──────────────────────────────────────────────────────────

    def test_9_4_fatigue_index_finite(self):
        eng = _make_engine()
        fi = eng.process(_make_valid_input(n=20))["metrics"]["fatigue_index"]
        assert math.isfinite(fi)

    # ── 9.5  deterministic: 100 identical calls ───────────────────────────────

    def test_9_5_deterministic_100_calls(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["fatigue_index"]
        for _ in range(99):
            assert eng.process(raw)["metrics"]["fatigue_index"] == first

    # ── 9.6  deterministic: two engine instances ──────────────────────────────

    def test_9_6_deterministic_two_instances(self):
        raw = _make_valid_input(n=30)
        fi1 = BiometricEngine().process(raw)["metrics"]["fatigue_index"]
        fi2 = BiometricEngine().process(raw)["metrics"]["fatigue_index"]
        assert fi1 == fi2

    # ── 9.7  increases with higher normalised load ────────────────────────────

    def test_9_7_higher_load_raises_fatigue(self):
        """W1 (load) is positive: more load → higher fatigue."""
        eng = _make_engine()
        low_load  = {"hr": self._c(70.0), "hrv": self._c(40.0),
                     "load": [10.0]*9 + [20.0]}   # norm mean ≈ 0.1
        high_load = {"hr": self._c(70.0), "hrv": self._c(40.0),
                     "load": [10.0] + [20.0]*9}   # norm mean ≈ 0.9
        fi_low  = eng.process(low_load)["metrics"]["fatigue_index"]
        fi_high = eng.process(high_load)["metrics"]["fatigue_index"]
        assert fi_high > fi_low, f"fi_high={fi_high:.6f} should exceed fi_low={fi_low:.6f}"

    # ── 9.8  high HRV suppresses fatigue (inverted W3) ───────────────────────

    def test_9_8_high_hrv_suppresses_fatigue(self):
        """W3 is subtracted: higher HRV mean → lower fatigue."""
        eng = _make_engine()
        low_hrv  = {"hr": self._c(70.0), "hrv": [40.0]*9 + [50.0],
                    "load": self._c(100.0)}
        high_hrv = {"hr": self._c(70.0), "hrv": [40.0] + [50.0]*9,
                    "load": self._c(100.0)}
        fi_low_hrv  = eng.process(low_hrv)["metrics"]["fatigue_index"]
        fi_high_hrv = eng.process(high_hrv)["metrics"]["fatigue_index"]
        assert fi_low_hrv > fi_high_hrv, \
            f"fi_low_hrv={fi_low_hrv:.6f} should exceed fi_high_hrv={fi_high_hrv:.6f}"

    # ── 9.9  high HR std raises fatigue (positive W2) ────────────────────────

    def test_9_9_high_hr_std_raises_fatigue(self):
        """W2 is positive: more variable HR → higher fatigue."""
        eng = _make_engine()
        stable   = {"hr": self._c(70.0),
                    "hrv": self._c(40.0), "load": self._c(100.0)}
        variable = {"hr": [50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,88.0,90.0],
                    "hrv": self._c(40.0), "load": self._c(100.0)}
        fi_stable   = eng.process(stable)["metrics"]["fatigue_index"]
        fi_variable = eng.process(variable)["metrics"]["fatigue_index"]
        assert fi_variable > fi_stable, \
            f"fi_variable={fi_variable:.6f} should exceed fi_stable={fi_stable:.6f}"

    # ── 9.10  formula cross-check ─────────────────────────────────────────────

    def test_9_10_formula_matches_manual_calculation(self):
        """Manual computation of fatigue_raw → sigmoid must match engine."""
        eng = _make_engine()
        # HR  [60]*5 + [70]*5  → norm [0]*5+[1]*5 → mean=0.5, pstdev=0.5
        # HRV [40]*8+[40,60]   → norm [0]*8+[0,1] → mean=0.1
        # load [100]*9+[200]   → norm [0]*9+[1]   → mean=0.1
        raw_input = {
            "hr":   [60.0]*5 + [70.0]*5,
            "hrv":  [40.0]*8 + [40.0, 60.0],
            "load": [100.0]*9 + [200.0],
        }
        expected_raw = FATIGUE_W1*0.1 + FATIGUE_W2*0.5 - FATIGUE_W3*0.1
        expected_fi  = max(0.0, min(1.0, 1.0/(1.0+math.exp(-expected_raw))))

        actual = eng.process(raw_input)["metrics"]["fatigue_index"]
        assert abs(actual - expected_fi) < 1e-5, \
            f"Manual={expected_fi:.8f}  Engine={actual:.8f}"

    # ── 9.11  constant signals → sigmoid(0) = 0.5 ────────────────────────────

    def test_9_11_all_constant_signals_gives_half(self):
        """All channels constant → all normalised channels zero → sigmoid(0) = 0.5."""
        eng = _make_engine()
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        fi = eng.process(raw)["metrics"]["fatigue_index"]
        assert abs(fi - 0.5) < 1e-5, f"Expected 0.5 (sigmoid(0)), got {fi}"

    # ── 9.12  FatigueResult dataclass fields ─────────────────────────────────

    def test_9_12_fatigue_result_has_value_and_raw(self):
        eng = _make_engine()
        raw_input = _make_valid_input(n=20)
        eng.validate_schema(raw_input)
        cleaned    = eng.clean_data(raw_input)
        normalized = eng.normalize(cleaned)
        metrics    = eng.compute_core_metrics(cleaned)
        fatigue    = eng.compute_fatigue_index(metrics, normalized)
        assert isinstance(fatigue, FatigueResult)
        assert hasattr(fatigue, "value")
        assert hasattr(fatigue, "raw")
        assert 0.0 <= fatigue.value <= 1.0

    # ── 9.13  FatigueResult.raw is finite ────────────────────────────────────

    def test_9_13_fatigue_result_raw_is_finite(self):
        eng = _make_engine()
        raw_input = _make_valid_input(n=20)
        cleaned    = eng.clean_data(raw_input)
        normalized = eng.normalize(cleaned)
        metrics    = eng.compute_core_metrics(cleaned)
        fatigue    = eng.compute_fatigue_index(metrics, normalized)
        assert math.isfinite(fatigue.raw), f"FatigueResult.raw not finite: {fatigue.raw}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — drift score (Phase 7.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftScore:
    """
    Tests for compute_drift() and its integration into process().

    Key facts used throughout:
        fatigue_drift = abs(fatigue.value - 0.5)  →  range [0, 0.5]
        All-constant channels → fatigue=0.5, hr_std=0, hrv_std=0
          → drift_raw = 0.5*0 + 0.3*0 + 0.2*0 = 0.0
        hr_drift and hrv_drift are raw-scale → clamp is load-bearing.
    """

    def _c(self, v: float, n: int = 10) -> list[float]:
        return [v] * n

    def test_10_1_drift_score_present(self):
        result = _make_engine().process(_make_valid_input())
        assert "drift_score" in result["metrics"]

    def test_10_2_drift_score_is_float(self):
        result = _make_engine().process(_make_valid_input())
        assert isinstance(result["metrics"]["drift_score"], float)

    def test_10_3_drift_score_in_unit_interval(self):
        eng = _make_engine()
        for n in (10, 25, 100):
            ds = eng.process(_make_valid_input(n=n))["metrics"]["drift_score"]
            assert 0.0 <= ds <= 1.0, f"drift_score={ds} out of [0,1] for n={n}"

    def test_10_4_drift_score_finite(self):
        ds = _make_engine().process(_make_valid_input(n=20))["metrics"]["drift_score"]
        assert math.isfinite(ds)

    def test_10_5_deterministic_100_calls(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["drift_score"]
        for _ in range(99):
            assert eng.process(raw)["metrics"]["drift_score"] == first

    def test_10_6_deterministic_two_instances(self):
        raw = _make_valid_input(n=30)
        ds1 = BiometricEngine().process(raw)["metrics"]["drift_score"]
        ds2 = BiometricEngine().process(raw)["metrics"]["drift_score"]
        assert ds1 == ds2

    def test_10_7_constant_signals_zero_drift(self):
        """All channels constant → fatigue=0.5, stds=0 → drift=0.0."""
        eng = _make_engine()
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        ds = eng.process(raw)["metrics"]["drift_score"]
        assert abs(ds - 0.0) < 1e-5, f"Expected 0.0, got {ds}"

    def test_10_8_higher_hr_std_raises_drift(self):
        """DW_HR is positive: more variable HR → higher drift_score."""
        eng = _make_engine()
        stable   = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        variable = {"hr": [50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,88.0,90.0],
                    "hrv": self._c(40.0), "load": self._c(100.0)}
        ds_s = eng.process(stable)["metrics"]["drift_score"]
        ds_v = eng.process(variable)["metrics"]["drift_score"]
        assert ds_v > ds_s, f"ds_variable={ds_v:.6f} should exceed ds_stable={ds_s:.6f}"

    def test_10_9_clamp_enforced_for_large_hr_std(self):
        """Very large HR std → drift_raw >> 1 → clamped to exactly 1.0."""
        eng = _make_engine()
        raw = {"hr": [float(i * 100) for i in range(10)],
               "hrv": self._c(40.0), "load": self._c(100.0)}
        ds = eng.process(raw)["metrics"]["drift_score"]
        assert ds <= 1.0
        assert abs(ds - 1.0) < 1e-5, f"Expected 1.0 after clamp, got {ds}"

    def test_10_10_formula_cross_check(self):
        """Reconstruct drift manually from output metrics and compare."""
        eng = _make_engine()
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        result = eng.process(raw)
        fi   = result["metrics"]["fatigue_index"]
        hstd = result["metrics"]["hr_std"]
        vstd = result["metrics"]["hrv_std"]
        expected = max(0.0, min(1.0,
            DRIFT_W_FATIGUE * abs(fi - 0.5)
            + DRIFT_W_HR    * abs(hstd)
            + DRIFT_W_HRV   * abs(vstd)
        ))
        assert abs(result["metrics"]["drift_score"] - expected) < 1e-5

    def test_10_11_drift_result_structure(self):
        eng = _make_engine()
        cleaned = eng.clean_data(_make_valid_input(n=20))
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        assert isinstance(drift, DriftResult)
        assert hasattr(drift, "score") and hasattr(drift, "components")
        assert 0.0 <= drift.score <= 1.0
        assert len(drift.components) == 3
        assert all(math.isfinite(c) for c in drift.components)

    def test_10_12_drift_result_score_finite(self):
        eng = _make_engine()
        cleaned = eng.clean_data(_make_valid_input(n=20))
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        assert math.isfinite(drift.score)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — injury risk (Phase 7.4.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestInjuryRisk:
    """
    Tests for compute_injury_risk() and its integration into process().

    Key facts used throughout:
        fatigue_signal = fatigue.value              in [0, 1]
        drift_signal   = drift.score                in [0, 1]
        load_signal    = clamp(load_mean/1000, 0,1) in [0, 1]

        All-constant channels (load=100):
            fatigue=0.5, drift=0.0, load_signal=0.1
            risk_raw = 0.45*0.5 + 0.35*0.0 + 0.20*0.1 = 0.225 + 0.0 + 0.02 = 0.245

        Load >= 1000 → load_signal = 1.0 (clamped).
    """

    def _c(self, v: float, n: int = 10) -> list[float]:
        return [v] * n

    def test_11_1_injury_risk_present(self):
        result = _make_engine().process(_make_valid_input())
        assert "injury_risk" in result["metrics"]

    def test_11_2_injury_risk_is_float(self):
        result = _make_engine().process(_make_valid_input())
        assert isinstance(result["metrics"]["injury_risk"], float)

    def test_11_3_injury_risk_in_unit_interval(self):
        eng = _make_engine()
        for n in (10, 25, 100):
            ir = eng.process(_make_valid_input(n=n))["metrics"]["injury_risk"]
            assert 0.0 <= ir <= 1.0, f"injury_risk={ir} out of [0,1] for n={n}"

    def test_11_4_injury_risk_finite(self):
        ir = _make_engine().process(_make_valid_input(n=20))["metrics"]["injury_risk"]
        assert math.isfinite(ir)

    def test_11_5_deterministic_100_calls(self):
        eng = _make_engine()
        raw = _make_valid_input(n=30)
        first = eng.process(raw)["metrics"]["injury_risk"]
        for _ in range(99):
            assert eng.process(raw)["metrics"]["injury_risk"] == first

    def test_11_6_deterministic_two_instances(self):
        raw = _make_valid_input(n=30)
        ir1 = BiometricEngine().process(raw)["metrics"]["injury_risk"]
        ir2 = BiometricEngine().process(raw)["metrics"]["injury_risk"]
        assert ir1 == ir2

    def test_11_7_higher_fatigue_raises_risk(self):
        """RW_FATIGUE is positive: higher fatigue → higher injury risk."""
        eng = _make_engine()
        # Low fatigue: constant signals → fatigue=0.5 (sigmoid neutral)
        low_fatigue = {"hr": self._c(70.0), "hrv": self._c(40.0),
                       "load": self._c(100.0)}
        # High fatigue: high load skewed + low HRV mean
        high_fatigue = {"hr": self._c(70.0),
                        "hrv": [40.0] + [10.0]*9,    # low HRV → high fatigue
                        "load": [10.0] + [500.0]*9}   # high load → high fatigue
        ir_low  = eng.process(low_fatigue)["metrics"]["injury_risk"]
        ir_high = eng.process(high_fatigue)["metrics"]["injury_risk"]
        assert ir_high > ir_low, \
            f"ir_high={ir_high:.6f} should exceed ir_low={ir_low:.6f}"

    def test_11_8_higher_load_raises_risk(self):
        """RW_LOAD is positive: load_mean/1000 increases → injury risk increases."""
        eng = _make_engine()
        low_load  = {"hr": self._c(70.0), "hrv": self._c(40.0),
                     "load": self._c(100.0)}   # load_signal = 0.1
        high_load = {"hr": self._c(70.0), "hrv": self._c(40.0),
                     "load": self._c(900.0)}   # load_signal = 0.9
        ir_low  = eng.process(low_load)["metrics"]["injury_risk"]
        ir_high = eng.process(high_load)["metrics"]["injury_risk"]
        assert ir_high > ir_low, \
            f"ir_high={ir_high:.6f} should exceed ir_low={ir_low:.6f}"

    def test_11_9_load_signal_clamped_at_1000(self):
        """load_mean >= 1000 → load_signal = 1.0 (clamped, not unbounded)."""
        eng = _make_engine()
        # load_mean = 2000 → clamp(2000/1000, 0, 1) = 1.0
        raw_high = {"hr": self._c(70.0), "hrv": self._c(40.0),
                    "load": self._c(2000.0)}
        # load_mean = 1000 → clamp(1000/1000, 0, 1) = 1.0
        raw_ceil = {"hr": self._c(70.0), "hrv": self._c(40.0),
                    "load": self._c(1000.0)}
        ir_high = eng.process(raw_high)["metrics"]["injury_risk"]
        ir_ceil = eng.process(raw_ceil)["metrics"]["injury_risk"]
        assert abs(ir_high - ir_ceil) < 1e-5, \
            f"Both load>=1000 should yield same risk; got {ir_high:.6f} vs {ir_ceil:.6f}"

    def test_11_10_formula_cross_check(self):
        """
        Manually compute injury_risk from output metrics and compare.
        All-constant channels (load=100):
            fatigue_signal = 0.5
            drift_signal   = 0.0
            load_signal    = clamp(100/1000, 0, 1) = 0.1
            risk_raw = 0.45*0.5 + 0.35*0.0 + 0.20*0.1 = 0.245
        """
        eng = _make_engine()
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        result = eng.process(raw)
        fi = result["metrics"]["fatigue_index"]
        ds = result["metrics"]["drift_score"]
        lm = result["metrics"]["load_mean"]
        load_signal = max(0.0, min(1.0, lm / 1000.0))
        expected = max(0.0, min(1.0,
            RISK_W_FATIGUE * fi
            + RISK_W_DRIFT * ds
            + RISK_W_LOAD  * load_signal
        ))
        actual = result["metrics"]["injury_risk"]
        assert abs(actual - expected) < 1e-5, \
            f"Manual={expected:.8f}  Engine={actual:.8f}"

    def test_11_11_injury_risk_result_structure(self):
        eng = _make_engine()
        cleaned = eng.clean_data(_make_valid_input(n=20))
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        risk    = eng.compute_injury_risk(metrics, fatigue, drift)
        assert isinstance(risk, InjuryRiskResult)
        assert hasattr(risk, "score") and hasattr(risk, "signals")
        assert 0.0 <= risk.score <= 1.0
        assert len(risk.signals) == 3
        assert all(math.isfinite(s) for s in risk.signals)
        # All three signals must themselves be in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in risk.signals), \
            f"Signal out of [0,1]: {risk.signals}"

    def test_11_12_injury_risk_result_score_finite(self):
        eng = _make_engine()
        cleaned = eng.clean_data(_make_valid_input(n=20))
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        risk    = eng.compute_injury_risk(metrics, fatigue, drift)
        assert math.isfinite(risk.score)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — anomaly signals (Phase 7.5.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalySignals:
    """
    Tests for compute_anomaly_signals() and the AnomalySignals dataclass.

    compute_anomaly_signals() is NOT wired into process() and adds no key to
    the output dict.  All tests call it directly via the engine's public API.

    Anchor (all-constant channels: hr=70, hrv=40, load=100)
    --------------------------------------------------------
        hr_std = 0, hrv_std = 0  →  hr_spread = 0.0, hrv_spread = 0.0
        fatigue = 0.5 (sigmoid(0)), drift = 0.0
        → AnomalySignals(0.5, 0.0, 0.0, 0.0)
    """

    def _pipeline(self, raw: dict) -> tuple:
        """Run the engine pipeline up to drift; return (eng, metrics, fatigue, drift)."""
        eng     = _make_engine()
        cleaned = eng.clean_data(raw)
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        return eng, metrics, fatigue, drift

    def _c(self, v: float, n: int = 10) -> list[float]:
        return [v] * n

    # ── 12.1  returns AnomalySignals instance ────────────────────────────────

    def test_12_1_returns_anomaly_signals_instance(self):
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        eng, metrics, fatigue, drift = self._pipeline(raw)
        result = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert isinstance(result, AnomalySignals)

    # ── 12.2  all four fields present ────────────────────────────────────────

    def test_12_2_all_four_fields_present(self):
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        for field in ("fatigue_signal", "drift_signal", "hr_spread", "hrv_spread"):
            assert hasattr(sig, field), f"Missing field: {field}"

    # ── 12.3  all four fields finite ─────────────────────────────────────────

    def test_12_3_all_fields_finite(self):
        raw = _make_valid_input(n=20)
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert math.isfinite(sig.fatigue_signal), "fatigue_signal not finite"
        assert math.isfinite(sig.drift_signal),   "drift_signal not finite"
        assert math.isfinite(sig.hr_spread),      "hr_spread not finite"
        assert math.isfinite(sig.hrv_spread),     "hrv_spread not finite"

    # ── 12.4  all four fields in [0, 1] ──────────────────────────────────────

    def test_12_4_all_fields_in_unit_interval(self):
        for n in (10, 25, 100):
            raw = _make_valid_input(n=n)
            eng, metrics, fatigue, drift = self._pipeline(raw)
            sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
            for name, val in (
                ("fatigue_signal", sig.fatigue_signal),
                ("drift_signal",   sig.drift_signal),
                ("hr_spread",      sig.hr_spread),
                ("hrv_spread",     sig.hrv_spread),
            ):
                assert 0.0 <= val <= 1.0, f"n={n}: {name}={val} out of [0, 1]"

    # ── 12.5  deterministic: 100 identical calls ──────────────────────────────

    def test_12_5_deterministic_100_calls(self):
        raw = _make_valid_input(n=30)
        eng, metrics, fatigue, drift = self._pipeline(raw)
        first = eng.compute_anomaly_signals(metrics, fatigue, drift)
        for _ in range(99):
            assert eng.compute_anomaly_signals(metrics, fatigue, drift) == first

    # ── 12.6  deterministic: two engine instances ─────────────────────────────

    def test_12_6_deterministic_two_instances(self):
        raw = _make_valid_input(n=30)
        _, metrics, fatigue, drift = self._pipeline(raw)
        sig1 = BiometricEngine().compute_anomaly_signals(metrics, fatigue, drift)
        sig2 = BiometricEngine().compute_anomaly_signals(metrics, fatigue, drift)
        assert sig1 == sig2

    # ── 12.7  anchor: constant channels → exact known values ──────────────────

    def test_12_7_constant_channels_anchor(self):
        """
        All-constant channels:
            fatigue_signal = 0.5   (sigmoid(0) with zero std/load)
            drift_signal   = 0.0   (all stds = 0)
            hr_spread      = 0.0   (hr_std = 0)
            hrv_spread     = 0.0   (hrv_std = 0)
        """
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert abs(sig.fatigue_signal - 0.5) < 1e-5, f"Expected 0.5, got {sig.fatigue_signal}"
        assert abs(sig.drift_signal   - 0.0) < 1e-5, f"Expected 0.0, got {sig.drift_signal}"
        assert abs(sig.hr_spread      - 0.0) < 1e-5, f"Expected 0.0, got {sig.hr_spread}"
        assert abs(sig.hrv_spread     - 0.0) < 1e-5, f"Expected 0.0, got {sig.hrv_spread}"

    # ── 12.8  fatigue_signal is exact pass-through of fatigue.value ───────────

    def test_12_8_fatigue_signal_equals_fatigue_value(self):
        raw = _make_valid_input(n=20)
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert abs(sig.fatigue_signal - fatigue.value) < 1e-9, (
            f"fatigue_signal={sig.fatigue_signal} != fatigue.value={fatigue.value}"
        )

    # ── 12.9  drift_signal is exact pass-through of drift.score ──────────────

    def test_12_9_drift_signal_equals_drift_score(self):
        raw = _make_valid_input(n=20)
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert abs(sig.drift_signal - drift.score) < 1e-9, (
            f"drift_signal={sig.drift_signal} != drift.score={drift.score}"
        )

    # ── 12.10  hr_spread formula cross-check ─────────────────────────────────

    def test_12_10_hr_spread_formula(self):
        """hr_spread = clamp(hr_std / ANOMALY_HR_REF, 0, 1)."""
        raw = _make_valid_input(n=20)
        eng, metrics, fatigue, drift = self._pipeline(raw)
        sig      = eng.compute_anomaly_signals(metrics, fatigue, drift)
        expected = max(0.0, min(1.0, metrics.hr_std / ANOMALY_HR_REF))
        assert abs(sig.hr_spread - expected) < 1e-5, (
            f"hr_spread={sig.hr_spread:.8f}  expected={expected:.8f}"
        )

    # ── 12.11  hrv_spread clamped at ANOMALY_HRV_REF ─────────────────────────

    def test_12_11_hrv_spread_clamped_at_reference(self):
        """hrv_std >= ANOMALY_HRV_REF (100 ms) → hrv_spread = 1.0."""
        high_hrv = {
            "hr":   self._c(70.0),
            "hrv":  [0.0] * 10 + [500.0] * 10,   # std ≈ 250 ms >> 100 ms
            "load": self._c(100.0),
        }
        eng, metrics, fatigue, drift = self._pipeline(high_hrv)
        assert metrics.hrv_std > ANOMALY_HRV_REF, (
            f"Precondition: hrv_std={metrics.hrv_std} must exceed {ANOMALY_HRV_REF}"
        )
        sig = eng.compute_anomaly_signals(metrics, fatigue, drift)
        assert abs(sig.hrv_spread - 1.0) < 1e-5, (
            f"Expected hrv_spread=1.0 (clamped), got {sig.hrv_spread}"
        )

    # ── 12.12  inputs are not mutated ─────────────────────────────────────────

    def test_12_12_inputs_not_mutated(self):
        """compute_anomaly_signals must leave all arguments byte-for-byte identical."""
        raw = _make_valid_input(n=20)
        eng, metrics, fatigue, drift = self._pipeline(raw)

        # Snapshot state before the call
        m_snap = (metrics.mean_hr, metrics.mean_hrv, metrics.load_mean,
                  metrics.hr_std, metrics.hrv_std)
        f_snap = (fatigue.value, fatigue.raw)
        d_snap = (drift.score, drift.components)

        eng.compute_anomaly_signals(metrics, fatigue, drift)

        assert (metrics.mean_hr, metrics.mean_hrv, metrics.load_mean,
                metrics.hr_std, metrics.hrv_std) == m_snap, "metrics was mutated"
        assert (fatigue.value, fatigue.raw)       == f_snap, "fatigue was mutated"
        assert (drift.score,   drift.components)  == d_snap, "drift was mutated"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — anomaly scoring (Phase 7.6.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyScoring:
    """
    Tests for compute_anomaly() and the AnomalyResult dataclass.

    compute_anomaly() IS wired into process() and its score IS emitted in the
    output dict under the key "anomaly_score".

    Formula
    -------
    anomaly_raw = AW_F * fatigue_signal + AW_D * drift_signal
                + AW_H * hr_spread      + AW_V * hrv_spread
    anomaly_score = clamp(anomaly_raw, 0.0, 1.0)
    AW_F=0.40, AW_D=0.30, AW_H=0.20, AW_V=0.10  (sum to 1.0)

    Anchor (all-constant channels: hr=70, hrv=40, load=100)
    --------------------------------------------------------
        fatigue_signal = 0.5, drift_signal = 0.0
        hr_spread = 0.0,      hrv_spread   = 0.0
        anomaly_raw   = 0.40*0.5 + 0.30*0.0 + 0.20*0.0 + 0.10*0.0 = 0.20
        anomaly_score = 0.20
    """

    def _pipeline(self, raw: dict) -> tuple:
        """Run full pipeline; return (eng, metrics, fatigue, drift, signals)."""
        eng     = _make_engine()
        cleaned = eng.clean_data(raw)
        norm    = eng.normalize(cleaned)
        metrics = eng.compute_core_metrics(cleaned)
        fatigue = eng.compute_fatigue_index(metrics, norm)
        drift   = eng.compute_drift(metrics, fatigue)
        signals = eng.compute_anomaly_signals(metrics, fatigue, drift)
        return eng, metrics, fatigue, drift, signals

    def _c(self, v: float, n: int = 10) -> list[float]:
        return [v] * n

    # ── 13.1  returns AnomalyResult instance ─────────────────────────────────

    def test_13_1_returns_anomaly_result_instance(self):
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        eng, _, _, _, signals = self._pipeline(raw)
        result = eng.compute_anomaly(signals)
        assert isinstance(result, AnomalyResult)

    # ── 13.2  .score is float in [0, 1] ──────────────────────────────────────

    def test_13_2_score_in_unit_interval(self):
        for n in (10, 25, 100):
            raw = _make_valid_input(n=n)
            eng, _, _, _, signals = self._pipeline(raw)
            result = eng.compute_anomaly(signals)
            assert 0.0 <= result.score <= 1.0, f"n={n}: score={result.score}"

    # ── 13.3  .score is finite ────────────────────────────────────────────────

    def test_13_3_score_is_finite(self):
        raw = _make_valid_input(n=20)
        eng, _, _, _, signals = self._pipeline(raw)
        result = eng.compute_anomaly(signals)
        assert math.isfinite(result.score)

    # ── 13.4  .signals is AnomalySignals ─────────────────────────────────────

    def test_13_4_signals_field_is_anomaly_signals(self):
        raw = _make_valid_input(n=20)
        eng, _, _, _, signals = self._pipeline(raw)
        result = eng.compute_anomaly(signals)
        assert isinstance(result.signals, AnomalySignals)

    # ── 13.5  deterministic: 100 identical calls ──────────────────────────────

    def test_13_5_deterministic_100_calls(self):
        raw = _make_valid_input(n=30)
        eng, _, _, _, signals = self._pipeline(raw)
        first = eng.compute_anomaly(signals)
        for _ in range(99):
            assert eng.compute_anomaly(signals) == first

    # ── 13.6  deterministic: two engine instances ─────────────────────────────

    def test_13_6_deterministic_two_instances(self):
        raw = _make_valid_input(n=30)
        _, _, _, _, signals = self._pipeline(raw)
        r1 = BiometricEngine().compute_anomaly(signals)
        r2 = BiometricEngine().compute_anomaly(signals)
        assert r1 == r2

    # ── 13.7  anchor: constant channels → score = 0.20 ───────────────────────

    def test_13_7_constant_channels_anchor(self):
        """
        Constant channels:
            fatigue_signal = 0.5, drift_signal = 0.0, hr_spread = 0.0, hrv_spread = 0.0
            anomaly = 0.40*0.5 + 0.30*0.0 + 0.20*0.0 + 0.10*0.0 = 0.20
        """
        raw = {"hr": self._c(70.0), "hrv": self._c(40.0), "load": self._c(100.0)}
        eng, _, _, _, signals = self._pipeline(raw)
        result = eng.compute_anomaly(signals)
        assert abs(result.score - 0.20) < 1e-5, (
            f"Expected 0.20, got {result.score}"
        )

    # ── 13.8  formula cross-check: manual reconstruction ─────────────────────

    def test_13_8_formula_cross_check(self):
        """Manual weighted sum must match the engine output exactly."""
        raw = _make_valid_input(n=20)
        eng, _, _, _, signals = self._pipeline(raw)
        result = eng.compute_anomaly(signals)
        manual_raw = (
            ANOMALY_W_FATIGUE * signals.fatigue_signal
            + ANOMALY_W_DRIFT * signals.drift_signal
            + ANOMALY_W_HR    * signals.hr_spread
            + ANOMALY_W_HRV   * signals.hrv_spread
        )
        manual_score = max(0.0, min(1.0, manual_raw))
        assert abs(result.score - manual_score) < 1e-5, (
            f"engine={result.score:.8f}  manual={manual_score:.8f}"
        )

    # ── 13.9  weights sum to 1.0 ─────────────────────────────────────────────

    def test_13_9_weights_sum_to_one(self):
        total = ANOMALY_W_FATIGUE + ANOMALY_W_DRIFT + ANOMALY_W_HR + ANOMALY_W_HRV
        assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, expected 1.0"

    # ── 13.10  anomaly_score present in process() output ─────────────────────

    def test_13_10_anomaly_score_in_process_output(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        assert result["status"] == "ok"
        assert "anomaly_score" in result["metrics"]

    # ── 13.11  process() output has exactly 12 keys ──────────────────────────

    def test_13_11_process_output_has_twelve_keys(self):
        eng = _make_engine()
        result = eng.process(_make_valid_input())
        expected_keys = {
            "mean_hr", "mean_hrv", "load_mean", "hr_std", "hrv_std",
            "fatigue_index", "drift_score", "injury_risk", "anomaly_score",
            "training_state",    # added Phase 7.7.0
            "recommended_load",  # added Phase 7.8.0
            "training_advice",   # added Phase 7.9.0
        }
        # metrics stays at 12 keys; data_features is top-level (Phase 7.10.0)
        assert set(result["metrics"].keys()) == expected_keys
        assert "data_features" in result
        assert result["data_features"] == {}   # no data_engine injected

    # ── 13.12  anomaly_score in process() output is finite and in [0, 1] ──────

    def test_13_12_process_anomaly_score_finite_and_bounded(self):
        eng = _make_engine()
        for n in (10, 25, 50):
            result = eng.process(_make_valid_input(n=n))
            score = result["metrics"]["anomaly_score"]
            assert math.isfinite(score), f"n={n}: score not finite"
            assert 0.0 <= score <= 1.0,  f"n={n}: score={score} out of [0, 1]"


# =============================================================================
# SECTION 17 — DataEngine integration (Phase 7.10.0)
# =============================================================================
#
# Coverage map
# ─────────────
#   Optional dependency
#     17.1   BiometricEngine(data_engine=None) is the default — no error
#     17.2   BiometricEngine(data_engine=<mock>) stores the engine
#     17.3   data_features always present in output dict (top-level key)
#
#   No data_engine → always empty
#     17.4   data_features == {} when data_engine=None (default)
#     17.5   data_features == {} when data_engine=None (explicit)
#     17.6   data_features not inside metrics (metrics key count unchanged)
#
#   With data_engine → features populated
#     17.7   data_features == DataEngine.run()["features"] on success
#     17.8   data_engine.run() receives {"values": input["load"]}
#     17.9   data_features is a dict
#
#   Fault tolerance
#     17.10  DataEngine.run() raises → data_features == {}, pipeline ok
#     17.11  DataEngine.run() returns non-dict → data_features == {}
#     17.12  DataEngine.run() returns dict with no "features" key → {}
#     17.13  Biometric pipeline result identical regardless of DataEngine
#
#   Input immutability
#     17.14  Raw input dict not mutated by _extract_data_features
#     17.15  DataEngine receives a copy of load, not the original list
#
#   Error path
#     17.16  process() error still includes data_features key
#     17.17  process() error data_features == {} when no data_engine
#     17.18  process() error data_features == {} even with data_engine
#             that would succeed (schema error hits before pipeline)
#
#   JSON safety
#     17.19  Whole result (incl. data_features) is JSON-serialisable
#
#   Determinism
#     17.20  100 process() calls → identical data_features each time
# =============================================================================


class _MockDataEngine:
    """Minimal stub implementing the DataEngine contract."""
    def run(self, inp: dict) -> dict:
        return {
            "status":   "ok",
            "features": {"mean": 42.0, "std": 1.5, "count": len(inp["values"])},
        }


class _CapturingDataEngine:
    """Records every payload passed to run()."""
    def __init__(self):
        self.calls: list[dict] = []

    def run(self, inp: dict) -> dict:
        self.calls.append({"values": list(inp.get("values", []))})
        return {"features": {"captured": True}}


class _FaultyDataEngine:
    def run(self, inp: dict) -> dict:
        raise RuntimeError("simulated DataEngine failure")


class _NoFeaturesDataEngine:
    def run(self, inp: dict) -> dict:
        return {"status": "ok"}          # no "features" key


class _BadReturnDataEngine:
    def run(self, inp: dict) -> dict:  # type: ignore[return-value]
        return "this is not a dict"     # type: ignore[return-value]


def _make_engine_with_de() -> BiometricEngine:
    return BiometricEngine(BiometricConfig(), data_engine=_MockDataEngine())


class TestDataEngineIntegration:

    # ── 17.1  default construction still works ────────────────────────────────

    def test_17_1_no_data_engine_default(self):
        eng = BiometricEngine()
        result = eng.process(_make_valid_input())
        assert result["status"] == "ok"

    # ── 17.2  data_engine is stored ───────────────────────────────────────────

    def test_17_2_data_engine_stored(self):
        de = _MockDataEngine()
        eng = BiometricEngine(data_engine=de)
        assert eng._data_engine is de

    # ── 17.3  data_features key always present ────────────────────────────────

    def test_17_3_data_features_always_present(self):
        for eng in (BiometricEngine(), _make_engine_with_de()):
            result = eng.process(_make_valid_input())
            assert "data_features" in result, "data_features key missing"

    # ── 17.4  no data_engine → data_features == {} (default) ─────────────────

    def test_17_4_no_data_engine_default_empty(self):
        result = BiometricEngine().process(_make_valid_input())
        assert result["data_features"] == {}

    # ── 17.5  no data_engine → data_features == {} (explicit None) ───────────

    def test_17_5_explicit_none_data_engine_empty(self):
        result = BiometricEngine(data_engine=None).process(_make_valid_input())
        assert result["data_features"] == {}

    # ── 17.6  data_features is NOT inside metrics ─────────────────────────────

    def test_17_6_data_features_not_inside_metrics(self):
        for eng in (BiometricEngine(), _make_engine_with_de()):
            result = eng.process(_make_valid_input())
            assert "data_features" not in result["metrics"]
            assert len(result["metrics"]) == 12   # count unchanged

    # ── 17.7  data_features populated from DataEngine on success ─────────────

    def test_17_7_data_features_populated(self):
        result = _make_engine_with_de().process(_make_valid_input())
        assert isinstance(result["data_features"], dict)
        assert len(result["data_features"]) > 0

    # ── 17.8  data_engine.run() receives {"values": input["load"]} ────────────

    def test_17_8_data_engine_receives_load_channel(self):
        cap = _CapturingDataEngine()
        raw = _make_valid_input(n=15)
        BiometricEngine(data_engine=cap).process(raw)
        assert len(cap.calls) == 1
        assert cap.calls[0]["values"] == raw["load"]

    # ── 17.9  data_features is a dict ────────────────────────────────────────

    def test_17_9_data_features_is_dict(self):
        result = _make_engine_with_de().process(_make_valid_input())
        assert isinstance(result["data_features"], dict)

    # ── 17.10  faulty DataEngine: pipeline still ok, features == {} ───────────

    def test_17_10_faulty_data_engine_fallback(self):
        eng    = BiometricEngine(data_engine=_FaultyDataEngine())
        result = eng.process(_make_valid_input())
        assert result["status"] == "ok"
        assert result["data_features"] == {}

    # ── 17.11  DataEngine returns non-dict → features == {} ──────────────────

    def test_17_11_non_dict_return_fallback(self):
        result = BiometricEngine(data_engine=_BadReturnDataEngine()).process(
            _make_valid_input()
        )
        assert result["data_features"] == {}

    # ── 17.12  DataEngine returns dict with no "features" key → {} ───────────

    def test_17_12_missing_features_key_fallback(self):
        result = BiometricEngine(data_engine=_NoFeaturesDataEngine()).process(
            _make_valid_input()
        )
        assert result["data_features"] == {}

    # ── 17.13  biometric metrics identical with/without DataEngine ─────────────

    def test_17_13_biometric_metrics_unaffected_by_data_engine(self):
        raw     = _make_valid_input(n=20)
        without = BiometricEngine().process(raw)
        with_de = _make_engine_with_de().process(raw)
        # Every metric must be identical regardless of DataEngine presence
        assert without["metrics"] == with_de["metrics"]
        assert without["status"]  == with_de["status"]

    # ── 17.14  raw input dict not mutated ─────────────────────────────────────

    def test_17_14_raw_input_not_mutated(self):
        import copy
        raw  = _make_valid_input(n=12)
        snap = copy.deepcopy(raw)
        _make_engine_with_de().process(raw)
        assert raw == snap

    # ── 17.15  DataEngine receives a copy of load, not the original list ──────

    def test_17_15_data_engine_gets_copy_of_load(self):
        class _MutatingDE:
            """Tries to corrupt its input list."""
            def run(self, inp: dict) -> dict:
                inp["values"].clear()      # mutate the passed list
                return {"features": {}}

        raw  = _make_valid_input(n=10)
        orig_load = list(raw["load"])
        BiometricEngine(data_engine=_MutatingDE()).process(raw)
        assert raw["load"] == orig_load    # original untouched

    # ── 17.16  error path includes data_features key ──────────────────────────

    def test_17_16_error_output_has_data_features(self):
        result = BiometricEngine().process("not a dict")
        assert "data_features" in result
        assert result["status"] == "error"

    # ── 17.17  error + no data_engine → data_features == {} ──────────────────

    def test_17_17_error_no_data_engine_empty_features(self):
        result = BiometricEngine().process("not a dict")
        assert result["data_features"] == {}

    # ── 17.18  error + data_engine present → data_features == {} ─────────────
    #          (schema error fires before pipeline; data_engine gets no valid input)

    def test_17_18_error_with_data_engine_empty_features(self):
        result = _make_engine_with_de().process("not a dict")
        assert result["data_features"] == {}
        assert result["status"] == "error"

    # ── 17.19  full result is JSON-serialisable with DataEngine ───────────────

    def test_17_19_json_serialisable(self):
        result = _make_engine_with_de().process(_make_valid_input())
        json.dumps(result)   # must not raise

    # ── 17.20  determinism: 100 calls → identical data_features ──────────────

    def test_17_20_data_features_deterministic(self):
        eng = _make_engine_with_de()
        raw = _make_valid_input(n=20)
        first = eng.process(raw)["data_features"]
        for _ in range(99):
            assert eng.process(raw)["data_features"] == first


# ─────────────────────────────────────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)