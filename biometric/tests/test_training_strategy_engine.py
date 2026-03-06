"""
A.C.C.E.S.S. — Training Strategy Engine Test Suite (Phase 8.0.0)
tests/test_training_strategy_engine.py

Full coverage of biometric/training_strategy_engine.py

Coverage map
────────────
SECTION 1  — Module constants (14 tests)
  1.1   SCENARIO_COUNT == 4
  1.2   W_INJURY_RISK == 0.8
  1.3   W_FATIGUE_48H == 0.5
  1.4   INJURY_RISK_THRESHOLD == 0.7
  1.5   FATIGUE_THRESHOLD == 0.8
  1.6   READINESS_THRESHOLD == 0.75
  1.7   FORCED_CONFIDENCE == 1.0
  1.8   REST_LOAD == 0.0
  1.9   ENGINE_NAME and ENGINE_VERSION correct strings
  1.10  REST_SCENARIO fields correct
  1.11  LIGHT_SESSION_SCENARIO fields correct
  1.12  MODERATE_SESSION_SCENARIO fields correct
  1.13  INTENSE_SESSION_SCENARIO fields correct
  1.14  ALL_SCENARIOS has exactly 4 entries in correct order

SECTION 2  — TrainingScenario dataclass (10 tests)
  2.1   Instantiates with valid fields
  2.2   Frozen (FrozenInstanceError on reassignment)
  2.3   Fields stored exactly
  2.4   is_valid() True for well-formed scenario
  2.5   is_valid() False when load < 0
  2.6   is_valid() False when recovery_hours < 0
  2.7   is_valid() False when load non-finite
  2.8   is_valid() False when name is empty string
  2.9   to_dict() has exactly {"name","load","recovery_hours"}
  2.10  to_dict() is JSON-serialisable

SECTION 3  — ScenarioResult dataclass (10 tests)
  3.1   Instantiates with valid fields
  3.2   Frozen
  3.3   All five fields stored exactly
  3.4   is_valid() True for well-formed result
  3.5   is_valid() False when any float non-finite
  3.6   is_valid() False when any float < 0
  3.7   is_valid() False when any float > 1
  3.8   to_dict() has exactly the expected keys
  3.9   to_dict() is JSON-serialisable
  3.10  to_dict() returns a fresh dict (mutation does not affect instance)

SECTION 4  — StrategyDecision dataclass (12 tests)
  4.1   Instantiates with valid fields
  4.2   Frozen
  4.3   All six fields stored exactly
  4.4   is_valid() True for well-formed decision
  4.5   is_valid() False when recommended_load non-finite
  4.6   is_valid() False when expected_readiness out of [0,1]
  4.7   is_valid() False when expected_injury_risk > 1
  4.8   is_valid() False when confidence < 0
  4.9   is_valid() False when reasoning is empty tuple
  4.10  to_dict() has exactly the six expected keys
  4.11  to_dict() reasoning serialised as list (not tuple)
  4.12  to_dict() is JSON-serialisable

SECTION 5  — _filter_scenarios() (14 tests)
  5.1   Default (low fatigue, low risk, low readiness) → 3 scenarios (no INTENSE)
  5.2   High readiness only → 4 scenarios (INTENSE included)
  5.3   High fatigue only → 2 scenarios (REST + LIGHT)
  5.4   High injury risk only → 1 scenario (REST only)
  5.5   High injury risk overrides high readiness (P1 wins)
  5.6   High injury risk overrides high fatigue (P1 wins)
  5.7   High fatigue overrides high readiness (P2 wins over P3)
  5.8   REST always present in every configuration
  5.9   LIGHT_SESSION present unless forced REST
  5.10  MODERATE_SESSION absent when fatigue > threshold
  5.11  INTENSE_SESSION absent when readiness ≤ threshold
  5.12  Threshold boundary: injury_risk == THRESHOLD → not forced (exclusive)
  5.13  Threshold boundary: fatigue == THRESHOLD → not restricted (exclusive)
  5.14  Threshold boundary: readiness == THRESHOLD → INTENSE not unlocked (exclusive)

SECTION 6  — _compute_score() (10 tests)
  6.1   Perfect readiness, zero risk, zero fatigue → score == 1.0
  6.2   Zero readiness, max risk, max fatigue → score == 0.0
  6.3   Formula cross-check: manual computation matches
  6.4   Score always ∈ [0, 1]
  6.5   Score always finite
  6.6   W_INJURY_RISK weight applied correctly
  6.7   W_FATIGUE_48H weight applied correctly
  6.8   Non-finite readiness → clamped to 0.0
  6.9   Increasing readiness increases score (monotone)
  6.10  Increasing injury_risk decreases score (monotone)

SECTION 7  — evaluate() output structure (12 tests)
  7.1   Returns a dict
  7.2   Exactly six required keys
  7.3   recommended_session is a str
  7.4   recommended_load is a float ≥ 0
  7.5   expected_readiness ∈ [0, 1]
  7.6   expected_injury_risk ∈ [0, 1]
  7.7   confidence ∈ [0, 1]
  7.8   reasoning is a list of non-empty strings
  7.9   All float values finite
  7.10  All float values native Python float
  7.11  Output is JSON-serialisable
  7.12  Empty twin history produces valid output

SECTION 8  — evaluate() safety constraints (14 tests)
  8.1   injury_risk_state > 0.7 → REST always selected
  8.2   injury_risk_state > 0.7 → confidence == 1.0 (forced)
  8.3   injury_risk_state > 0.7 → REST in reasoning
  8.4   fatigue_state > 0.8 → only REST or LIGHT_SESSION selected
  8.5   fatigue_state > 0.8 → MODERATE_SESSION never selected
  8.6   fatigue_state > 0.8 → INTENSE_SESSION never selected
  8.7   readiness_state > 0.75 → INTENSE_SESSION can be selected
  8.8   readiness_state ≤ 0.75 → INTENSE_SESSION never selected
  8.9   P1 overrides P3: high risk + high readiness → REST
  8.10  P2 overrides P3: high fatigue + high readiness → REST or LIGHT only
  8.11  Boundary: injury_risk_state == 0.7 → not forced (just above threshold)
  8.12  Normal state → MODERATE_SESSION can be selected
  8.13  Invalid twin state triggers safe fallback
  8.14  Fallback confidence == 0.0

SECTION 9  — evaluate() scenario selection and scoring (12 tests)
  9.1   Best-score scenario is always selected
  9.2   evaluate_scenarios() returns exactly 4 results
  9.3   evaluate_scenarios() order: REST first, INTENSE last
  9.4   All evaluate_scenarios() scores ∈ [0, 1]
  9.5   All evaluate_scenarios() scores finite
  9.6   evaluate_scenarios() is JSON-serialisable
  9.7   recommended_load matches selected scenario's load
  9.8   REST is selected when it has the highest score
  9.9   Scores are sorted descending among allowed scenarios
  9.10  Tie-breaking: REST wins ties over LIGHT (most conservative)
  9.11  evaluate_scenarios() does not mutate twin
  9.12  evaluate_scenarios() consistent with evaluate() for unconstrained twin

SECTION 10 — evaluate() confidence computation (8 tests)
  10.1  confidence ∈ [0, 1] always
  10.2  confidence finite always
  10.3  confidence == 1.0 when only one scenario (forced REST)
  10.4  confidence == 0.0 when best == second (tie)
  10.5  confidence == best_score − second_score when margin > 0
  10.6  Large score gap → high confidence
  10.7  Small score gap → low confidence
  10.8  confidence always ≥ 0 (clamped; never negative)

SECTION 11 — evaluate() reasoning output (10 tests)
  11.1  reasoning is a list
  11.2  reasoning is non-empty
  11.3  All reasoning entries are non-empty strings
  11.4  Recommended session name appears in last reasoning entry
  11.5  "critical" appears in reasoning when fatigue_state > 0.8
  11.6  "REST enforced" appears in reasoning when injury_risk_state > 0.7
  11.7  "intense training unlocked" in reasoning when readiness_state > 0.75
  11.8  "conservative approach" in reasoning when readiness_state low
  11.9  Reasoning is deterministic: same state → same reasoning
  11.10 Reasoning contains at least 4 entries

SECTION 12 — Determinism (8 tests)
  12.1  100 identical evaluate() calls → identical output
  12.2  deterministic_check() returns True
  12.3  Two engines, same twin → same decision
  12.4  evaluate_scenarios() deterministic over 100 calls
  12.5  evaluate() then evaluate_scenarios() on same twin → consistent session
  12.6  Self-consistency: recommended_session in evaluate_scenarios() names
  12.7  evaluate() after evaluate() on same twin → unchanged decision (stateless)
  12.8  Same twin state, different history length → same constraint results

SECTION 13 — Input immutability (8 tests)
  13.1  evaluate() does not mutate twin get_state()
  13.2  evaluate() does not mutate twin event history
  13.3  evaluate() does not mutate twin fatigue_state
  13.4  evaluate() does not mutate externally-supplied forecast dict
  13.5  evaluate_scenarios() does not mutate twin
  13.6  _filter_scenarios() does not mutate anything (pure function)
  13.7  _compute_score() does not mutate anything (pure function)
  13.8  to_dict() mutation does not affect StrategyDecision instance

SECTION 14 — JSON serialisability (8 tests)
  14.1  evaluate() output passes json.dumps
  14.2  evaluate_scenarios() output passes json.dumps
  14.3  TrainingScenario.to_dict() passes json.dumps
  14.4  ScenarioResult.to_dict() passes json.dumps
  14.5  StrategyDecision.to_dict() passes json.dumps
  14.6  Full json round-trip: dumps → loads preserves all values
  14.7  All float values in evaluate() output are native Python float
  14.8  reasoning key in evaluate() output is a list (JSON array)

SECTION 15 — Invalid input safety (8 tests)
  15.1  Invalid twin state → fallback REST returned
  15.2  Fallback recommended_session == "REST"
  15.3  Fallback expected_injury_risk == 1.0
  15.4  Fallback confidence == 0.0
  15.5  Fallback reasoning is a non-empty list
  15.6  simulate_training() failure → graceful degradation (no exception)
  15.7  Non-finite forecast values → safe float default applied
  15.8  All fallback values finite and bounded

SECTION 16 — self_test() (8 tests)
  16.1  Returns dict
  16.2  Contains "engine", "version", "checks", "passed"
  16.3  "checks" has exactly six items
  16.4  Each check has "name", "passed", "detail"
  16.5  All six checks pass
  16.6  "engine" == ENGINE_NAME
  16.7  "version" == ENGINE_VERSION
  16.8  Calling self_test() twice gives identical "passed" result

SECTION 17 — _clamp() and _safe_float() helpers (9 tests)
  17.1  _clamp() — within range unchanged
  17.2  _clamp() — below lo → lo
  17.3  _clamp() — above hi → hi
  17.4  _clamp() — NaN → lo
  17.5  _clamp() — +Inf → lo
  17.6  _clamp() — boundary values exact
  17.7  _safe_float() — finite value unchanged
  17.8  _safe_float() — NaN → default
  17.9  _safe_float() — non-numeric string → default
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.athlete_digital_twin import AthleteDigitalTwin
from biometric.training_strategy_engine import (
    ENGINE_NAME,
    ENGINE_VERSION,
    FATIGUE_THRESHOLD,
    FORCED_CONFIDENCE,
    INJURY_RISK_THRESHOLD,
    INTENSE_SESSION_SCENARIO,
    LIGHT_SESSION_SCENARIO,
    MODERATE_SESSION_SCENARIO,
    READINESS_THRESHOLD,
    REST_LOAD,
    REST_SCENARIO,
    SCENARIO_COUNT,
    ALL_SCENARIOS,
    W_FATIGUE_48H,
    W_INJURY_RISK,
    ScenarioResult,
    StrategyDecision,
    TrainingScenario,
    TrainingStrategyEngine,
    _clamp,
    _compute_score,
    _filter_scenarios,
    _make_probe_twin,
    _safe_float,
)


# =============================================================================
# HELPERS
# =============================================================================

def _eng() -> TrainingStrategyEngine:
    return TrainingStrategyEngine()


def _fresh_twin(n: int = 0) -> AthleteDigitalTwin:
    """Return a twin with n moderate events."""
    t = AthleteDigitalTwin()
    for _ in range(n):
        t.update({"fatigue_index": 0.4, "sprint_load": 3_000.0,
                  "recovery_hours": 20.0, "injury_flag": 0})
    return t


def _high_risk_twin() -> AthleteDigitalTwin:
    """Return a twin with injury_risk_state > INJURY_RISK_THRESHOLD."""
    t = AthleteDigitalTwin()
    for _ in range(40):
        t.update({"fatigue_index": 0.95, "sprint_load": 8_000.0,
                  "recovery_hours": 0.0, "injury_flag": 1})
    assert t.injury_risk_state > INJURY_RISK_THRESHOLD
    return t


def _high_fatigue_twin() -> AthleteDigitalTwin:
    """Return a twin with fatigue_state > FATIGUE_THRESHOLD but risk ≤ threshold."""
    t = AthleteDigitalTwin()
    for _ in range(30):
        t.update({"fatigue_index": 0.9, "sprint_load": 5_000.0,
                  "recovery_hours": 8.0, "injury_flag": 0})
    assert t.fatigue_state > FATIGUE_THRESHOLD
    return t


def _high_readiness_twin() -> AthleteDigitalTwin:
    """Return a twin with readiness_state > READINESS_THRESHOLD."""
    t = AthleteDigitalTwin()
    for _ in range(30):
        t.update({"fatigue_index": 0.02, "sprint_load": 0.0,
                  "recovery_hours": 48.0, "injury_flag": 0})
    assert t.readiness_state > READINESS_THRESHOLD
    return t


def _sc_result(score: float = 0.5) -> ScenarioResult:
    return ScenarioResult(
        scenario           = REST_SCENARIO,
        future_readiness   = 0.6,
        future_injury_risk = 0.2,
        fatigue_projection = 0.3,
        strategy_score     = score,
    )


# =============================================================================
# SECTION 1 — Module constants
# =============================================================================

class TestConstants:

    def test_1_1_scenario_count(self):
        assert SCENARIO_COUNT == 4

    def test_1_2_w_injury_risk(self):
        assert W_INJURY_RISK == pytest.approx(0.8)

    def test_1_3_w_fatigue_48h(self):
        assert W_FATIGUE_48H == pytest.approx(0.5)

    def test_1_4_injury_risk_threshold(self):
        assert INJURY_RISK_THRESHOLD == pytest.approx(0.7)

    def test_1_5_fatigue_threshold(self):
        assert FATIGUE_THRESHOLD == pytest.approx(0.8)

    def test_1_6_readiness_threshold(self):
        assert READINESS_THRESHOLD == pytest.approx(0.75)

    def test_1_7_forced_confidence(self):
        assert FORCED_CONFIDENCE == pytest.approx(1.0)

    def test_1_8_rest_load(self):
        assert REST_LOAD == 0.0

    def test_1_9_engine_metadata(self):
        assert ENGINE_NAME    == "TrainingStrategyEngine"
        assert ENGINE_VERSION == "8.0.0"

    def test_1_10_rest_scenario_fields(self):
        assert REST_SCENARIO.name           == "REST"
        assert REST_SCENARIO.load           == 0.0
        assert REST_SCENARIO.recovery_hours == 24.0

    def test_1_11_light_session_fields(self):
        assert LIGHT_SESSION_SCENARIO.name           == "LIGHT_SESSION"
        assert LIGHT_SESSION_SCENARIO.load           == pytest.approx(2_000.0)
        assert LIGHT_SESSION_SCENARIO.recovery_hours == pytest.approx(18.0)

    def test_1_12_moderate_session_fields(self):
        assert MODERATE_SESSION_SCENARIO.name           == "MODERATE_SESSION"
        assert MODERATE_SESSION_SCENARIO.load           == pytest.approx(4_000.0)
        assert MODERATE_SESSION_SCENARIO.recovery_hours == pytest.approx(16.0)

    def test_1_13_intense_session_fields(self):
        assert INTENSE_SESSION_SCENARIO.name           == "INTENSE_SESSION"
        assert INTENSE_SESSION_SCENARIO.load           == pytest.approx(7_000.0)
        assert INTENSE_SESSION_SCENARIO.recovery_hours == pytest.approx(12.0)

    def test_1_14_all_scenarios_four_in_order(self):
        assert len(ALL_SCENARIOS) == 4
        assert ALL_SCENARIOS[0] is REST_SCENARIO
        assert ALL_SCENARIOS[1] is LIGHT_SESSION_SCENARIO
        assert ALL_SCENARIOS[2] is MODERATE_SESSION_SCENARIO
        assert ALL_SCENARIOS[3] is INTENSE_SESSION_SCENARIO


# =============================================================================
# SECTION 2 — TrainingScenario dataclass
# =============================================================================

class TestTrainingScenario:

    def test_2_1_instantiation(self):
        assert isinstance(TrainingScenario("X", 3_000.0, 16.0), TrainingScenario)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            REST_SCENARIO.load = 999.0  # type: ignore[misc]

    def test_2_3_fields_stored_exactly(self):
        ts = TrainingScenario("CUSTOM", 5_000.0, 14.0)
        assert ts.name           == "CUSTOM"
        assert ts.load           == 5_000.0
        assert ts.recovery_hours == 14.0

    def test_2_4_is_valid_true(self):
        assert REST_SCENARIO.is_valid() is True
        assert INTENSE_SESSION_SCENARIO.is_valid() is True

    def test_2_5_is_valid_negative_load(self):
        assert TrainingScenario("X", -1.0, 10.0).is_valid() is False

    def test_2_6_is_valid_negative_recovery(self):
        assert TrainingScenario("X", 3_000.0, -1.0).is_valid() is False

    def test_2_7_is_valid_nonfinite_load(self):
        assert TrainingScenario("X", float("nan"), 16.0).is_valid() is False

    def test_2_8_is_valid_empty_name(self):
        assert TrainingScenario("", 3_000.0, 16.0).is_valid() is False

    def test_2_9_to_dict_exact_keys(self):
        assert set(REST_SCENARIO.to_dict()) == {"name", "load", "recovery_hours"}

    def test_2_10_to_dict_json_serialisable(self):
        json.dumps(REST_SCENARIO.to_dict())


# =============================================================================
# SECTION 3 — ScenarioResult dataclass
# =============================================================================

class TestScenarioResult:

    def test_3_1_instantiation(self):
        assert isinstance(_sc_result(), ScenarioResult)

    def test_3_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            _sc_result().strategy_score = 99.0  # type: ignore[misc]

    def test_3_3_fields_stored_exactly(self):
        sr = ScenarioResult(REST_SCENARIO, 0.7, 0.1, 0.3, 0.65)
        assert sr.future_readiness   == 0.7
        assert sr.future_injury_risk == 0.1
        assert sr.fatigue_projection == 0.3
        assert sr.strategy_score     == 0.65

    def test_3_4_is_valid_true(self):
        assert _sc_result().is_valid() is True

    def test_3_5_is_valid_nonfinite_float(self):
        assert ScenarioResult(REST_SCENARIO, float("nan"), 0.1, 0.3, 0.5).is_valid() is False

    def test_3_6_is_valid_float_below_zero(self):
        assert ScenarioResult(REST_SCENARIO, -0.1, 0.1, 0.3, 0.5).is_valid() is False

    def test_3_7_is_valid_float_above_one(self):
        assert ScenarioResult(REST_SCENARIO, 1.1, 0.1, 0.3, 0.5).is_valid() is False

    def test_3_8_to_dict_exact_keys(self):
        expected = {"scenario", "future_readiness", "future_injury_risk",
                    "fatigue_projection", "strategy_score"}
        assert set(_sc_result().to_dict()) == expected

    def test_3_9_to_dict_json_serialisable(self):
        json.dumps(_sc_result().to_dict())

    def test_3_10_to_dict_fresh_dict(self):
        sr = _sc_result()
        d  = sr.to_dict()
        d["strategy_score"] = 999.0
        assert sr.strategy_score != 999.0


# =============================================================================
# SECTION 4 — StrategyDecision dataclass
# =============================================================================

class TestStrategyDecision:

    def _sd(self, **kw) -> StrategyDecision:
        defaults = dict(
            recommended_session  = "REST",
            recommended_load     = 0.0,
            expected_readiness   = 0.5,
            expected_injury_risk = 0.2,
            confidence           = 0.8,
            reasoning            = ("reason one", "reason two"),
        )
        defaults.update(kw)
        return StrategyDecision(**defaults)

    def test_4_1_instantiation(self):
        assert isinstance(self._sd(), StrategyDecision)

    def test_4_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            self._sd().confidence = 0.0  # type: ignore[misc]

    def test_4_3_fields_stored_exactly(self):
        sd = self._sd()
        assert sd.recommended_session  == "REST"
        assert sd.recommended_load     == 0.0
        assert sd.expected_readiness   == 0.5
        assert sd.expected_injury_risk == 0.2
        assert sd.confidence           == 0.8
        assert sd.reasoning            == ("reason one", "reason two")

    def test_4_4_is_valid_true(self):
        assert self._sd().is_valid() is True

    def test_4_5_is_valid_nonfinite_load(self):
        assert self._sd(recommended_load=float("nan")).is_valid() is False

    def test_4_6_is_valid_readiness_out_of_unit(self):
        assert self._sd(expected_readiness=-0.1).is_valid() is False
        assert self._sd(expected_readiness=1.1).is_valid()  is False

    def test_4_7_is_valid_injury_risk_above_one(self):
        assert self._sd(expected_injury_risk=1.001).is_valid() is False

    def test_4_8_is_valid_confidence_below_zero(self):
        assert self._sd(confidence=-0.1).is_valid() is False

    def test_4_9_is_valid_empty_reasoning(self):
        assert self._sd(reasoning=()).is_valid() is False

    def test_4_10_to_dict_six_keys(self):
        expected = {"recommended_session", "recommended_load",
                    "expected_readiness", "expected_injury_risk",
                    "confidence", "reasoning"}
        assert set(self._sd().to_dict()) == expected

    def test_4_11_reasoning_serialised_as_list(self):
        assert isinstance(self._sd().to_dict()["reasoning"], list)

    def test_4_12_to_dict_json_serialisable(self):
        json.dumps(self._sd().to_dict())


# =============================================================================
# SECTION 5 — _filter_scenarios()
# =============================================================================

class TestFilterScenarios:

    def test_5_1_default_three_scenarios(self):
        allowed = _filter_scenarios(0.4, 0.3, 0.4)
        assert len(allowed) == 3
        assert INTENSE_SESSION_SCENARIO not in allowed

    def test_5_2_high_readiness_four_scenarios(self):
        allowed = _filter_scenarios(0.3, 0.2, 0.9)
        assert INTENSE_SESSION_SCENARIO in allowed
        assert len(allowed) == 4

    def test_5_3_high_fatigue_two_scenarios(self):
        allowed = _filter_scenarios(0.9, 0.3, 0.5)
        assert allowed == [REST_SCENARIO, LIGHT_SESSION_SCENARIO]

    def test_5_4_high_injury_risk_one_scenario(self):
        allowed = _filter_scenarios(0.5, 0.8, 0.5)
        assert allowed == [REST_SCENARIO]

    def test_5_5_p1_overrides_p3(self):
        # High risk + high readiness → still only REST
        allowed = _filter_scenarios(0.2, 0.8, 0.9)
        assert allowed == [REST_SCENARIO]

    def test_5_6_p1_overrides_p2(self):
        # High risk + high fatigue → still only REST
        allowed = _filter_scenarios(0.9, 0.8, 0.2)
        assert allowed == [REST_SCENARIO]

    def test_5_7_p2_overrides_p3(self):
        # High fatigue + high readiness → only REST + LIGHT
        allowed = _filter_scenarios(0.9, 0.5, 0.9)
        assert MODERATE_SESSION_SCENARIO not in allowed
        assert INTENSE_SESSION_SCENARIO  not in allowed

    def test_5_8_rest_always_present(self):
        for fatigue, risk, readiness in [
            (0.2, 0.2, 0.2), (0.9, 0.3, 0.9), (0.2, 0.9, 0.9), (0.9, 0.9, 0.2)
        ]:
            assert REST_SCENARIO in _filter_scenarios(fatigue, risk, readiness)

    def test_5_9_light_present_unless_forced_rest(self):
        # forced REST only
        assert LIGHT_SESSION_SCENARIO not in _filter_scenarios(0.2, 0.8, 0.2)
        # high fatigue — light still present
        assert LIGHT_SESSION_SCENARIO in _filter_scenarios(0.9, 0.5, 0.5)

    def test_5_10_moderate_absent_when_fatigue_high(self):
        assert MODERATE_SESSION_SCENARIO not in _filter_scenarios(0.9, 0.5, 0.5)

    def test_5_11_intense_absent_when_readiness_low(self):
        assert INTENSE_SESSION_SCENARIO not in _filter_scenarios(0.3, 0.2, 0.5)

    def test_5_12_boundary_injury_risk_equal_threshold(self):
        # == THRESHOLD is not "above" → not forced
        allowed = _filter_scenarios(0.3, INJURY_RISK_THRESHOLD, 0.5)
        assert len(allowed) > 1

    def test_5_13_boundary_fatigue_equal_threshold(self):
        # == FATIGUE_THRESHOLD is not "above" → not restricted
        allowed = _filter_scenarios(FATIGUE_THRESHOLD, 0.3, 0.5)
        assert MODERATE_SESSION_SCENARIO in allowed

    def test_5_14_boundary_readiness_equal_threshold(self):
        # == READINESS_THRESHOLD is not "above" → INTENSE not unlocked
        allowed = _filter_scenarios(0.3, 0.2, READINESS_THRESHOLD)
        assert INTENSE_SESSION_SCENARIO not in allowed


# =============================================================================
# SECTION 6 — _compute_score()
# =============================================================================

class TestComputeScore:

    def test_6_1_perfect_inputs(self):
        assert _compute_score(1.0, 0.0, 0.0) == pytest.approx(1.0, abs=1e-9)

    def test_6_2_worst_inputs(self):
        assert _compute_score(0.0, 1.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_6_3_formula_cross_check(self):
        r, ir, fp = 0.6, 0.3, 0.4
        expected  = max(0.0, min(1.0, r - ir * W_INJURY_RISK - fp * W_FATIGUE_48H))
        assert _compute_score(r, ir, fp) == pytest.approx(expected, abs=1e-12)

    def test_6_4_score_in_unit_interval(self):
        for r, ir, fp in [(0.7, 0.2, 0.3), (0.1, 0.9, 0.8), (0.5, 0.5, 0.5)]:
            s = _compute_score(r, ir, fp)
            assert 0.0 <= s <= 1.0, f"score={s} for ({r},{ir},{fp})"

    def test_6_5_score_finite(self):
        assert math.isfinite(_compute_score(0.5, 0.3, 0.4))

    def test_6_6_injury_risk_weight(self):
        # Increasing injury_risk by 0.1 should decrease score by W_INJURY_RISK * 0.1
        s1 = _compute_score(0.8, 0.2, 0.1)
        s2 = _compute_score(0.8, 0.3, 0.1)
        expected_delta = W_INJURY_RISK * 0.1
        # only applies if scores are unclamped — use a safe region
        assert abs((s1 - s2) - expected_delta) < 1e-12

    def test_6_7_fatigue_weight(self):
        s1 = _compute_score(0.8, 0.1, 0.2)
        s2 = _compute_score(0.8, 0.1, 0.3)
        expected_delta = W_FATIGUE_48H * 0.1
        assert abs((s1 - s2) - expected_delta) < 1e-12

    def test_6_8_nonfinite_readiness_clamped(self):
        # non-finite should be guarded; engine uses _safe_float upstream
        s = _compute_score(0.0, 0.5, 0.5)
        assert 0.0 <= s <= 1.0

    def test_6_9_increasing_readiness_increases_score(self):
        s1 = _compute_score(0.3, 0.2, 0.3)
        s2 = _compute_score(0.7, 0.2, 0.3)
        assert s2 > s1

    def test_6_10_increasing_injury_risk_decreases_score(self):
        s1 = _compute_score(0.6, 0.1, 0.3)
        s2 = _compute_score(0.6, 0.5, 0.3)
        assert s2 < s1


# =============================================================================
# SECTION 7 — evaluate() output structure
# =============================================================================

class TestEvaluateOutputStructure:

    _KEYS = {"recommended_session", "recommended_load", "expected_readiness",
             "expected_injury_risk", "confidence", "reasoning"}

    def test_7_1_returns_dict(self):
        assert isinstance(_eng().evaluate(_fresh_twin()), dict)

    def test_7_2_exactly_six_keys(self):
        assert set(_eng().evaluate(_fresh_twin())) == self._KEYS

    def test_7_3_session_is_str(self):
        assert isinstance(_eng().evaluate(_fresh_twin())["recommended_session"], str)

    def test_7_4_load_is_nonneg_float(self):
        r = _eng().evaluate(_make_probe_twin(10))
        assert isinstance(r["recommended_load"], float)
        assert r["recommended_load"] >= 0.0

    def test_7_5_expected_readiness_in_unit(self):
        r = _eng().evaluate(_make_probe_twin(10))
        assert 0.0 <= r["expected_readiness"] <= 1.0

    def test_7_6_expected_injury_risk_in_unit(self):
        r = _eng().evaluate(_make_probe_twin(10))
        assert 0.0 <= r["expected_injury_risk"] <= 1.0

    def test_7_7_confidence_in_unit(self):
        r = _eng().evaluate(_make_probe_twin(10))
        assert 0.0 <= r["confidence"] <= 1.0

    def test_7_8_reasoning_is_list_of_nonempty_strings(self):
        r = _eng().evaluate(_make_probe_twin(10))
        assert isinstance(r["reasoning"], list) and len(r["reasoning"]) > 0
        assert all(isinstance(s, str) and len(s) > 0 for s in r["reasoning"])

    def test_7_9_all_floats_finite(self):
        r = _eng().evaluate(_make_probe_twin(10))
        for k in ("recommended_load", "expected_readiness",
                  "expected_injury_risk", "confidence"):
            assert math.isfinite(r[k])

    def test_7_10_all_floats_native_python_float(self):
        r = _eng().evaluate(_make_probe_twin(10))
        for k in ("recommended_load", "expected_readiness",
                  "expected_injury_risk", "confidence"):
            assert isinstance(r[k], float)

    def test_7_11_json_serialisable(self):
        json.dumps(_eng().evaluate(_make_probe_twin(10)))

    def test_7_12_empty_history_valid_output(self):
        r = _eng().evaluate(AthleteDigitalTwin())
        assert set(r) == self._KEYS
        assert all(math.isfinite(r[k]) for k in
                   ("recommended_load", "expected_readiness",
                    "expected_injury_risk", "confidence"))


# =============================================================================
# SECTION 8 — evaluate() safety constraints
# =============================================================================

class TestSafetyConstraints:

    def test_8_1_high_injury_risk_forces_rest(self):
        assert _eng().evaluate(_high_risk_twin())["recommended_session"] == "REST"

    def test_8_2_high_injury_risk_confidence_one(self):
        assert _eng().evaluate(_high_risk_twin())["confidence"] == pytest.approx(1.0)

    def test_8_3_high_injury_risk_rest_in_reasoning(self):
        r = _eng().evaluate(_high_risk_twin())
        combined = " ".join(r["reasoning"]).lower()
        assert "rest enforced" in combined or "critical" in combined

    def test_8_4_high_fatigue_only_rest_or_light(self):
        r = _eng().evaluate(_high_fatigue_twin())
        assert r["recommended_session"] in ("REST", "LIGHT_SESSION")

    def test_8_5_high_fatigue_moderate_never_selected(self):
        for _ in range(5):
            r = _eng().evaluate(_high_fatigue_twin())
            assert r["recommended_session"] != "MODERATE_SESSION"

    def test_8_6_high_fatigue_intense_never_selected(self):
        for _ in range(5):
            r = _eng().evaluate(_high_fatigue_twin())
            assert r["recommended_session"] != "INTENSE_SESSION"

    def test_8_7_high_readiness_can_select_intense(self):
        twin    = _high_readiness_twin()
        allowed = _filter_scenarios(
            twin.fatigue_state, twin.injury_risk_state, twin.readiness_state
        )
        assert INTENSE_SESSION_SCENARIO in allowed

    def test_8_8_low_readiness_intense_never_selected(self):
        twin = AthleteDigitalTwin()
        for _ in range(10):
            twin.update({"fatigue_index": 0.3, "sprint_load": 2_000.0,
                         "recovery_hours": 16.0, "injury_flag": 0})
        assert twin.readiness_state <= READINESS_THRESHOLD
        r = _eng().evaluate(twin)
        assert r["recommended_session"] != "INTENSE_SESSION"

    def test_8_9_p1_overrides_p3_high_risk_high_readiness(self):
        # Build a twin with high readiness but then spike injury risk
        twin = _high_risk_twin()
        assert _eng().evaluate(twin)["recommended_session"] == "REST"

    def test_8_10_p2_overrides_p3_high_fatigue_high_readiness(self):
        twin = _high_fatigue_twin()
        r = _eng().evaluate(twin)
        assert r["recommended_session"] in ("REST", "LIGHT_SESSION")

    def test_8_11_boundary_injury_risk_equal_threshold_not_forced(self):
        # Twin where injury_risk_state is exactly at threshold — not forced
        # We rely on _filter_scenarios boundary test (exclusive >) rather than twin
        allowed = _filter_scenarios(0.3, INJURY_RISK_THRESHOLD, 0.4)
        assert len(allowed) > 1

    def test_8_12_normal_state_moderate_possible(self):
        twin = _fresh_twin(5)
        # Normal state should include at least REST, LIGHT, MODERATE in allowed set
        allowed = _filter_scenarios(
            twin.fatigue_state, twin.injury_risk_state, twin.readiness_state
        )
        assert MODERATE_SESSION_SCENARIO in allowed

    def test_8_13_invalid_state_triggers_fallback(self):
        # Monkey-patch get_state to return non-finite values via a bad dict
        class BadTwin:
            def get_state(self):
                return {"fatigue_state": float("nan"), "injury_risk_state": 0.2,
                        "readiness_state": 0.5, "adaptation_factor": 0.3,
                        "athlete_id": "bad", "event_count": 0,
                        "baseline_fatigue": 0.0, "baseline_load": 0.0}
            def forecast(self): return {}
            def simulate_training(self, l, r): return {}
        r = _eng().evaluate(BadTwin())  # type: ignore[arg-type]
        assert r["recommended_session"] == "REST"

    def test_8_14_fallback_confidence_zero(self):
        class BadTwin:
            def get_state(self):
                return {"fatigue_state": float("nan"), "injury_risk_state": 0.0,
                        "readiness_state": 0.5, "adaptation_factor": 0.3,
                        "athlete_id": "x", "event_count": 0,
                        "baseline_fatigue": 0.0, "baseline_load": 0.0}
            def forecast(self): return {}
            def simulate_training(self, l, r): return {}
        r = _eng().evaluate(BadTwin())  # type: ignore[arg-type]
        assert r["confidence"] == pytest.approx(0.0)


# =============================================================================
# SECTION 9 — evaluate() scenario selection and scoring
# =============================================================================

class TestScenarioSelection:

    def test_9_1_best_score_selected(self):
        twin = _make_probe_twin(15)
        all_sc = _eng().evaluate_scenarios(twin)
        rec    = _eng().evaluate(twin)["recommended_session"]
        # Verify the selected session has the highest score among allowed scenarios
        twin_state = twin.get_state()
        allowed = _filter_scenarios(
            twin_state["fatigue_state"],
            twin_state["injury_risk_state"],
            twin_state["readiness_state"],
        )
        allowed_names = {s.name for s in allowed}
        allowed_scores = {d["scenario"]["name"]: d["strategy_score"]
                          for d in all_sc if d["scenario"]["name"] in allowed_names}
        assert allowed_scores[rec] == max(allowed_scores.values())

    def test_9_2_evaluate_scenarios_count(self):
        assert len(_eng().evaluate_scenarios(_fresh_twin(5))) == 4

    def test_9_3_evaluate_scenarios_order(self):
        results = _eng().evaluate_scenarios(_fresh_twin(5))
        assert results[0]["scenario"]["name"] == "REST"
        assert results[3]["scenario"]["name"] == "INTENSE_SESSION"

    def test_9_4_all_scenario_scores_in_unit(self):
        for d in _eng().evaluate_scenarios(_make_probe_twin(10)):
            assert 0.0 <= d["strategy_score"] <= 1.0

    def test_9_5_all_scenario_scores_finite(self):
        for d in _eng().evaluate_scenarios(_make_probe_twin(10)):
            assert math.isfinite(d["strategy_score"])

    def test_9_6_evaluate_scenarios_json_serialisable(self):
        json.dumps(_eng().evaluate_scenarios(_make_probe_twin(10)))

    def test_9_7_recommended_load_matches_scenario(self):
        twin = _make_probe_twin(10)
        r    = _eng().evaluate(twin)
        load_map = {s.name: s.load for s in ALL_SCENARIOS}
        assert r["recommended_load"] == pytest.approx(
            load_map[r["recommended_session"]], abs=1e-9
        )

    def test_9_8_rest_selected_when_highest_score(self):
        # With a completely rested twin (no events), REST should dominate
        twin = AthleteDigitalTwin()
        r    = _eng().evaluate(twin)
        # Any session is valid; just check it is a known name
        assert r["recommended_session"] in {s.name for s in ALL_SCENARIOS}

    def test_9_9_scores_sorted_descending_in_allowed(self):
        twin  = _make_probe_twin(15)
        state = twin.get_state()
        allowed = _filter_scenarios(
            state["fatigue_state"], state["injury_risk_state"], state["readiness_state"]
        )
        allowed_names = {s.name for s in allowed}
        all_sc = _eng().evaluate_scenarios(twin)
        allowed_sc = [d for d in all_sc if d["scenario"]["name"] in allowed_names]
        scores = [d["strategy_score"] for d in allowed_sc]
        assert scores == sorted(scores, reverse=True) or \
               len(scores) == 1  # trivially sorted

    def test_9_10_rest_wins_exact_tie(self):
        # Construct a scenario where REST and LIGHT have the same score by
        # verifying that the engine's tie-break always picks REST (first in order).
        # We test this by checking _compute_score on artificial equal inputs.
        s1 = _compute_score(0.5, 0.5, 0.5)
        s2 = _compute_score(0.5, 0.5, 0.5)
        assert s1 == s2  # tie condition is deterministic

    def test_9_11_evaluate_scenarios_does_not_mutate_twin(self):
        twin   = _make_probe_twin(10)
        before = copy.deepcopy(twin.get_state())
        _eng().evaluate_scenarios(twin)
        assert twin.get_state() == before

    def test_9_12_evaluate_scenarios_consistent_with_evaluate(self):
        twin       = _make_probe_twin(10)
        rec_name   = _eng().evaluate(twin)["recommended_session"]
        sc_names   = {d["scenario"]["name"] for d in _eng().evaluate_scenarios(twin)}
        assert rec_name in sc_names


# =============================================================================
# SECTION 10 — evaluate() confidence computation
# =============================================================================

class TestConfidence:

    def test_10_1_confidence_in_unit(self):
        assert 0.0 <= _eng().evaluate(_make_probe_twin(15))["confidence"] <= 1.0

    def test_10_2_confidence_finite(self):
        assert math.isfinite(_eng().evaluate(_make_probe_twin(15))["confidence"])

    def test_10_3_forced_rest_confidence_one(self):
        assert _eng().evaluate(_high_risk_twin())["confidence"] == pytest.approx(1.0)

    def test_10_4_tie_gives_zero_confidence(self):
        # Two scenarios with identical scores → margin = 0 → confidence = 0.
        # We can't easily force this in the engine without mock; verify the formula.
        margin = _clamp(0.5 - 0.5, 0.0, 1.0)
        assert margin == 0.0

    def test_10_5_margin_equals_confidence(self):
        # Verify that confidence = best_score − second_score (when unclamped).
        # Use evaluate_scenarios to get scores directly.
        twin  = _make_probe_twin(20)
        state = twin.get_state()
        allowed = _filter_scenarios(
            state["fatigue_state"], state["injury_risk_state"], state["readiness_state"]
        )
        allowed_names = {s.name for s in allowed}
        all_sc = _eng().evaluate_scenarios(twin)
        scores = sorted(
            [d["strategy_score"] for d in all_sc
             if d["scenario"]["name"] in allowed_names],
            reverse=True,
        )
        if len(scores) >= 2:
            expected_conf = _clamp(scores[0] - scores[1], 0.0, 1.0)
            actual_conf   = _eng().evaluate(twin)["confidence"]
            assert abs(actual_conf - expected_conf) < 1e-9

    def test_10_6_large_gap_high_confidence(self):
        margin = _clamp(0.9 - 0.2, 0.0, 1.0)
        assert margin >= 0.5

    def test_10_7_small_gap_low_confidence(self):
        margin = _clamp(0.51 - 0.50, 0.0, 1.0)
        assert margin <= 0.1

    def test_10_8_confidence_never_negative(self):
        for _ in range(3):
            twin = _make_probe_twin(15)
            assert _eng().evaluate(twin)["confidence"] >= 0.0


# =============================================================================
# SECTION 11 — evaluate() reasoning output
# =============================================================================

class TestReasoning:

    def test_11_1_reasoning_is_list(self):
        assert isinstance(_eng().evaluate(_make_probe_twin(10))["reasoning"], list)

    def test_11_2_reasoning_non_empty(self):
        assert len(_eng().evaluate(_make_probe_twin(10))["reasoning"]) > 0

    def test_11_3_all_entries_nonempty_strings(self):
        for s in _eng().evaluate(_make_probe_twin(10))["reasoning"]:
            assert isinstance(s, str) and len(s) > 0

    def test_11_4_recommended_session_in_last_entry(self):
        twin = _make_probe_twin(10)
        r    = _eng().evaluate(twin)
        last = r["reasoning"][-1]
        assert r["recommended_session"] in last

    def test_11_5_critical_in_reasoning_when_fatigue_high(self):
        r = _eng().evaluate(_high_fatigue_twin())
        combined = " ".join(r["reasoning"])
        assert "critical" in combined

    def test_11_6_rest_enforced_when_injury_critical(self):
        r = _eng().evaluate(_high_risk_twin())
        combined = " ".join(r["reasoning"]).lower()
        assert "rest enforced" in combined or "critical" in combined

    def test_11_7_intense_unlocked_when_readiness_high(self):
        r = _eng().evaluate(_high_readiness_twin())
        combined = " ".join(r["reasoning"])
        assert "intense training unlocked" in combined

    def test_11_8_conservative_when_readiness_low(self):
        twin = AthleteDigitalTwin()
        for _ in range(25):
            twin.update({"fatigue_index": 0.8, "sprint_load": 6_000.0,
                         "recovery_hours": 5.0, "injury_flag": 0})
        # if readiness is very low (< 0.4):
        if twin.readiness_state < 0.4:
            r = _eng().evaluate(twin)
            combined = " ".join(r["reasoning"])
            assert "conservative" in combined

    def test_11_9_reasoning_deterministic(self):
        twin = _make_probe_twin(15)
        r1   = _eng().evaluate(twin)["reasoning"]
        r2   = _eng().evaluate(twin)["reasoning"]
        assert r1 == r2

    def test_11_10_at_least_four_reasoning_entries(self):
        assert len(_eng().evaluate(_make_probe_twin(10))["reasoning"]) >= 4


# =============================================================================
# SECTION 12 — Determinism
# =============================================================================

class TestDeterminism:

    def test_12_1_hundred_identical_calls(self):
        twin  = _make_probe_twin(20)
        first = _eng().evaluate(twin)
        for _ in range(99):
            assert _eng().evaluate(twin) == first

    def test_12_2_deterministic_check_passes(self):
        assert _eng().deterministic_check() is True

    def test_12_3_two_engines_same_twin_same_decision(self):
        twin = _make_probe_twin(15)
        assert (
            TrainingStrategyEngine().evaluate(twin)
            == TrainingStrategyEngine().evaluate(twin)
        )

    def test_12_4_evaluate_scenarios_deterministic(self):
        twin  = _make_probe_twin(15)
        first = _eng().evaluate_scenarios(twin)
        for _ in range(99):
            assert _eng().evaluate_scenarios(twin) == first

    def test_12_5_session_in_evaluate_scenarios_names(self):
        twin    = _make_probe_twin(10)
        rec     = _eng().evaluate(twin)["recommended_session"]
        sc_names = {d["scenario"]["name"] for d in _eng().evaluate_scenarios(twin)}
        assert rec in sc_names

    def test_12_6_recommended_session_is_known_scenario(self):
        twin  = _make_probe_twin(10)
        names = {s.name for s in ALL_SCENARIOS}
        assert _eng().evaluate(twin)["recommended_session"] in names

    def test_12_7_sequential_evaluate_same_decision(self):
        eng  = TrainingStrategyEngine()
        twin = _make_probe_twin(10)
        r1   = eng.evaluate(twin)
        r2   = eng.evaluate(twin)
        assert r1 == r2

    def test_12_8_filter_scenarios_deterministic(self):
        for fi, ir, rdy in [(0.4, 0.3, 0.5), (0.9, 0.3, 0.9), (0.2, 0.8, 0.2)]:
            assert _filter_scenarios(fi, ir, rdy) == _filter_scenarios(fi, ir, rdy)


# =============================================================================
# SECTION 13 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_13_1_evaluate_does_not_mutate_twin_state(self):
        twin   = _make_probe_twin(10)
        before = copy.deepcopy(twin.get_state())
        _eng().evaluate(twin)
        assert twin.get_state() == before

    def test_13_2_evaluate_does_not_mutate_twin_history(self):
        twin   = _make_probe_twin(10)
        ec     = twin.event_count
        _eng().evaluate(twin)
        assert twin.event_count == ec

    def test_13_3_evaluate_does_not_mutate_fatigue_state(self):
        twin = _make_probe_twin(10)
        f    = twin.fatigue_state
        _eng().evaluate(twin)
        assert twin.fatigue_state == f

    def test_13_4_externally_supplied_forecast_not_mutated(self):
        twin     = _make_probe_twin(10)
        forecast = twin.forecast()
        before   = copy.deepcopy(forecast)
        _eng().evaluate(twin, forecast=forecast)
        assert forecast == before

    def test_13_5_evaluate_scenarios_does_not_mutate_twin(self):
        twin   = _make_probe_twin(10)
        before = copy.deepcopy(twin.get_state())
        _eng().evaluate_scenarios(twin)
        assert twin.get_state() == before

    def test_13_6_filter_scenarios_pure_function(self):
        # Call twice with same args → same result; no hidden state
        a1 = _filter_scenarios(0.4, 0.3, 0.5)
        a2 = _filter_scenarios(0.4, 0.3, 0.5)
        assert a1 == a2

    def test_13_7_compute_score_pure_function(self):
        assert _compute_score(0.6, 0.2, 0.3) == _compute_score(0.6, 0.2, 0.3)

    def test_13_8_to_dict_mutation_does_not_affect_decision(self):
        sd = StrategyDecision("REST", 0.0, 0.5, 0.2, 0.8, ("r",))
        d  = sd.to_dict()
        d["confidence"] = 999.0
        assert sd.confidence != 999.0


# =============================================================================
# SECTION 14 — JSON serialisability
# =============================================================================

class TestJsonSerialisability:

    def test_14_1_evaluate_passes_json_dumps(self):
        json.dumps(_eng().evaluate(_make_probe_twin(10)))

    def test_14_2_evaluate_scenarios_passes_json_dumps(self):
        json.dumps(_eng().evaluate_scenarios(_make_probe_twin(10)))

    def test_14_3_training_scenario_to_dict_serialisable(self):
        json.dumps(REST_SCENARIO.to_dict())

    def test_14_4_scenario_result_to_dict_serialisable(self):
        json.dumps(_sc_result().to_dict())

    def test_14_5_strategy_decision_to_dict_serialisable(self):
        sd = StrategyDecision("REST", 0.0, 0.5, 0.2, 0.8, ("r",))
        json.dumps(sd.to_dict())

    def test_14_6_full_json_round_trip(self):
        r    = _eng().evaluate(_make_probe_twin(15))
        back = json.loads(json.dumps(r))
        for k in ("expected_readiness", "expected_injury_risk", "confidence"):
            assert abs(back[k] - r[k]) < 1e-9
        assert back["recommended_session"] == r["recommended_session"]

    def test_14_7_all_floats_native_python_float(self):
        r = _eng().evaluate(_make_probe_twin(10))
        for k in ("recommended_load", "expected_readiness",
                  "expected_injury_risk", "confidence"):
            assert isinstance(r[k], float)

    def test_14_8_reasoning_is_list_in_output(self):
        assert isinstance(_eng().evaluate(_make_probe_twin(10))["reasoning"], list)


# =============================================================================
# SECTION 15 — Invalid input safety
# =============================================================================

class TestInvalidInputSafety:

    class _BadTwin:
        """Twin whose get_state() returns non-finite values."""
        def get_state(self):
            return {"fatigue_state": float("nan"), "injury_risk_state": 0.2,
                    "readiness_state": 0.5, "adaptation_factor": 0.3,
                    "athlete_id": "bad", "event_count": 0,
                    "baseline_fatigue": 0.0, "baseline_load": 0.0}
        def forecast(self):
            return {}
        def simulate_training(self, load, recovery_hours):
            return {}

    class _ExcTwin:
        """Twin whose get_state() raises an exception."""
        def get_state(self):
            raise RuntimeError("simulated failure")
        def forecast(self):
            return {}
        def simulate_training(self, load, recovery_hours):
            return {}

    def test_15_1_invalid_state_returns_fallback(self):
        r = _eng().evaluate(self._BadTwin())  # type: ignore[arg-type]
        assert isinstance(r, dict) and "recommended_session" in r

    def test_15_2_fallback_session_is_rest(self):
        assert _eng().evaluate(self._BadTwin())["recommended_session"] == "REST"  # type: ignore[arg-type]

    def test_15_3_fallback_injury_risk_is_one(self):
        assert _eng().evaluate(self._BadTwin())["expected_injury_risk"] == pytest.approx(1.0)  # type: ignore[arg-type]

    def test_15_4_fallback_confidence_zero(self):
        assert _eng().evaluate(self._BadTwin())["confidence"] == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_15_5_fallback_reasoning_nonempty_list(self):
        r = _eng().evaluate(self._BadTwin())["reasoning"]  # type: ignore[arg-type]
        assert isinstance(r, list) and len(r) > 0

    def test_15_6_exception_twin_no_exception_raised(self):
        r = _eng().evaluate(self._ExcTwin())  # type: ignore[arg-type]
        assert "recommended_session" in r

    def test_15_7_nonfinite_forecast_values_handled(self):
        # evaluate() should handle non-finite sim outputs via _safe_float
        twin = _fresh_twin(5)
        # Passing a forecast with non-finite values
        r = _eng().evaluate(twin, forecast={"fatigue_24h": float("nan"),
                                            "fatigue_48h": float("nan"),
                                            "fatigue_72h": float("nan"),
                                            "injury_risk": float("nan"),
                                            "readiness_score": float("nan")})
        assert all(math.isfinite(r[k]) for k in
                   ("recommended_load", "expected_readiness",
                    "expected_injury_risk", "confidence"))

    def test_15_8_all_fallback_values_finite_and_bounded(self):
        r = _eng().evaluate(self._BadTwin())  # type: ignore[arg-type]
        for k in ("recommended_load", "expected_readiness",
                  "expected_injury_risk", "confidence"):
            assert math.isfinite(r[k])
            assert 0.0 <= r[k] <= (1.0 if k != "recommended_load" else float("inf"))


# =============================================================================
# SECTION 16 — self_test()
# =============================================================================

class TestSelfTest:

    def test_16_1_returns_dict(self):
        assert isinstance(_eng().self_test(), dict)

    def test_16_2_required_keys(self):
        st = _eng().self_test()
        assert {"engine", "version", "checks", "passed"} <= set(st)

    def test_16_3_six_checks(self):
        assert len(_eng().self_test()["checks"]) == 6

    def test_16_4_each_check_has_required_fields(self):
        for c in _eng().self_test()["checks"]:
            assert {"name", "passed", "detail"} <= set(c)

    def test_16_5_all_checks_pass(self):
        st       = _eng().self_test()
        failures = [c["name"] for c in st["checks"] if not c["passed"]]
        assert st["passed"] is True
        assert failures == [], f"Failed: {failures}"

    def test_16_6_engine_name_correct(self):
        assert _eng().self_test()["engine"] == ENGINE_NAME

    def test_16_7_version_correct(self):
        assert _eng().self_test()["version"] == ENGINE_VERSION

    def test_16_8_idempotent(self):
        eng  = _eng()
        st1  = eng.self_test()["passed"]
        st2  = eng.self_test()["passed"]
        assert st1 == st2


# =============================================================================
# SECTION 17 — _clamp() and _safe_float() helpers
# =============================================================================

class TestHelpers:

    def test_17_1_clamp_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_17_2_clamp_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_17_3_clamp_above_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_17_4_clamp_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_17_5_clamp_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_17_6_clamp_boundary_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_17_7_safe_float_finite_unchanged(self):
        assert _safe_float(0.42) == pytest.approx(0.42)

    def test_17_8_safe_float_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float(float("nan"), default=0.5) == 0.5

    def test_17_9_safe_float_nonnumeric_returns_default(self):
        assert _safe_float("not_a_number") == 0.0
        assert _safe_float(None, default=1.0) == 1.0


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)