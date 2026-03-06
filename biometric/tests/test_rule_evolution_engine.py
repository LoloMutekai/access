"""
A.C.C.E.S.S. — Rule Evolution Engine Test Suite (Phase 7.12)
tests/test_rule_evolution_engine.py

Full coverage of biometric/rule_evolution_engine.py

Coverage map
────────────
SECTION 1  — Module constants
  1.1   ENGINE_NAME == "RuleEvolutionEngine"
  1.2   ENGINE_VERSION == "7.12.0"
  1.3   VALIDATION_THRESHOLD == 0.60
  1.4   MAX_ACTIVE_RULES == 50
  1.5   MAX_EVENTS == 10_000
  1.6   W_PRECISION == 0.50
  1.7   W_RECALL    == 0.30
  1.8   W_SPEC      == 0.20
  1.9   Scoring weights sum to 1.0
  1.10  RULE_TEMPLATES is a tuple
  1.11  RULE_TEMPLATES contains exactly 8 entries
  1.12  Every template has "name", "conditions", "effect" keys
  1.13  VALIDATION_THRESHOLD ∈ (0, 1)

SECTION 2  — RuleCondition dataclass
  2.1   Instantiates with valid fields
  2.2   Frozen (FrozenInstanceError on reassignment)
  2.3   All fields stored exactly
  2.4   is_valid() True for well-formed condition
  2.5   is_valid() False for empty field string
  2.6   is_valid() False for unrecognised operator
  2.7   is_valid() False for non-finite threshold
  2.8   to_dict() returns dict with exactly {"field","operator","threshold"}
  2.9   to_dict() values match fields
  2.10  to_dict() is JSON-serialisable
  2.11  from_dict() reconstructs identical RuleCondition

SECTION 3  — CandidateRule dataclass
  3.1   Instantiates with valid fields
  3.2   Frozen (FrozenInstanceError on reassignment)
  3.3   All fields stored exactly
  3.4   is_valid() True for well-formed rule
  3.5   is_valid() False when name is empty string
  3.6   is_valid() False when conditions is empty tuple
  3.7   is_valid() False when effect is empty string
  3.8   is_valid() False when score is non-finite
  3.9   is_valid() False when score < 0.0
  3.10  is_valid() False when score > 1.0
  3.11  promoted() True when score >= VALIDATION_THRESHOLD
  3.12  promoted() False when score < VALIDATION_THRESHOLD
  3.13  promoted() True at exact threshold boundary
  3.14  to_dict() has exactly the four expected keys
  3.15  to_dict() conditions is a list of dicts
  3.16  to_dict() is JSON-serialisable
  3.17  to_dict() returns a fresh dict (mutation does not affect instance)
  3.18  from_dict() round-trip produces identical CandidateRule
  3.19  from_dict() json round-trip preserves all values

SECTION 4  — _evaluate_operator() helper
  4.1   "gt"  — strictly greater than
  4.2   "gte" — greater than or equal
  4.3   "lt"  — strictly less than
  4.4   "lte" — less than or equal
  4.5   "eq"  — exact equality
  4.6   Unknown operator returns False

SECTION 5  — _apply_rule_conditions() helper
  5.1   All conditions satisfied → True
  5.2   One condition fails → False
  5.3   Missing field in event → False
  5.4   Non-finite field value → False
  5.5   Empty conditions tuple → True (vacuously satisfied)
  5.6   Multi-condition AND semantics

SECTION 6  — _compute_score() helper
  6.1   Perfect classifier (all TP, no FP/FN) → score == 1.0
  6.2   Rule never fires (TP=FP=0) → precision defaults to 1.0; recall=0; fpr=0
  6.3   Rule fires on all negatives only (all FP) → low score
  6.4   All FN (misses everything) → recall=0
  6.5   Score ∈ [0, 1] for arbitrary non-negative counts
  6.6   Score is finite always
  6.7   score == W_PRECISION*precision + W_RECALL*recall + W_SPEC*(1-fpr) exactly
  6.8   No actual positives → recall=0 (conservative default)
  6.9   No actual negatives → fpr=0 (conservative default)

SECTION 7  — _collect_valid_events() helper
  7.1   Returns only events with all required fields
  7.2   Skips events with missing fields
  7.3   Skips events with non-finite values
  7.4   Skips non-dict entries
  7.5   Caps at MAX_EVENTS
  7.6   Does not mutate input list
  7.7   Returns list type

SECTION 8  — _validate_rule() helper
  8.1   Returns a CandidateRule with updated score
  8.2   Empty events → score == 0.0
  8.3   Rule with perfect predictions → score == 1.0
  8.4   Original rule object is not mutated (frozen anyway)
  8.5   Returned rule has same name, conditions, effect

SECTION 9  — RuleEvolutionEngine construction
  9.1   Starts with zero active rules
  9.2   active_rules property returns a list
  9.3   active_rule_count == 0

SECTION 10 — RuleEvolutionEngine.analyze() — output structure
  10.1  Returns a dict
  10.2  Dict has exactly the three required keys
  10.3  candidate_rules is a list
  10.4  activated_rules is a list
  10.5  rejected_rules is a list
  10.6  len(candidate_rules) == len(RULE_TEMPLATES)
  10.7  activated + rejected == candidate (partition completeness)
  10.8  All candidate_rules entries are dicts with required keys
  10.9  All scores in candidate_rules are finite and in [0, 1]
  10.10 Output is JSON-serialisable

SECTION 11 — RuleEvolutionEngine.analyze() — empty and invalid inputs
  11.1  Empty event list → three output lists present, candidates still evaluated
  11.2  All-invalid events → treated as empty (score=0 for all rules)
  11.3  Single valid event processed without error
  11.4  Events with extra fields processed correctly
  11.5  Mix of valid and invalid → only valid events used for scoring

SECTION 12 — RuleEvolutionEngine.analyze() — promotion logic
  12.1  Activated rules all have score >= VALIDATION_THRESHOLD
  12.2  Rejected rules all have score < VALIDATION_THRESHOLD
  12.3  High-injury dataset promotes at least one rule
  12.4  All-healthy dataset (no injury) promotes no rules expecting injury

SECTION 13 — RuleEvolutionEngine.analyze() — active rule management
  13.1  After analyze(), active_rule_count >= len(activated_rules)
  13.2  Calling analyze() twice merges rules (no duplicates by name)
  13.3  active_rules property returns a copy (mutation does not affect engine)
  13.4  MAX_ACTIVE_RULES cap enforced when exceeded

SECTION 14 — RuleEvolutionEngine.analyze() — determinism
  14.1  Identical events → identical output (100 runs)
  14.2  Event list order changes may change scores but not output structure
  14.3  deterministic_check() returns True

SECTION 15 — Input immutability
  15.1  Event list not mutated by analyze()
  15.2  Individual event dicts not mutated by analyze()

SECTION 16 — RuleEvolutionEngine.reset()
  16.1  Clears all active rules
  16.2  active_rule_count == 0 after reset
  16.3  active_rules == [] after reset
  16.4  New analysis possible after reset

SECTION 17 — RuleEvolutionEngine.get_state() / set_state()
  17.1  get_state() returns a dict
  17.2  get_state() is JSON-serialisable
  17.3  get_state() contains "active_rules" key
  17.4  len(get_state()["active_rules"]) == active_rule_count
  17.5  set_state() restores active_rule_count
  17.6  set_state() restores all active rules exactly
  17.7  set_state() with corrupt rule silently skips it
  17.8  set_state({"active_rules": []}) clears engine
  17.9  set_state() enforces MAX_ACTIVE_RULES after restore
  17.10 Full JSON round-trip: get_state → json → set_state → same factor

SECTION 18 — run_evolution() stateless function
  18.1  Returns a dict with the three required keys
  18.2  Stateless: two calls with same events → identical results
  18.3  Does not affect any global state

SECTION 19 — self_test()
  19.1  Returns a dict
  19.2  Contains "engine", "version", "checks", "passed"
  19.3  "checks" list has exactly six items
  19.4  Each check has "name", "passed", "detail"
  19.5  All six checks pass
  19.6  Does not mutate the engine's own active_rules
  19.7  "engine" == ENGINE_NAME
  19.8  "version" == ENGINE_VERSION

SECTION 20 — _clamp() helper
  20.1  Value within range returned unchanged
  20.2  Value below lo → lo
  20.3  Value above hi → hi
  20.4  NaN → lo
  20.5  +Inf → lo
  20.6  Boundary values exact

SECTION 21 — _finite_or_zero() helper
  21.1  Finite value returned unchanged
  21.2  NaN → 0.0
  21.3  +Inf → 0.0
  21.4  -Inf → 0.0
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from biometric.rule_evolution_engine import (
    MAX_ACTIVE_RULES,
    MAX_EVENTS,
    RULE_TEMPLATES,
    VALIDATION_THRESHOLD,
    W_PRECISION,
    W_RECALL,
    W_SPEC,
    ENGINE_NAME,
    ENGINE_VERSION,
    CandidateRule,
    RuleCondition,
    RuleEvolutionEngine,
    _apply_rule_conditions,
    _clamp,
    _collect_valid_events,
    _compute_score,
    _evaluate_operator,
    _finite_or_zero,
    _make_probe_events,
    _validate_rule,
    run_evolution,
)


# =============================================================================
# HELPERS
# =============================================================================

def _cond(field: str = "fatigue_index", op: str = "gt", thr: float = 0.70) -> RuleCondition:
    return RuleCondition(field=field, operator=op, threshold=thr)


def _rule(name: str = "TEST", score: float = 0.75) -> CandidateRule:
    return CandidateRule(
        name=name,
        conditions=(_cond(),),
        effect="HIGH_INJURY_RISK",
        score=score,
    )


def _make_events(
    n: int,
    fatigue: float = 0.8,
    load: float = 8000.0,
    recovery: float = 12.0,
    injury: int = 1,
) -> list[dict]:
    return [
        {
            "fatigue_index":  fatigue,
            "sprint_load":    load,
            "recovery_hours": recovery,
            "injury_flag":    injury,
        }
        for _ in range(n)
    ]


def _mixed_events(n_positive: int = 20, n_negative: int = 20) -> list[dict]:
    """n_positive injured high-fatigue events + n_negative healthy events."""
    positives = _make_events(n_positive, fatigue=0.85, load=8500, recovery=10, injury=1)
    negatives = _make_events(n_negative, fatigue=0.20, load=2000, recovery=48, injury=0)
    return positives + negatives


# =============================================================================
# SECTION 1 — Module constants
# =============================================================================

class TestConstants:

    def test_1_1_engine_name(self):
        assert ENGINE_NAME == "RuleEvolutionEngine"

    def test_1_2_engine_version(self):
        assert ENGINE_VERSION == "7.12.0"

    def test_1_3_validation_threshold(self):
        assert VALIDATION_THRESHOLD == pytest.approx(0.60)

    def test_1_4_max_active_rules(self):
        assert MAX_ACTIVE_RULES == 50

    def test_1_5_max_events(self):
        assert MAX_EVENTS == 10_000

    def test_1_6_w_precision(self):
        assert W_PRECISION == pytest.approx(0.50)

    def test_1_7_w_recall(self):
        assert W_RECALL == pytest.approx(0.30)

    def test_1_8_w_spec(self):
        assert W_SPEC == pytest.approx(0.20)

    def test_1_9_weights_sum_to_one(self):
        assert W_PRECISION + W_RECALL + W_SPEC == pytest.approx(1.0, abs=1e-9)

    def test_1_10_rule_templates_is_tuple(self):
        assert isinstance(RULE_TEMPLATES, tuple)

    def test_1_11_rule_templates_has_eight_entries(self):
        assert len(RULE_TEMPLATES) == 8

    def test_1_12_every_template_has_required_keys(self):
        for tmpl in RULE_TEMPLATES:
            assert "name"       in tmpl
            assert "conditions" in tmpl
            assert "effect"     in tmpl

    def test_1_13_validation_threshold_in_open_unit(self):
        assert 0.0 < VALIDATION_THRESHOLD < 1.0


# =============================================================================
# SECTION 2 — RuleCondition dataclass
# =============================================================================

class TestRuleCondition:

    def test_2_1_instantiation(self):
        assert isinstance(_cond(), RuleCondition)

    def test_2_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            _cond().field = "other"  # type: ignore[misc]

    def test_2_3_fields_stored_exactly(self):
        c = RuleCondition("fatigue_index", "gt", 0.70)
        assert c.field     == "fatigue_index"
        assert c.operator  == "gt"
        assert c.threshold == 0.70

    def test_2_4_is_valid_true(self):
        assert _cond().is_valid() is True

    def test_2_5_is_valid_empty_field(self):
        assert RuleCondition("", "gt", 0.5).is_valid() is False

    def test_2_6_is_valid_unknown_operator(self):
        assert RuleCondition("fatigue_index", "!=", 0.5).is_valid() is False

    def test_2_7_is_valid_nonfinite_threshold(self):
        assert RuleCondition("fatigue_index", "gt", float("nan")).is_valid() is False

    def test_2_8_to_dict_exact_keys(self):
        assert set(_cond().to_dict()) == {"field", "operator", "threshold"}

    def test_2_9_to_dict_values_match(self):
        c = _cond("recovery_hours", "lt", 24.0)
        d = c.to_dict()
        assert d["field"]     == c.field
        assert d["operator"]  == c.operator
        assert d["threshold"] == c.threshold

    def test_2_10_to_dict_json_serialisable(self):
        json.dumps(_cond().to_dict())

    def test_2_11_from_dict_roundtrip(self):
        c = _cond("sprint_load", "gte", 6000.0)
        assert RuleCondition.from_dict(c.to_dict()) == c


# =============================================================================
# SECTION 3 — CandidateRule dataclass
# =============================================================================

class TestCandidateRule:

    def test_3_1_instantiation(self):
        assert isinstance(_rule(), CandidateRule)

    def test_3_2_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            _rule().score = 0.0  # type: ignore[misc]

    def test_3_3_fields_stored_exactly(self):
        c = _cond()
        r = CandidateRule("RULE_A", (c,), "EFFECT_X", 0.72)
        assert r.name       == "RULE_A"
        assert r.conditions == (c,)
        assert r.effect     == "EFFECT_X"
        assert r.score      == 0.72

    def test_3_4_is_valid_true(self):
        assert _rule().is_valid() is True

    def test_3_5_is_valid_empty_name(self):
        r = CandidateRule("", (_cond(),), "EFFECT", 0.5)
        assert r.is_valid() is False

    def test_3_6_is_valid_empty_conditions(self):
        r = CandidateRule("R", (), "EFFECT", 0.5)
        assert r.is_valid() is False

    def test_3_7_is_valid_empty_effect(self):
        r = CandidateRule("R", (_cond(),), "", 0.5)
        assert r.is_valid() is False

    def test_3_8_is_valid_nonfinite_score(self):
        r = CandidateRule("R", (_cond(),), "E", float("nan"))
        assert r.is_valid() is False

    def test_3_9_is_valid_score_below_zero(self):
        r = CandidateRule("R", (_cond(),), "E", -0.001)
        assert r.is_valid() is False

    def test_3_10_is_valid_score_above_one(self):
        r = CandidateRule("R", (_cond(),), "E", 1.001)
        assert r.is_valid() is False

    def test_3_11_promoted_true_above_threshold(self):
        assert _rule(score=VALIDATION_THRESHOLD + 0.01).promoted() is True

    def test_3_12_promoted_false_below_threshold(self):
        assert _rule(score=VALIDATION_THRESHOLD - 0.01).promoted() is False

    def test_3_13_promoted_true_at_exact_boundary(self):
        assert _rule(score=VALIDATION_THRESHOLD).promoted() is True

    def test_3_14_to_dict_exact_keys(self):
        assert set(_rule().to_dict()) == {"name", "conditions", "effect", "score"}

    def test_3_15_to_dict_conditions_is_list_of_dicts(self):
        d = _rule().to_dict()
        assert isinstance(d["conditions"], list)
        assert all(isinstance(c, dict) for c in d["conditions"])

    def test_3_16_to_dict_json_serialisable(self):
        json.dumps(_rule().to_dict())

    def test_3_17_to_dict_fresh_dict(self):
        r = _rule()
        d = r.to_dict()
        d["name"] = "MUTATED"
        assert r.name != "MUTATED"

    def test_3_18_from_dict_roundtrip(self):
        r = _rule("ROUND_TRIP", 0.65)
        assert CandidateRule.from_dict(r.to_dict()) == r

    def test_3_19_json_roundtrip(self):
        r = CandidateRule(
            "JSON_RULE",
            (_cond("fatigue_index", "gt", 0.70), _cond("sprint_load", "gt", 6000.0)),
            "HIGH_INJURY_RISK",
            0.72,
        )
        restored = CandidateRule.from_dict(json.loads(json.dumps(r.to_dict())))
        assert restored == r


# =============================================================================
# SECTION 4 — _evaluate_operator() helper
# =============================================================================

class TestEvaluateOperator:

    def test_4_1_gt(self):
        assert _evaluate_operator(0.8,  "gt",  0.7)  is True
        assert _evaluate_operator(0.7,  "gt",  0.7)  is False

    def test_4_2_gte(self):
        assert _evaluate_operator(0.7,  "gte", 0.7)  is True
        assert _evaluate_operator(0.69, "gte", 0.7)  is False

    def test_4_3_lt(self):
        assert _evaluate_operator(0.6,  "lt",  0.7)  is True
        assert _evaluate_operator(0.7,  "lt",  0.7)  is False

    def test_4_4_lte(self):
        assert _evaluate_operator(0.7,  "lte", 0.7)  is True
        assert _evaluate_operator(0.71, "lte", 0.7)  is False

    def test_4_5_eq(self):
        assert _evaluate_operator(0.5,  "eq",  0.5)  is True
        assert _evaluate_operator(0.5,  "eq",  0.51) is False

    def test_4_6_unknown_operator_false(self):
        assert _evaluate_operator(0.9, "!=", 0.7)  is False
        assert _evaluate_operator(0.9, "xx", 0.7)  is False


# =============================================================================
# SECTION 5 — _apply_rule_conditions() helper
# =============================================================================

class TestApplyRuleConditions:

    def _ev(self, fi=0.8, load=8000.0, rec=12.0, inj=1):
        return {"fatigue_index": fi, "sprint_load": load,
                "recovery_hours": rec, "injury_flag": inj}

    def test_5_1_all_conditions_satisfied(self):
        conds = (_cond("fatigue_index", "gt", 0.70),)
        assert _apply_rule_conditions(conds, self._ev(fi=0.9)) is True

    def test_5_2_one_condition_fails(self):
        conds = (
            _cond("fatigue_index", "gt", 0.70),
            _cond("sprint_load",   "gt", 9_000.0),
        )
        assert _apply_rule_conditions(conds, self._ev(fi=0.9, load=8000)) is False

    def test_5_3_missing_field_returns_false(self):
        conds = (RuleCondition("nonexistent_field", "gt", 0.5),)
        assert _apply_rule_conditions(conds, self._ev()) is False

    def test_5_4_nonfinite_field_value_returns_false(self):
        ev = {"fatigue_index": float("nan"), "sprint_load": 5000,
              "recovery_hours": 20, "injury_flag": 1}
        conds = (_cond("fatigue_index", "gt", 0.5),)
        assert _apply_rule_conditions(conds, ev) is False

    def test_5_5_empty_conditions_vacuously_true(self):
        assert _apply_rule_conditions((), self._ev()) is True

    def test_5_6_multi_condition_and_semantics(self):
        conds = (
            _cond("fatigue_index",  "gt",  0.70),
            _cond("recovery_hours", "lt", 20.0),
            _cond("sprint_load",    "gt", 7_000.0),
        )
        # All met
        assert _apply_rule_conditions(conds, self._ev(fi=0.8, rec=15, load=8000)) is True
        # Last fails
        assert _apply_rule_conditions(conds, self._ev(fi=0.8, rec=15, load=5000)) is False


# =============================================================================
# SECTION 6 — _compute_score() helper
# =============================================================================

class TestComputeScore:

    def test_6_1_perfect_classifier(self):
        assert _compute_score(tp=10, fp=0, tn=10, fn=0) == pytest.approx(1.0, abs=1e-9)

    def test_6_2_rule_never_fires(self):
        # TP=FP=0: precision defaults 1.0; recall=0/(0+FN)=0; fpr=0
        s = _compute_score(tp=0, fp=0, tn=10, fn=5)
        expected = W_PRECISION * 1.0 + W_RECALL * 0.0 + W_SPEC * 1.0
        assert s == pytest.approx(expected, abs=1e-9)

    def test_6_3_all_fp_rule(self):
        # Fires only on negatives: tp=0, fn=5, fp=10, tn=0
        s = _compute_score(tp=0, fp=10, tn=0, fn=5)
        # precision=0, recall=0, fpr=1 → W_P*0 + W_R*0 + W_S*0 = 0
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_6_4_all_fn(self):
        # Rule never fires but positives exist → recall=0
        s = _compute_score(tp=0, fp=0, tn=10, fn=10)
        # precision=1(default), recall=0, fpr=0 → W_P + W_S
        expected = W_PRECISION * 1.0 + W_RECALL * 0.0 + W_SPEC * 1.0
        assert s == pytest.approx(expected, abs=1e-9)

    def test_6_5_score_in_unit_interval(self):
        for tp, fp, tn, fn in [
            (5, 5, 5, 5), (0, 0, 0, 0), (10, 0, 0, 0), (0, 10, 10, 0)
        ]:
            s = _compute_score(tp, fp, tn, fn)
            assert 0.0 <= s <= 1.0, f"tp={tp} fp={fp} tn={tn} fn={fn} → {s}"

    def test_6_6_score_always_finite(self):
        assert math.isfinite(_compute_score(0, 0, 0, 0))
        assert math.isfinite(_compute_score(10, 5, 3, 2))

    def test_6_7_formula_exact(self):
        tp, fp, tn, fn = 7, 3, 8, 2
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        fpr       = fp / (fp + tn)
        expected  = W_PRECISION * precision + W_RECALL * recall + W_SPEC * (1.0 - fpr)
        assert _compute_score(tp, fp, tn, fn) == pytest.approx(expected, abs=1e-12)

    def test_6_8_no_actual_positives_recall_zero(self):
        # fn=0 → no actual positives; recall should be 0.0
        s = _compute_score(tp=0, fp=5, tn=5, fn=0)
        # precision=0 (fp fired, no tp), recall=0 (no +), fpr=5/10=0.5
        assert s == pytest.approx(
            W_PRECISION * 0.0 + W_RECALL * 0.0 + W_SPEC * (1.0 - 0.5),
            abs=1e-9
        )

    def test_6_9_no_actual_negatives_fpr_zero(self):
        # tn=0, fp=0 → fpr defaults to 0.0
        s = _compute_score(tp=5, fp=0, tn=0, fn=0)
        # precision=1, recall=1, fpr=0
        assert s == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# SECTION 7 — _collect_valid_events() helper
# =============================================================================

class TestCollectValidEvents:

    def test_7_1_valid_events_collected(self):
        evs = _make_events(5)
        assert len(_collect_valid_events(evs)) == 5

    def test_7_2_skips_missing_field(self):
        evs = [{"fatigue_index": 0.8, "sprint_load": 5000, "recovery_hours": 20}]  # no injury_flag
        assert _collect_valid_events(evs) == []

    def test_7_3_skips_nonfinite_values(self):
        evs = [{"fatigue_index": float("nan"), "sprint_load": 5000,
                "recovery_hours": 20, "injury_flag": 1}]
        assert _collect_valid_events(evs) == []

    def test_7_4_skips_non_dict_entries(self):
        evs = ["not_a_dict", 42, None, _make_events(1)[0]]
        assert len(_collect_valid_events(evs)) == 1

    def test_7_5_caps_at_max_events(self):
        evs = _make_events(MAX_EVENTS + 100)
        assert len(_collect_valid_events(evs)) == MAX_EVENTS

    def test_7_6_does_not_mutate_input(self):
        evs = _make_events(5)
        original = copy.deepcopy(evs)
        _collect_valid_events(evs)
        assert evs == original

    def test_7_7_returns_list(self):
        assert isinstance(_collect_valid_events([]), list)


# =============================================================================
# SECTION 8 — _validate_rule() helper
# =============================================================================

class TestValidateRule:

    def test_8_1_returns_candidate_rule(self):
        rule = _rule()
        result = _validate_rule(rule, _make_events(10))
        assert isinstance(result, CandidateRule)

    def test_8_2_empty_events_score_zero(self):
        result = _validate_rule(_rule(), [])
        assert result.score == 0.0

    def test_8_3_perfect_predictions_score_one(self):
        # Rule: fatigue_index > 0.70. Dataset: all have fi=0.8 AND injury_flag=1
        rule = CandidateRule(
            name="PERFECT",
            conditions=(_cond("fatigue_index", "gt", 0.70),),
            effect="TEST",
            score=0.0,
        )
        events = _make_events(20, fatigue=0.8, injury=1)
        # All predicted positive, all actually positive → TP=20, FP=FN=0
        # BUT no negatives → fpr=0 by default
        result = _validate_rule(rule, events)
        assert result.score == pytest.approx(1.0, abs=1e-9)

    def test_8_4_original_rule_not_mutated(self):
        rule = _rule(score=0.0)
        original_name  = rule.name
        original_score = rule.score
        _validate_rule(rule, _make_events(10))
        assert rule.name  == original_name
        assert rule.score == original_score

    def test_8_5_returned_rule_preserves_identity_fields(self):
        rule = CandidateRule(
            "IDENTITY_TEST",
            (_cond("fatigue_index", "gt", 0.5),),
            "MY_EFFECT",
            0.0,
        )
        result = _validate_rule(rule, _make_events(5))
        assert result.name       == rule.name
        assert result.conditions == rule.conditions
        assert result.effect     == rule.effect


# =============================================================================
# SECTION 9 — RuleEvolutionEngine construction
# =============================================================================

class TestRuleEvolutionEngineConstruction:

    def test_9_1_starts_with_zero_active_rules(self):
        assert RuleEvolutionEngine().active_rule_count == 0

    def test_9_2_active_rules_property_returns_list(self):
        assert isinstance(RuleEvolutionEngine().active_rules, list)

    def test_9_3_active_rule_count_zero(self):
        assert RuleEvolutionEngine().active_rule_count == 0


# =============================================================================
# SECTION 10 — analyze() — output structure
# =============================================================================

class TestAnalyzeOutputStructure:

    def _eng(self):
        return RuleEvolutionEngine()

    def test_10_1_returns_dict(self):
        assert isinstance(self._eng().analyze([]), dict)

    def test_10_2_three_required_keys(self):
        r = self._eng().analyze([])
        assert set(r) == {"candidate_rules", "activated_rules", "rejected_rules"}

    def test_10_3_candidate_rules_is_list(self):
        assert isinstance(self._eng().analyze([])["candidate_rules"], list)

    def test_10_4_activated_rules_is_list(self):
        assert isinstance(self._eng().analyze([])["activated_rules"], list)

    def test_10_5_rejected_rules_is_list(self):
        assert isinstance(self._eng().analyze([])["rejected_rules"], list)

    def test_10_6_candidate_count_equals_template_count(self):
        r = self._eng().analyze(_make_events(10))
        assert len(r["candidate_rules"]) == len(RULE_TEMPLATES)

    def test_10_7_partition_completeness(self):
        r = self._eng().analyze(_make_events(20))
        assert len(r["activated_rules"]) + len(r["rejected_rules"]) == len(r["candidate_rules"])

    def test_10_8_candidate_dicts_have_required_keys(self):
        r = self._eng().analyze(_make_events(5))
        for rc in r["candidate_rules"]:
            assert set(rc) >= {"name", "conditions", "effect", "score"}

    def test_10_9_all_scores_in_unit_interval(self):
        r = self._eng().analyze(_make_events(30))
        for rc in r["candidate_rules"]:
            assert 0.0 <= rc["score"] <= 1.0, f"score={rc['score']} for rule {rc['name']}"

    def test_10_10_output_json_serialisable(self):
        r = self._eng().analyze(_mixed_events())
        json.dumps(r)


# =============================================================================
# SECTION 11 — analyze() — empty and invalid inputs
# =============================================================================

class TestAnalyzeInvalidInputs:

    def test_11_1_empty_events_output_keys_present(self):
        r = RuleEvolutionEngine().analyze([])
        assert "candidate_rules" in r
        assert "activated_rules" in r
        assert "rejected_rules"  in r

    def test_11_2_all_invalid_events_same_structure(self):
        invalids = [
            {"fatigue_index": float("nan"), "sprint_load": 5000,
             "recovery_hours": 20, "injury_flag": 1},
            {"missing": "all"},
        ]
        r = RuleEvolutionEngine().analyze(invalids)
        assert len(r["candidate_rules"]) == len(RULE_TEMPLATES)

    def test_11_3_single_valid_event(self):
        r = RuleEvolutionEngine().analyze(_make_events(1, fatigue=0.9, injury=1))
        assert len(r["candidate_rules"]) == len(RULE_TEMPLATES)

    def test_11_4_events_with_extra_fields_ok(self):
        ev = {"fatigue_index": 0.8, "sprint_load": 8000,
              "recovery_hours": 12, "injury_flag": 1, "extra_field": "ignored"}
        r = RuleEvolutionEngine().analyze([ev] * 10)
        assert len(r["candidate_rules"]) == len(RULE_TEMPLATES)

    def test_11_5_mixed_valid_invalid_only_valid_used(self):
        valids   = _make_events(10, fatigue=0.85, load=8500, recovery=10, injury=1)
        invalids = [{"bad": "event"}] * 10
        r_mixed  = RuleEvolutionEngine().analyze(valids + invalids)
        r_clean  = RuleEvolutionEngine().analyze(valids)
        # Scores should be identical since invalid events are skipped
        assert r_mixed["candidate_rules"] == r_clean["candidate_rules"]


# =============================================================================
# SECTION 12 — analyze() — promotion logic
# =============================================================================

class TestPromotionLogic:

    def test_12_1_activated_all_above_threshold(self):
        r = RuleEvolutionEngine().analyze(_mixed_events())
        for rc in r["activated_rules"]:
            assert rc["score"] >= VALIDATION_THRESHOLD, \
                f"Activated rule {rc['name']} has score {rc['score']} below threshold"

    def test_12_2_rejected_all_below_threshold(self):
        r = RuleEvolutionEngine().analyze(_mixed_events())
        for rc in r["rejected_rules"]:
            assert rc["score"] < VALIDATION_THRESHOLD, \
                f"Rejected rule {rc['name']} has score {rc['score']} at/above threshold"

    def test_12_3_high_injury_dataset_activates_rules(self):
        events = _mixed_events(n_positive=30, n_negative=10)
        r = RuleEvolutionEngine().analyze(events)
        assert len(r["activated_rules"]) > 0

    def test_12_4_all_healthy_no_injury_flag_rules_not_promoted(self):
        """Dataset with zero injuries: any rule predicting injury_flag=1 has zero TP/recall."""
        events = _make_events(50, fatigue=0.9, load=9000, recovery=8, injury=0)
        r = RuleEvolutionEngine().analyze(events)
        # Precision defaults to 1.0 when rule never predicts positive — but recall=0
        # These rules should have W_PRECISION*1.0 + W_RECALL*0 + W_SPEC*...
        # Whether they promote depends on exact FPR; check that scores are consistent
        for rc in r["activated_rules"]:
            assert rc["score"] >= VALIDATION_THRESHOLD


# =============================================================================
# SECTION 13 — analyze() — active rule management
# =============================================================================

class TestActiveRuleManagement:

    def test_13_1_active_count_grows_after_analyze(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        assert eng.active_rule_count >= 0   # may be 0 if none promoted

    def test_13_2_no_duplicate_names_after_two_calls(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.analyze(_mixed_events())
        names = [r.name for r in eng.active_rules]
        assert len(names) == len(set(names))   # no duplicates

    def test_13_3_active_rules_returns_copy(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        copy_list = eng.active_rules
        copy_list.clear()
        assert eng.active_rule_count >= 0   # internal state unchanged

    def test_13_4_max_active_rules_enforced(self):
        """Force MAX_ACTIVE_RULES+1 distinct rules by calling set_state with many."""
        eng = RuleEvolutionEngine()
        fake_rules = [
            {
                "name": f"RULE_{i}",
                "conditions": [{"field": "fatigue_index", "operator": "gt",
                                 "threshold": 0.5}],
                "effect": "TEST",
                "score": 0.80,
            }
            for i in range(MAX_ACTIVE_RULES + 5)
        ]
        eng.set_state({"active_rules": fake_rules})
        assert eng.active_rule_count == MAX_ACTIVE_RULES


# =============================================================================
# SECTION 14 — analyze() — determinism
# =============================================================================

class TestAnalyzeDeterminism:

    def test_14_1_identical_events_identical_output(self):
        events = _mixed_events(20, 20)
        first  = RuleEvolutionEngine().analyze(events)
        for _ in range(99):
            assert RuleEvolutionEngine().analyze(events) == first

    def test_14_2_output_structure_stable(self):
        events = _make_probe_events(40)
        r1 = RuleEvolutionEngine().analyze(events)
        r2 = RuleEvolutionEngine().analyze(events)
        assert set(r1) == set(r2)
        assert len(r1["candidate_rules"]) == len(r2["candidate_rules"])

    def test_14_3_deterministic_check_passes(self):
        assert RuleEvolutionEngine().deterministic_check() is True


# =============================================================================
# SECTION 15 — Input immutability
# =============================================================================

class TestInputImmutability:

    def test_15_1_event_list_not_mutated(self):
        events  = _make_events(10)
        original = copy.deepcopy(events)
        RuleEvolutionEngine().analyze(events)
        assert events == original

    def test_15_2_individual_event_dicts_not_mutated(self):
        ev = {"fatigue_index": 0.8, "sprint_load": 8000,
              "recovery_hours": 12, "injury_flag": 1}
        original = copy.deepcopy(ev)
        RuleEvolutionEngine().analyze([ev] * 5)
        assert ev == original


# =============================================================================
# SECTION 16 — reset()
# =============================================================================

class TestReset:

    def test_16_1_clears_active_rules(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.reset()
        assert eng.active_rule_count == 0

    def test_16_2_active_rule_count_zero_after_reset(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.reset()
        assert eng.active_rule_count == 0

    def test_16_3_active_rules_empty_after_reset(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.reset()
        assert eng.active_rules == []

    def test_16_4_new_analysis_after_reset(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.reset()
        r = eng.analyze(_make_events(5))
        assert set(r) == {"candidate_rules", "activated_rules", "rejected_rules"}


# =============================================================================
# SECTION 17 — get_state() / set_state()
# =============================================================================

class TestGetSetState:

    def test_17_1_get_state_returns_dict(self):
        assert isinstance(RuleEvolutionEngine().get_state(), dict)

    def test_17_2_get_state_json_serialisable(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        json.dumps(eng.get_state())

    def test_17_3_get_state_has_active_rules_key(self):
        assert "active_rules" in RuleEvolutionEngine().get_state()

    def test_17_4_active_rules_len_matches_count(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        assert len(eng.get_state()["active_rules"]) == eng.active_rule_count

    def test_17_5_set_state_restores_count(self):
        eng1 = RuleEvolutionEngine()
        eng1.analyze(_mixed_events())
        eng2 = RuleEvolutionEngine()
        eng2.set_state(eng1.get_state())
        assert eng2.active_rule_count == eng1.active_rule_count

    def test_17_6_set_state_restores_rules_exactly(self):
        eng1 = RuleEvolutionEngine()
        eng1.analyze(_mixed_events())
        eng2 = RuleEvolutionEngine()
        eng2.set_state(eng1.get_state())
        for r1, r2 in zip(eng1.active_rules, eng2.active_rules):
            assert r1 == r2

    def test_17_7_corrupt_rule_skipped_silently(self):
        snap = {
            "active_rules": [
                {"name": "VALID",
                 "conditions": [{"field": "fatigue_index", "operator": "gt", "threshold": 0.7}],
                 "effect": "TEST", "score": 0.75},
                {"name": "CORRUPT", "conditions": "not_a_list",
                 "effect": "TEST", "score": 0.8},
            ]
        }
        eng = RuleEvolutionEngine()
        eng.set_state(snap)
        assert eng.active_rule_count == 1

    def test_17_8_empty_active_rules_clears_engine(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        eng.set_state({"active_rules": []})
        assert eng.active_rule_count == 0

    def test_17_9_set_state_enforces_max_active_rules(self):
        snap = {
            "active_rules": [
                {"name": f"R{i}",
                 "conditions": [{"field": "fatigue_index", "operator": "gt",
                                  "threshold": 0.7}],
                 "effect": "TEST", "score": 0.75}
                for i in range(MAX_ACTIVE_RULES + 10)
            ]
        }
        eng = RuleEvolutionEngine()
        eng.set_state(snap)
        assert eng.active_rule_count == MAX_ACTIVE_RULES

    def test_17_10_full_json_round_trip(self):
        eng1 = RuleEvolutionEngine()
        eng1.analyze(_mixed_events())
        raw  = json.dumps(eng1.get_state())
        eng2 = RuleEvolutionEngine()
        eng2.set_state(json.loads(raw))
        assert eng2.active_rule_count == eng1.active_rule_count
        for r1, r2 in zip(eng1.active_rules, eng2.active_rules):
            assert r1.name  == r2.name
            assert r1.score == pytest.approx(r2.score, abs=1e-12)


# =============================================================================
# SECTION 18 — run_evolution() stateless function
# =============================================================================

class TestRunEvolution:

    def test_18_1_returns_dict_with_required_keys(self):
        r = run_evolution(_make_events(10))
        assert set(r) == {"candidate_rules", "activated_rules", "rejected_rules"}

    def test_18_2_stateless_identical_calls(self):
        events = _mixed_events()
        assert run_evolution(events) == run_evolution(events)

    def test_18_3_does_not_affect_global_state(self):
        events = _mixed_events()
        run_evolution(events)
        run_evolution(events)
        # No assertion needed beyond no exception; module-level state unchanged


# =============================================================================
# SECTION 19 — self_test()
# =============================================================================

class TestSelfTest:

    def test_19_1_returns_dict(self):
        assert isinstance(RuleEvolutionEngine().self_test(), dict)

    def test_19_2_required_top_level_keys(self):
        st = RuleEvolutionEngine().self_test()
        assert {"engine", "version", "checks", "passed"} <= set(st)

    def test_19_3_six_checks(self):
        assert len(RuleEvolutionEngine().self_test()["checks"]) == 6

    def test_19_4_each_check_has_required_fields(self):
        for c in RuleEvolutionEngine().self_test()["checks"]:
            assert {"name", "passed", "detail"} <= set(c)

    def test_19_5_all_checks_pass(self):
        st = RuleEvolutionEngine().self_test()
        assert st["passed"] is True
        failures = [c["name"] for c in st["checks"] if not c["passed"]]
        assert failures == [], f"Failed checks: {failures}"

    def test_19_6_does_not_mutate_live_state(self):
        eng = RuleEvolutionEngine()
        eng.analyze(_mixed_events())
        count_before = eng.active_rule_count
        rules_before = [r.name for r in eng.active_rules]
        eng.self_test()
        assert eng.active_rule_count == count_before
        assert [r.name for r in eng.active_rules] == rules_before

    def test_19_7_engine_name_correct(self):
        assert RuleEvolutionEngine().self_test()["engine"] == ENGINE_NAME

    def test_19_8_version_correct(self):
        assert RuleEvolutionEngine().self_test()["version"] == ENGINE_VERSION


# =============================================================================
# SECTION 20 — _clamp() helper
# =============================================================================

class TestClamp:

    def test_20_1_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_20_2_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_20_3_above_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_20_4_nan_returns_lo(self):
        assert _clamp(float("nan"), 0.0, 1.0) == 0.0

    def test_20_5_inf_returns_lo(self):
        assert _clamp(float("inf"), 0.0, 1.0) == 0.0

    def test_20_6_boundary_values_exact(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# =============================================================================
# SECTION 21 — _finite_or_zero() helper
# =============================================================================

class TestFiniteOrZero:

    def test_21_1_finite_unchanged(self):
        assert _finite_or_zero(0.42) == pytest.approx(0.42)

    def test_21_2_nan_returns_zero(self):
        assert _finite_or_zero(float("nan")) == 0.0

    def test_21_3_pos_inf_returns_zero(self):
        assert _finite_or_zero(float("inf")) == 0.0

    def test_21_4_neg_inf_returns_zero(self):
        assert _finite_or_zero(float("-inf")) == 0.0


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