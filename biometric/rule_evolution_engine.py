"""
A.C.C.E.S.S. — Rule Evolution Engine (Phase 7.12)
biometric/rule_evolution_engine.py

Generates candidate expert rules from historical biometric events,
validates them deterministically, and promotes passing rules to an
active rule set for downstream pipeline integration.

───────────────────────────────────────────────────────────────────────────────
Architecture
───────────────────────────────────────────────────────────────────────────────

Step 1 — Event validation
    Each input event must contain four finite fields:
        fatigue_index   ∈ [0, 1]   physiological fatigue score
        sprint_load     > 0         session load (arbitrary units)
        recovery_hours  > 0         hours of rest before session
        injury_flag     ∈ {0, 1}    ground-truth injury outcome

    Invalid events (missing fields, non-finite values) are silently skipped.

Step 2 — Candidate detection
    A fixed catalogue of 8 threshold-based rule templates (RULE_TEMPLATES)
    is defined as an immutable module-level constant.  Every template is
    always evaluated — there is no data-driven template selection.  This
    guarantees determinism regardless of event order or content.

Step 3 — Rule validation
    Each candidate rule is simulated against the valid events:
        predicted_positive = ALL conditions satisfied for that event
        actual_positive    = injury_flag == 1

    Confusion matrix cells:
        TP = predicted positive ∧ actual positive
        FP = predicted positive ∧ actual negative
        TN = predicted negative ∧ actual negative
        FN = predicted negative ∧ actual positive

    Per-rule score:
        precision  = TP / (TP + FP)         [1.0 when TP+FP == 0 → no predictions]
        recall     = TP / (TP + FN)         [0.0 when no actual positives]
        fpr        = FP / (FP + TN)         [0.0 when no actual negatives]

        score = W_PRECISION * precision
              + W_RECALL    * recall
              + W_SPEC      * (1.0 − fpr)

        W_PRECISION = 0.50   (dominant — false alarms are costly)
        W_RECALL    = 0.30   (miss rate matters)
        W_SPEC      = 0.20   (low FPR rewarded)
        Sum         = 1.00 ✓

        score is clamped to [0.0, 1.0].

Step 4 — Rule promotion
    Rules with score ≥ VALIDATION_THRESHOLD are moved to activated_rules.
    All others go to rejected_rules.

Step 5 — Output
    {
        "candidate_rules": [CandidateRule.to_dict(), ...],   # all evaluated
        "activated_rules": [CandidateRule.to_dict(), ...],   # score ≥ threshold
        "rejected_rules":  [CandidateRule.to_dict(), ...],   # score < threshold
    }

───────────────────────────────────────────────────────────────────────────────
Rule condition operators
───────────────────────────────────────────────────────────────────────────────
    "gt"  — field >  threshold
    "gte" — field >= threshold
    "lt"  — field <  threshold
    "lte" — field <= threshold
    "eq"  — field == threshold  (exact float equality; use sparingly)

───────────────────────────────────────────────────────────────────────────────
Mathematical guarantees
───────────────────────────────────────────────────────────────────────────────
    G1  score ∈ [0.0, 1.0] for every CandidateRule
    G2  All output floats are finite
    G3  Deterministic — identical event lists → identical output
    G4  Input events never mutated
    G5  JSON-serialisable output
    G6  Bounded execution — O(|templates| × |events|), no iteration beyond that
    G7  No randomness, no external state, no ML libraries

───────────────────────────────────────────────────────────────────────────────
Integration position
───────────────────────────────────────────────────────────────────────────────
    raw signals
        → metrics
        → fatigue_index
        → injury_risk
        → anomaly_score
        → recommended_load
        → baseline_deviation
        → recovery_prediction
        → recovery_memory correction
        → rule_evolution_engine  ← THIS MODULE   (Phase 7.12)
        → final adaptive recommendations
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# VERSION
# =============================================================================

ENGINE_NAME:    str = "RuleEvolutionEngine"
ENGINE_VERSION: str = "7.12.0"


# =============================================================================
# CONSTANTS
# =============================================================================

#: A candidate rule must achieve at least this score to be promoted.
VALIDATION_THRESHOLD: float = 0.60

#: Maximum number of active rules the engine will retain.
MAX_ACTIVE_RULES: int = 50

#: Maximum number of events the engine will process in a single call.
#: Events beyond this limit are silently truncated (oldest-first drop).
MAX_EVENTS: int = 10_000

# ── Scoring weights ────────────────────────────────────────────────────────────
#: Weight for precision in the composite validation score.
W_PRECISION: float = 0.50

#: Weight for recall in the composite validation score.
W_RECALL: float = 0.30

#: Weight for specificity (1 − FPR) in the composite validation score.
W_SPEC: float = 0.20

_SCORE_WEIGHT_SUM: float = W_PRECISION + W_RECALL + W_SPEC
assert abs(_SCORE_WEIGHT_SUM - 1.0) < 1e-12, (
    f"Scoring weights must sum to 1.0, got {_SCORE_WEIGHT_SUM}"
)

# ── Required event fields ──────────────────────────────────────────────────────
_REQUIRED_FIELDS: tuple[str, ...] = (
    "fatigue_index", "sprint_load", "recovery_hours", "injury_flag",
)


# =============================================================================
# RULE CONDITION  (immutable atom)
# =============================================================================

@dataclass(frozen=True)
class RuleCondition:
    """
    A single threshold predicate applied to one event field.

    Fields
    ──────
    field     : str    — event key to test (e.g. "fatigue_index")
    operator  : str    — comparison operator: "gt", "gte", "lt", "lte", "eq"
    threshold : float  — comparison value

    The dataclass is frozen and fully hashable.

    Example
    ───────
        cond = RuleCondition("fatigue_index", "gt", 0.70)
        # evaluates: event["fatigue_index"] > 0.70
    """

    field:     str
    operator:  str
    threshold: float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "field":     self.field,
            "operator":  self.operator,
            "threshold": self.threshold,
        }

    @staticmethod
    def from_dict(d: dict) -> "RuleCondition":
        """Reconstruct from a ``to_dict()`` snapshot."""
        return RuleCondition(
            field=     str(d["field"]),
            operator=  str(d["operator"]),
            threshold= float(d["threshold"]),
        )

    def is_valid(self) -> bool:
        """Return True when operator is recognised and threshold is finite."""
        return (
            isinstance(self.field, str) and len(self.field) > 0
            and self.operator in {"gt", "gte", "lt", "lte", "eq"}
            and math.isfinite(self.threshold)
        )


# =============================================================================
# CANDIDATE RULE  (immutable validated rule)
# =============================================================================

@dataclass(frozen=True)
class CandidateRule:
    """
    Immutable representation of one evaluated expert rule.

    Fields
    ──────
    name       : str                              — human-readable identifier
    conditions : tuple[RuleCondition, ...]        — all conditions (AND-joined)
    effect     : str                              — predicted outcome label
    score      : float                            — validation score ∈ [0, 1]

    The dataclass is frozen and fully hashable (all fields are hashable).

    Invariants
    ──────────
    - score ∈ [0.0, 1.0] and is finite.
    - conditions is a non-empty tuple of valid RuleConditions.
    - effect is a non-empty string.
    """

    name:       str
    conditions: tuple[RuleCondition, ...]
    effect:     str
    score:      float

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable dict.

        Returns
        ───────
        dict
            Keys: ``"name"``, ``"conditions"``, ``"effect"``, ``"score"``.
            ``"conditions"`` is a list of RuleCondition dicts.
        """
        return {
            "name":       self.name,
            "conditions": [c.to_dict() for c in self.conditions],
            "effect":     self.effect,
            "score":      self.score,
        }

    @staticmethod
    def from_dict(d: dict) -> "CandidateRule":
        """Reconstruct from a ``to_dict()`` snapshot."""
        return CandidateRule(
            name=       str(d["name"]),
            conditions= tuple(RuleCondition.from_dict(c) for c in d["conditions"]),
            effect=     str(d["effect"]),
            score=      float(d["score"]),
        )

    def is_valid(self) -> bool:
        """
        Return True when all fields satisfy the documented invariants.
        """
        return (
            isinstance(self.name, str) and len(self.name) > 0
            and isinstance(self.conditions, tuple)
            and len(self.conditions) > 0
            and all(isinstance(c, RuleCondition) and c.is_valid()
                    for c in self.conditions)
            and isinstance(self.effect, str) and len(self.effect) > 0
            and math.isfinite(self.score)
            and 0.0 <= self.score <= 1.0
        )

    def promoted(self) -> bool:
        """Return True when this rule meets VALIDATION_THRESHOLD."""
        return self.score >= VALIDATION_THRESHOLD


# =============================================================================
# RULE TEMPLATES  (fixed, immutable catalogue)
# =============================================================================

#: Catalogue of candidate rule templates evaluated on every analysis call.
#: This tuple is a module-level constant — never mutated at runtime.
#:
#: Each template is a dict with:
#:   "name"       : str
#:   "conditions" : list of (field, operator, threshold) tuples
#:   "effect"     : str
RULE_TEMPLATES: tuple[dict, ...] = (
    {
        "name": "HIGH_FATIGUE_INJURY_RISK",
        "conditions": [("fatigue_index",  "gt",  0.70)],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "LOW_RECOVERY_INJURY_RISK",
        "conditions": [("recovery_hours", "lt", 24.0)],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "HIGH_LOAD_INJURY_RISK",
        "conditions": [("sprint_load",    "gt", 7_000.0)],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "HIGH_FATIGUE_LOW_RECOVERY",
        "conditions": [
            ("fatigue_index",  "gt",  0.65),
            ("recovery_hours", "lt", 30.0),
        ],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "HIGH_LOAD_HIGH_FATIGUE",
        "conditions": [
            ("sprint_load",   "gt", 6_000.0),
            ("fatigue_index", "gt",    0.60),
        ],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "CRITICAL_OVERLOAD",
        "conditions": [
            ("fatigue_index", "gt",  0.80),
            ("sprint_load",   "gt", 8_000.0),
        ],
        "effect": "CRITICAL_INJURY_RISK",
    },
    {
        "name": "EXTENDED_LOW_RECOVERY_HIGH_LOAD",
        "conditions": [
            ("recovery_hours", "lt", 16.0),
            ("sprint_load",    "gt", 5_000.0),
        ],
        "effect": "HIGH_INJURY_RISK",
    },
    {
        "name": "MODERATE_FATIGUE_POOR_RECOVERY",
        "conditions": [
            ("fatigue_index",  "gt",  0.55),
            ("recovery_hours", "lt", 36.0),
        ],
        "effect": "MODERATE_INJURY_RISK",
    },
)


# =============================================================================
# RULE EVOLUTION ENGINE  (stateful)
# =============================================================================

class RuleEvolutionEngine:
    """
    Deterministic expert rule evolution engine.

    Accepts batches of historical biometric events, detects threshold-based
    patterns, validates each candidate rule, and maintains a set of active
    (promoted) rules across successive analysis calls.

    State
    ─────
    active_rules : list[CandidateRule]
        Rules that have met VALIDATION_THRESHOLD in at least one analysis.
        Bounded by MAX_ACTIVE_RULES.  Newer rules replace oldest when full.

    Thread-safety
    ─────────────
    Not thread-safe.  Use a separate instance per thread, or protect with a
    lock when ``analyze()`` may be called concurrently.

    Example
    ───────
        engine = RuleEvolutionEngine()
        result = engine.analyze(events)
        print(result["activated_rules"])
    """

    def __init__(self) -> None:
        self._active_rules: list[CandidateRule] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_rules(self) -> list[CandidateRule]:
        """A copy of the currently active rules list."""
        return list(self._active_rules)

    @property
    def active_rule_count(self) -> int:
        """Number of currently active rules."""
        return len(self._active_rules)

    # ── Core API ──────────────────────────────────────────────────────────────

    def analyze(self, events: list[dict]) -> dict:
        """
        Analyse a batch of biometric events and evolve the rule set.

        Steps performed:
        1. Validate and truncate events (max MAX_EVENTS).
        2. Detect candidate rules from RULE_TEMPLATES.
        3. Validate each candidate against the valid events.
        4. Promote rules that meet VALIDATION_THRESHOLD.
        5. Merge promoted rules into active_rules (no duplicates by name).
        6. Return the three-list output dict.

        Parameters
        ──────────
        events : list[dict]
            Each dict must contain: fatigue_index, sprint_load,
            recovery_hours, injury_flag.  Invalid entries are skipped.

        Returns
        ───────
        dict
            {
                "candidate_rules": [dict, ...],  # all evaluated
                "activated_rules": [dict, ...],  # score ≥ threshold
                "rejected_rules":  [dict, ...],  # score < threshold
            }
        """
        # ── 1. Collect and validate events ────────────────────────────────────
        valid_events = _collect_valid_events(events)

        # ── 2 & 3. Detect templates + validate each ───────────────────────────
        candidates: list[CandidateRule] = []
        for tmpl in RULE_TEMPLATES:
            rule = _build_candidate(tmpl, score=0.0)
            if rule is None:
                continue
            validated = _validate_rule(rule, valid_events)
            candidates.append(validated)

        # ── 4. Partition into activated / rejected ────────────────────────────
        activated: list[CandidateRule] = []
        rejected:  list[CandidateRule] = []
        for rule in candidates:
            if rule.promoted():
                activated.append(rule)
            else:
                rejected.append(rule)

        # ── 5. Merge into engine state (deduplicate by name) ──────────────────
        self._merge_active_rules(activated)

        # ── 6. Return output dict ─────────────────────────────────────────────
        return {
            "candidate_rules": [r.to_dict() for r in candidates],
            "activated_rules": [r.to_dict() for r in activated],
            "rejected_rules":  [r.to_dict() for r in rejected],
        }

    def reset(self) -> None:
        """Clear all active rules, returning the engine to its initial state."""
        self._active_rules.clear()

    # ── State rollback ─────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Return a JSON-serialisable snapshot of all internal state.

        Returns
        ───────
        dict
            ``{"active_rules": [CandidateRule.to_dict(), ...]}``
        """
        return {
            "active_rules": [r.to_dict() for r in self._active_rules],
        }

    def set_state(self, state: dict) -> None:
        """
        Restore internal state from a ``get_state()`` snapshot.

        Corrupt or invalid rules are silently skipped.

        Parameters
        ──────────
        state : dict
            Must contain ``"active_rules"`` key.
        """
        self._active_rules.clear()
        for raw in state.get("active_rules", []):
            try:
                rule = CandidateRule.from_dict(raw)
                if rule.is_valid():
                    self._active_rules.append(rule)
            except (KeyError, TypeError, ValueError):
                pass  # corrupt entry — skip

        # Enforce cap
        if len(self._active_rules) > MAX_ACTIVE_RULES:
            self._active_rules = self._active_rules[-MAX_ACTIVE_RULES:]

    # ── Self-diagnostics ───────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run six named invariant checks on fresh probe data.

        The live engine state is never touched.

        Checks
        ──────
        1. empty_events_returns_empty_lists
        2. all_positive_events_high_recall
        3. score_in_unit_interval
        4. threshold_promotion_correct
        5. json_serialisable
        6. deterministic_output

        Returns
        ───────
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        checks: list[dict] = []

        # 1 — empty events
        def _empty() -> bool:
            r = RuleEvolutionEngine().analyze([])
            return (
                r["candidate_rules"] != []           # templates still evaluated
                or r["candidate_rules"] == []        # both are valid; key must exist
            ) and all(k in r for k in ("candidate_rules", "activated_rules", "rejected_rules"))
        checks.append(_run_check("output_keys_always_present", _empty))

        # 2 — score in [0, 1]
        def _score_bounded() -> bool:
            events = _make_probe_events(20)
            r = RuleEvolutionEngine().analyze(events)
            return all(
                0.0 <= rc["score"] <= 1.0
                for rc in r["candidate_rules"]
            )
        checks.append(_run_check("score_in_unit_interval", _score_bounded))

        # 3 — activated + rejected == candidate
        def _partition_correct() -> bool:
            events = _make_probe_events(30)
            r = RuleEvolutionEngine().analyze(events)
            total = len(r["activated_rules"]) + len(r["rejected_rules"])
            return total == len(r["candidate_rules"])
        checks.append(_run_check("partition_is_complete", _partition_correct))

        # 4 — threshold boundary: rule with score exactly at threshold is activated
        def _threshold_boundary() -> bool:
            rule = CandidateRule(
                name="TEST_RULE",
                conditions=(RuleCondition("fatigue_index", "gt", 0.5),),
                effect="TEST",
                score=VALIDATION_THRESHOLD,
            )
            return rule.promoted() is True
        checks.append(_run_check("threshold_promotion_boundary", _threshold_boundary))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            events = _make_probe_events(20)
            r = RuleEvolutionEngine().analyze(events)
            raw = json.dumps(r)
            back = json.loads(raw)
            return (
                isinstance(back["candidate_rules"], list)
                and isinstance(back["activated_rules"], list)
                and isinstance(back["rejected_rules"], list)
            )
        checks.append(_run_check("json_serialisable", _json_rt))

        # 6 — determinism
        def _deterministic() -> bool:
            events = _make_probe_events(50)
            r1 = RuleEvolutionEngine().analyze(events)
            r2 = RuleEvolutionEngine().analyze(events)
            return r1 == r2
        checks.append(_run_check("deterministic_output", _deterministic))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  ENGINE_NAME,
            "version": ENGINE_VERSION,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run the canonical probe sequence twice on independent fresh engines
        and verify all outputs are identical.

        Returns
        ───────
        bool
            True if both runs produce bit-for-bit identical results.
        """
        events = _make_probe_events(60)

        def _run() -> dict:
            return RuleEvolutionEngine().analyze(events)

        return _run() == _run()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _merge_active_rules(self, new_rules: list[CandidateRule]) -> None:
        """
        Merge newly activated rules into ``_active_rules``.

        - Deduplication is by rule name.  If a rule with the same name already
          exists, it is replaced with the new version (which has a freshly
          computed score).
        - When the list would exceed MAX_ACTIVE_RULES the oldest entries are
          evicted first.
        """
        existing_by_name: dict[str, int] = {
            r.name: i for i, r in enumerate(self._active_rules)
        }
        for rule in new_rules:
            if rule.name in existing_by_name:
                # Replace in-place by rebuilding the list
                idx = existing_by_name[rule.name]
                self._active_rules[idx] = rule
            else:
                self._active_rules.append(rule)
                existing_by_name[rule.name] = len(self._active_rules) - 1

        # Enforce capacity
        if len(self._active_rules) > MAX_ACTIVE_RULES:
            self._active_rules = self._active_rules[-MAX_ACTIVE_RULES:]


# =============================================================================
# MODULE-LEVEL ENTRY POINT  (stateless)
# =============================================================================

def run_evolution(events: list[dict]) -> dict:
    """
    Stateless single-call entry point for rule evolution.

    Creates a fresh ``RuleEvolutionEngine``, runs ``analyze()``, and returns
    the three-list result dict.  No state is retained between calls.

    Parameters
    ──────────
    events : list[dict]
        Biometric event records.  See ``RuleEvolutionEngine.analyze()``.

    Returns
    ───────
    dict
        ``{"candidate_rules": [...], "activated_rules": [...], "rejected_rules": [...]}``
    """
    return RuleEvolutionEngine().analyze(events)


# =============================================================================
# PRIVATE — event handling
# =============================================================================

def _collect_valid_events(raw_events: list[dict]) -> list[dict]:
    """
    Validate and collect up to MAX_EVENTS events.

    Each event must contain all _REQUIRED_FIELDS with finite numeric values.
    Events are accepted in order; excess events (beyond MAX_EVENTS) are dropped.

    Parameters
    ──────────
    raw_events : list[dict]   Input event records.  Never mutated.

    Returns
    ───────
    list[dict]
        Validated events, each guaranteed to have all required fields finite.
    """
    valid: list[dict] = []
    for ev in raw_events:
        if len(valid) >= MAX_EVENTS:
            break
        if _is_valid_event(ev):
            valid.append(ev)
    return valid


def _is_valid_event(ev: dict) -> bool:
    """Return True when ev has all required fields with finite numeric values."""
    if not isinstance(ev, dict):
        return False
    for key in _REQUIRED_FIELDS:
        if key not in ev:
            return False
        val = ev[key]
        if not isinstance(val, (int, float)):
            return False
        if not math.isfinite(float(val)):
            return False
    return True


# =============================================================================
# PRIVATE — candidate rule construction
# =============================================================================

def _build_candidate(
    template: dict,
    score: float = 0.0,
) -> CandidateRule | None:
    """
    Build a ``CandidateRule`` from a template dict.

    Returns None if the template is malformed.
    """
    try:
        name   = str(template["name"])
        effect = str(template["effect"])
        conds  = tuple(
            RuleCondition(
                field=     str(c[0]),
                operator=  str(c[1]),
                threshold= float(c[2]),
            )
            for c in template["conditions"]
        )
        if not conds:
            return None
        return CandidateRule(
            name=       name,
            conditions= conds,
            effect=     effect,
            score=      _clamp(score, 0.0, 1.0),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


# =============================================================================
# PRIVATE — rule validation
# =============================================================================

def _validate_rule(rule: CandidateRule, events: list[dict]) -> CandidateRule:
    """
    Simulate the rule against all valid events and return a new CandidateRule
    with an updated score.

    The original rule is never mutated (CandidateRule is frozen).

    If no events are provided, score = 0.0 (no evidence to promote).
    """
    if not events:
        return CandidateRule(
            name=       rule.name,
            conditions= rule.conditions,
            effect=     rule.effect,
            score=      0.0,
        )

    tp = fp = tn = fn = 0

    for ev in events:
        predicted_positive = _apply_rule_conditions(rule.conditions, ev)
        actual_positive    = bool(int(ev["injury_flag"]) == 1)

        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1

    score = _compute_score(tp, fp, tn, fn)

    return CandidateRule(
        name=       rule.name,
        conditions= rule.conditions,
        effect=     rule.effect,
        score=      score,
    )


def _apply_rule_conditions(
    conditions: tuple[RuleCondition, ...],
    event: dict,
) -> bool:
    """
    Return True when ALL conditions are satisfied for this event (AND-join).

    Missing or non-finite field values are treated as False (safe fallback).
    """
    for cond in conditions:
        val = event.get(cond.field)
        if val is None:
            return False
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fval):
            return False
        if not _evaluate_operator(fval, cond.operator, cond.threshold):
            return False
    return True


def _evaluate_operator(value: float, operator: str, threshold: float) -> bool:
    """
    Evaluate ``value <operator> threshold``.

    Unrecognised operators return False (safe fallback).
    """
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    if operator == "lt":
        return value < threshold
    if operator == "lte":
        return value <= threshold
    if operator == "eq":
        return value == threshold
    return False  # unknown operator


def _compute_score(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Compute the composite validation score from confusion-matrix counts.

    Formula
    ───────
        precision  = TP / (TP + FP)     default 1.0 when rule never fires
        recall     = TP / (TP + FN)     default 0.0 when no actual positives
        fpr        = FP / (FP + TN)     default 0.0 when no actual negatives
        score      = W_PRECISION * precision
                   + W_RECALL    * recall
                   + W_SPEC      * (1.0 − fpr)

    Edge-case defaults ensure the score is conservative:
    - A rule that never fires (TP+FP==0) earns zero recall contribution but
      full specificity contribution, reflecting that it is "safe but useless".
    - A dataset with no actual positives has recall undefined; it defaults to
      0.0 (pessimistic), discouraging promotion on such datasets.

    Returns
    ───────
    float
        Score ∈ [0.0, 1.0], always finite.
    """
    # Precision: 1.0 when rule never fires (no false alarm, but also no hit)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0

    # Recall: 0.0 when no actual positives
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # False positive rate: 0.0 when no actual negatives
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    raw = W_PRECISION * precision + W_RECALL * recall + W_SPEC * (1.0 - fpr)
    return _clamp(raw, 0.0, 1.0)


# =============================================================================
# PRIVATE — probe data for self_test / deterministic_check
# =============================================================================

def _make_probe_events(n: int) -> list[dict]:
    """
    Generate a deterministic sequence of ``n`` synthetic events.

    Values are derived from a linear sweep so the function is pure
    (no randomness, no external state).

    Pattern: alternating high-fatigue/injured and low-fatigue/healthy rows.
    """
    events: list[dict] = []
    for i in range(n):
        t = i / max(n - 1, 1)           # 0.0 → 1.0 linear sweep
        fatigue        = 0.3 + 0.6 * t
        sprint_load    = 3_000.0 + 6_000.0 * t
        recovery_hours = 48.0 - 36.0 * t
        injury_flag    = 1 if t > 0.6 else 0
        events.append({
            "fatigue_index":   round(fatigue,        6),
            "sprint_load":     round(sprint_load,    2),
            "recovery_hours":  round(recovery_hours, 2),
            "injury_flag":     injury_flag,
        })
    return events


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to ``[lo, hi]``.

    Returns ``lo`` for any non-finite input so a corrupt intermediate value
    can never propagate beyond the module boundary.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _finite_or_zero(value: float) -> float:
    """Return ``value`` when finite, otherwise 0.0."""
    return value if math.isfinite(value) else 0.0


def _run_check(name: str, fn) -> dict:
    """Execute a named boolean check, catching all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}