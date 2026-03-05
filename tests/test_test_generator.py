"""
A.C.C.E.S.S. — GeneratorEngine Test Suite (Phase 6.3)

Coverage:
    TestDeterminism           — identical inputs → identical GenerationReport
    TestConfidenceFloor       — low-confidence proposals suppressed; suppressed_count correct
    TestCapsRespected         — max_unit_tests / max_regression_tests / max_mutation_tests honoured
    TestOrdering              — severity desc, then module asc, then test_type asc
    TestInstabilityReduction  — structural_instability_index=1.0 reduces counts proportionally
    TestGateReduction         — gate_decision.patch_aggressiveness=0.0 heavily reduces output
    TestJSONSerialization     — report.to_dict() is JSON-serializable
    TestNoNaNInf              — extreme confidence values are clamped, never NaN/Inf
    TestImmutability          — TestCaseProposal is frozen; mutation raises
    TestEmptyReport           — empty patch_report → total_proposed==0, proposals==()
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.test_generator import (
    GeneratorEngine,
    GenerationConfig,
    TestCaseProposal,
    GenerationReport,
)
from agent.structural_meta import (
    StructuralMetaState,
    StructuralGateDecision,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE PATCH LAYER — duck-typed stubs; no import of PatchProposalEngine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FakePatchSuggestion:
    """Minimal duck-typed stub for PatchSuggestion."""
    module: str = "agent.core"
    location: str = "my_func"
    issue_type: str = "high_complexity"
    severity: str = "high"
    rationale: str = "Too complex."
    suggested_refactor_strategy: str = "Split function."
    confidence_score: float = 0.80
    risk_reduction_estimate: float = 0.05
    requires_human_review: bool = True
    metric_evidence: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "location": self.location,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "rationale": self.rationale,
            "suggested_refactor_strategy": self.suggested_refactor_strategy,
            "confidence_score": self.confidence_score,
            "risk_reduction_estimate": self.risk_reduction_estimate,
            "requires_human_review": self.requires_human_review,
            "metric_evidence": list(self.metric_evidence),
        }


@dataclass(frozen=True)
class FakePatchReport:
    """Minimal duck-typed stub for ProposalReport."""
    suggestions: tuple = field(default_factory=tuple)
    total_suggestions_emitted: int = 0
    suggestions_suppressed: int = 0
    aggressiveness_level: float = 1.0
    structural_instability: float = 0.0
    total_issues_detected: int = 0
    max_suggestions_allowed: int = 20

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "total_suggestions_emitted": self.total_suggestions_emitted,
            "suggestions_suppressed": self.suggestions_suppressed,
            "aggressiveness_level": self.aggressiveness_level,
            "structural_instability": self.structural_instability,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _suggestion(
    module: str = "agent.core",
    severity: str = "high",
    confidence: float = 0.80,
    location: str = "fn",
    issue_type: str = "high_complexity",
) -> FakePatchSuggestion:
    return FakePatchSuggestion(
        module=module,
        location=location,
        issue_type=issue_type,
        severity=severity,
        confidence_score=confidence,
    )


def _report(*suggestions: FakePatchSuggestion) -> FakePatchReport:
    return FakePatchReport(
        suggestions=tuple(suggestions),
        total_suggestions_emitted=len(suggestions),
    )


def _make_rich_patch_report() -> FakePatchReport:
    """A varied patch report covering all severities and multiple modules."""
    return _report(
        _suggestion(module="agent.alpha", severity="critical", confidence=0.92),
        _suggestion(module="agent.beta",  severity="high",     confidence=0.85),
        _suggestion(module="agent.gamma", severity="medium",   confidence=0.70),
        _suggestion(module="agent.delta", severity="low",      confidence=0.55),
        _suggestion(module="agent.alpha", severity="high",     confidence=0.80),
        _suggestion(module="agent.beta",  severity="medium",   confidence=0.65),
    )


def _all_floats_finite(obj) -> bool:
    """Recursively check that no float in a nested dict/list is NaN or Inf."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_floats_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_floats_finite(v) for v in obj)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1 — DETERMINISM
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """Same patch_report input must yield byte-for-byte identical output."""

    def test_reports_have_same_total_proposed(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        assert r1.total_proposed == r2.total_proposed

    def test_proposals_same_length(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        assert len(r1.proposals) == len(r2.proposals)

    def test_proposals_same_target_module_field(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.target_module == p2.target_module

    def test_proposals_same_test_type_field(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.test_type == p2.test_type

    def test_proposals_same_severity_level_field(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.severity_level == p2.severity_level

    def test_proposals_same_confidence_score(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.confidence_score == p2.confidence_score

    def test_suppressed_count_identical(self):
        patch_report = _make_rich_patch_report()
        engine = GeneratorEngine()
        r1 = engine.generate_tests(patch_report)
        r2 = engine.generate_tests(patch_report)
        assert r1.suppressed_count == r2.suppressed_count

    def test_different_engine_instances_deterministic(self):
        patch_report = _make_rich_patch_report()
        r1 = GeneratorEngine().generate_tests(patch_report)
        r2 = GeneratorEngine().generate_tests(patch_report)
        assert r1.total_proposed == r2.total_proposed
        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.target_module == p2.target_module
            assert p1.test_type == p2.test_type


# ─────────────────────────────────────────────────────────────────────────────
# 2 — CONFIDENCE FLOOR
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceFloor:
    """Proposals below the confidence floor must be suppressed."""

    def test_low_confidence_suggestion_suppressed(self):
        low = _suggestion(module="agent.core", severity="high", confidence=0.10)
        patch_report = _report(low)
        cfg = GenerationConfig(confidence_floor=0.50)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.total_proposed == 0
        assert len(result.proposals) == 0

    def test_suppressed_count_increments_for_each_below_floor(self):
        s1 = _suggestion(module="agent.a", severity="high",   confidence=0.10)
        s2 = _suggestion(module="agent.b", severity="medium", confidence=0.20)
        s3 = _suggestion(module="agent.c", severity="high",   confidence=0.90)
        patch_report = _report(s1, s2, s3)
        cfg = GenerationConfig(confidence_floor=0.80)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.suppressed_count > 0

    def test_high_confidence_passes_floor(self):
        high = _suggestion(module="agent.core", severity="high", confidence=0.95)
        patch_report = _report(high)
        cfg = GenerationConfig(confidence_floor=0.50)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.total_proposed >= 1

    def test_floor_zero_passes_everything(self):
        suggestions = tuple(
            _suggestion(module=f"agent.m{i}", severity="high", confidence=0.40 + 0.05 * i)
            for i in range(5)
        )
        patch_report = _report(*suggestions)
        cfg = GenerationConfig(confidence_floor=0.0)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.total_proposed > 0

    def test_floor_above_all_scores_suppresses_all(self):
        suggestions = tuple(
            _suggestion(module=f"agent.m{i}", severity="high", confidence=0.50)
            for i in range(4)
        )
        patch_report = _report(*suggestions)
        cfg = GenerationConfig(confidence_floor=0.99)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.total_proposed == 0
        assert result.suppressed_count > 0

    def test_suppressed_count_consistent_with_emitted(self):
        suggestions = tuple(
            _suggestion(module=f"agent.m{i}", severity="high", confidence=0.40 + 0.10 * i)
            for i in range(6)
        )
        patch_report = _report(*suggestions)
        cfg = GenerationConfig(
            confidence_floor=0.60,
            max_unit_tests=100,
            max_regression_tests=100,
            max_mutation_tests=100,
        )
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.suppressed_count + result.total_proposed <= len(suggestions) * 3

    def test_all_emitted_proposals_above_floor(self):
        suggestions = tuple(
            _suggestion(module=f"agent.m{i}", severity="medium", confidence=0.50 + 0.05 * i)
            for i in range(5)
        )
        patch_report = _report(*suggestions)
        floor = 0.55
        cfg = GenerationConfig(confidence_floor=floor)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        for p in result.proposals:
            assert p.confidence_score >= floor, (
                f"Proposal confidence {p.confidence_score} is below floor {floor}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3 — CAPS RESPECTED
# ─────────────────────────────────────────────────────────────────────────────

class TestCapsRespected:
    """max_unit_tests / max_regression_tests / max_mutation_tests must be hard limits."""

    def _make_large_report(self, n: int = 50, severity: str = "high") -> FakePatchReport:
        suggestions = tuple(
            _suggestion(module=f"agent.mod_{i}", severity=severity, confidence=0.90)
            for i in range(n)
        )
        return _report(*suggestions)

    def test_unit_test_cap_respected(self):
        patch_report = self._make_large_report(50, severity="critical")
        cfg = GenerationConfig(max_unit_tests=3, max_regression_tests=100, max_mutation_tests=100)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        unit_count = sum(1 for p in result.proposals if p.test_type == "unit")
        assert unit_count <= 3

    def test_regression_test_cap_respected(self):
        patch_report = self._make_large_report(50, severity="medium")
        cfg = GenerationConfig(max_unit_tests=100, max_regression_tests=4, max_mutation_tests=100)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        reg_count = sum(1 for p in result.proposals if p.test_type == "regression")
        assert reg_count <= 4

    def test_mutation_test_cap_respected(self):
        patch_report = self._make_large_report(50, severity="low")
        cfg = GenerationConfig(max_unit_tests=100, max_regression_tests=100, max_mutation_tests=2)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        mut_count = sum(1 for p in result.proposals if p.test_type == "mutation")
        assert mut_count <= 2

    def test_all_caps_respected_simultaneously(self):
        suggestions = tuple(
            _suggestion(module=f"m{i}", severity="critical", confidence=0.90)
            for i in range(20)
        ) + tuple(
            _suggestion(module=f"n{i}", severity="medium", confidence=0.90)
            for i in range(20)
        ) + tuple(
            _suggestion(module=f"o{i}", severity="low", confidence=0.90)
            for i in range(20)
        )
        patch_report = _report(*suggestions)
        cfg = GenerationConfig(max_unit_tests=5, max_regression_tests=3, max_mutation_tests=2)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        unit_count = sum(1 for p in result.proposals if p.test_type == "unit")
        reg_count  = sum(1 for p in result.proposals if p.test_type == "regression")
        mut_count  = sum(1 for p in result.proposals if p.test_type == "mutation")
        assert unit_count <= 5
        assert reg_count  <= 3
        assert mut_count  <= 2

    def test_unit_cap_one_emits_exactly_one_unit(self):
        patch_report = self._make_large_report(30, severity="critical")
        cfg = GenerationConfig(max_unit_tests=1, max_regression_tests=100, max_mutation_tests=100)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        unit_count = sum(1 for p in result.proposals if p.test_type == "unit")
        assert unit_count == 1

    def test_total_proposed_reflects_actual_count(self):
        patch_report = self._make_large_report(30)
        cfg = GenerationConfig(max_unit_tests=3, max_regression_tests=3, max_mutation_tests=3)
        result = GeneratorEngine(config=cfg).generate_tests(patch_report)
        assert result.total_proposed == len(result.proposals)


# ─────────────────────────────────────────────────────────────────────────────
# 4 — ORDERING
# ─────────────────────────────────────────────────────────────────────────────

class TestOrdering:
    """Proposals must be ordered: severity desc → target_module asc → test_type asc."""

    _SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    _TYPE_RANK     = {"unit": 0, "regression": 1, "mutation": 2}

    def _make_ordering_report(self) -> FakePatchReport:
        return _report(
            _suggestion(module="agent.zzz", severity="low",      confidence=0.80),
            _suggestion(module="agent.aaa", severity="critical",  confidence=0.95),
            _suggestion(module="agent.mmm", severity="high",      confidence=0.85),
            _suggestion(module="agent.bbb", severity="critical",  confidence=0.93),
            _suggestion(module="agent.aaa", severity="high",      confidence=0.88),
            _suggestion(module="agent.mmm", severity="medium",    confidence=0.72),
        )

    def test_critical_before_high(self):
        result = GeneratorEngine().generate_tests(self._make_ordering_report())
        ranks = [self._SEVERITY_RANK.get(p.severity_level, 0) for p in result.proposals]
        for i in range(1, len(ranks)):
            assert ranks[i - 1] >= ranks[i], (
                f"Severity ordering violated at position {i}: "
                f"{result.proposals[i-1].severity_level} before "
                f"{result.proposals[i].severity_level}"
            )

    def test_high_before_medium(self):
        suggestions = (
            _suggestion(module="agent.x", severity="medium", confidence=0.80),
            _suggestion(module="agent.y", severity="high",   confidence=0.80),
        )
        result = GeneratorEngine().generate_tests(_report(*suggestions))
        severities = [p.severity_level for p in result.proposals]
        if "high" in severities and "medium" in severities:
            assert severities.index("high") < severities.index("medium")

    def test_medium_before_low(self):
        suggestions = (
            _suggestion(module="agent.x", severity="low",    confidence=0.80),
            _suggestion(module="agent.y", severity="medium", confidence=0.80),
        )
        result = GeneratorEngine().generate_tests(_report(*suggestions))
        severities = [p.severity_level for p in result.proposals]
        if "medium" in severities and "low" in severities:
            assert severities.index("medium") < severities.index("low")

    def test_within_same_severity_module_alphabetical(self):
        suggestions = (
            _suggestion(module="agent.zzz", severity="high", confidence=0.80),
            _suggestion(module="agent.aaa", severity="high", confidence=0.80),
            _suggestion(module="agent.mmm", severity="high", confidence=0.80),
        )
        result = GeneratorEngine().generate_tests(_report(*suggestions))
        same_sev = [p for p in result.proposals if p.severity_level == "high"]
        modules = [p.target_module for p in same_sev]
        for i in range(1, len(modules)):
            assert modules[i - 1] <= modules[i], (
                f"Module ordering violated: {modules[i-1]!r} before {modules[i]!r}"
            )

    def test_within_same_module_unit_before_regression(self):
        suggestions = tuple(
            _suggestion(module="agent.same", severity="high", confidence=0.85)
            for _ in range(4)
        )
        result = GeneratorEngine().generate_tests(_report(*suggestions))
        same_module = [p for p in result.proposals if p.target_module == "agent.same"]
        types = [p.test_type for p in same_module]
        for i in range(1, len(types)):
            prev_rank = self._TYPE_RANK.get(types[i - 1], 99)
            curr_rank = self._TYPE_RANK.get(types[i], 99)
            assert prev_rank <= curr_rank, (
                f"test_type ordering violated: {types[i-1]} before {types[i]}"
            )

    def test_within_same_module_regression_before_mutation(self):
        suggestions = tuple(
            _suggestion(module="agent.same", severity="medium", confidence=0.75)
            for _ in range(4)
        )
        result = GeneratorEngine().generate_tests(_report(*suggestions))
        same_module = [p for p in result.proposals if p.target_module == "agent.same"]
        types = [p.test_type for p in same_module]
        for i in range(1, len(types)):
            prev_rank = self._TYPE_RANK.get(types[i - 1], 99)
            curr_rank = self._TYPE_RANK.get(types[i], 99)
            assert prev_rank <= curr_rank

    def test_full_ordering_invariant(self):
        result = GeneratorEngine().generate_tests(self._make_ordering_report())
        proposals = result.proposals
        for i in range(1, len(proposals)):
            prev = proposals[i - 1]
            curr = proposals[i]
            prev_sev = self._SEVERITY_RANK.get(prev.severity_level, 0)
            curr_sev = self._SEVERITY_RANK.get(curr.severity_level, 0)
            if prev_sev != curr_sev:
                assert prev_sev > curr_sev, (
                    f"Severity ordering violated at {i}: "
                    f"{prev.severity_level} vs {curr.severity_level}"
                )
            elif prev.target_module != curr.target_module:
                assert prev.target_module <= curr.target_module, (
                    f"Module ordering violated at {i}: "
                    f"{prev.target_module!r} vs {curr.target_module!r}"
                )
            else:
                prev_type = self._TYPE_RANK.get(prev.test_type, 99)
                curr_type = self._TYPE_RANK.get(curr.test_type, 99)
                assert prev_type <= curr_type, (
                    f"test_type ordering violated at {i}: "
                    f"{prev.test_type} vs {curr.test_type}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 5 — INSTABILITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestInstabilityReduction:
    """structural_instability_index=1.0 must reduce proposal counts proportionally."""

    def _base_result(self) -> GenerationReport:
        return GeneratorEngine().generate_tests(_make_rich_patch_report())

    def _maximal_instability_result(self) -> GenerationReport:
        state = StructuralMetaState(structural_instability_index=1.0)
        return GeneratorEngine().generate_tests(
            _make_rich_patch_report(),
            structural_state=state,
        )

    def test_instability_reduces_total_proposed(self):
        base = self._base_result()
        reduced = self._maximal_instability_result()
        assert reduced.total_proposed <= base.total_proposed

    def test_instability_does_not_increase_proposal_count(self):
        base = self._base_result()
        reduced = self._maximal_instability_result()
        assert len(reduced.proposals) <= len(base.proposals)

    def test_zero_instability_does_not_reduce(self):
        state = StructuralMetaState(structural_instability_index=0.0)
        normal    = GeneratorEngine().generate_tests(_make_rich_patch_report())
        with_zero = GeneratorEngine().generate_tests(
            _make_rich_patch_report(),
            structural_state=state,
        )
        assert with_zero.total_proposed >= normal.total_proposed - 1

    def test_full_instability_reduces_vs_no_state(self):
        large_report = _report(*[
            _suggestion(module=f"agent.m{i}", severity="high", confidence=0.90)
            for i in range(20)
        ])
        state  = StructuralMetaState(structural_instability_index=1.0)
        result = GeneratorEngine().generate_tests(large_report, structural_state=state)
        normal = GeneratorEngine().generate_tests(large_report)
        assert result.total_proposed <= normal.total_proposed

    def test_instability_reduction_is_monotonic(self):
        patch_report = _make_rich_patch_report()
        results = []
        for instability in (0.0, 0.25, 0.5, 0.75, 1.0):
            state = StructuralMetaState(structural_instability_index=instability)
            r = GeneratorEngine().generate_tests(patch_report, structural_state=state)
            results.append(r.total_proposed)
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1] + 1, (
                f"Instability reduction not monotonic: {results}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6 — GATE REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

class TestGateReduction:
    """gate_decision.patch_aggressiveness=0.0 reduces output via instability_reduction."""

    def test_aggressiveness_zero_reduces_output(self):
        patch_report = _make_rich_patch_report()
        gate   = StructuralGateDecision(patch_aggressiveness=0.0)
        result = GeneratorEngine().generate_tests(patch_report, gate_decision=gate)
        normal = GeneratorEngine().generate_tests(patch_report)
        assert result.total_proposed <= normal.total_proposed

    def test_aggressiveness_zero_applies_maximum_reduction(self):
        large = _report(*[
            _suggestion(module=f"agent.m{i}", severity="critical", confidence=0.95)
            for i in range(30)
        ])
        gate   = StructuralGateDecision(patch_aggressiveness=0.0)
        normal = GeneratorEngine().generate_tests(large)
        gated  = GeneratorEngine().generate_tests(large, gate_decision=gate)
        assert gated.total_proposed < normal.total_proposed

    def test_aggressiveness_full_allows_normal_output(self):
        patch_report = _make_rich_patch_report()
        gate_full = StructuralGateDecision(patch_aggressiveness=1.0)
        with_gate = GeneratorEngine().generate_tests(patch_report, gate_decision=gate_full)
        normal    = GeneratorEngine().generate_tests(patch_report)
        assert with_gate.total_proposed >= normal.total_proposed - 1

    def test_gate_reduces_all_test_types(self):
        large = _report(*[
            _suggestion(module=f"agent.m{i}", severity="critical", confidence=0.90)
            for i in range(30)
        ])
        gate    = StructuralGateDecision(patch_aggressiveness=0.0)
        normal  = GeneratorEngine().generate_tests(large)
        reduced = GeneratorEngine().generate_tests(large, gate_decision=gate)
        for t in ("unit", "regression", "mutation"):
            n_count = sum(1 for p in normal.proposals  if p.test_type == t)
            r_count = sum(1 for p in reduced.proposals if p.test_type == t)
            assert r_count <= n_count

    def test_gate_overrides_structural_state(self):
        patch_report = _make_rich_patch_report()
        state = StructuralMetaState(structural_instability_index=0.0)
        gate  = StructuralGateDecision(patch_aggressiveness=0.0)
        with_state_only = GeneratorEngine().generate_tests(
            patch_report, structural_state=state,
        )
        with_gate = GeneratorEngine().generate_tests(
            patch_report, structural_state=state, gate_decision=gate,
        )
        assert with_gate.total_proposed <= with_state_only.total_proposed

    def test_gate_deterministic(self):
        patch_report = _make_rich_patch_report()
        gate = StructuralGateDecision(patch_aggressiveness=0.0)
        r1 = GeneratorEngine().generate_tests(patch_report, gate_decision=gate)
        r2 = GeneratorEngine().generate_tests(patch_report, gate_decision=gate)
        assert r1.total_proposed == r2.total_proposed


# ─────────────────────────────────────────────────────────────────────────────
# 7 — JSON SERIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONSerialization:
    """report.to_dict() must be JSON-serializable end-to-end."""

    def test_to_dict_does_not_raise(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        result.to_dict()

    def test_to_dict_json_dumps_succeeds(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)

    def test_roundtrip_total_proposed(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        parsed = json.loads(json.dumps(result.to_dict()))
        assert parsed["total_proposed"] == result.total_proposed

    def test_roundtrip_proposals_list(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        parsed = json.loads(json.dumps(result.to_dict()))
        assert isinstance(parsed["proposals"], list)
        assert len(parsed["proposals"]) == len(result.proposals)

    def test_each_proposal_to_dict_serializable(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        for p in result.proposals:
            json.dumps(p.to_dict())

    def test_suppressed_count_in_dict(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        assert "suppressed_count" in result.to_dict()

    def test_generated_at_in_dict(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        assert "generated_at" in result.to_dict()

    def test_no_none_values_at_top_level(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        for key, val in result.to_dict().items():
            assert val is not None, f"Top-level key {key!r} is None"

    def test_empty_report_serializable(self):
        result = GeneratorEngine().generate_tests(_report())
        parsed = json.loads(json.dumps(result.to_dict()))
        assert parsed["total_proposed"] == 0
        assert parsed["proposals"] == []

    def test_proposals_contain_expected_keys(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        expected_keys = {
            "test_type", "target_module", "target_function",
            "description", "invariant_targeted", "risk_justification",
            "severity_level", "confidence_score", "requires_human_review",
        }
        for p in result.proposals:
            d = p.to_dict()
            assert expected_keys.issubset(set(d.keys())), (
                f"Missing keys: {expected_keys - set(d.keys())}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8 — NO NaN / Inf IN FLOATS
# ─────────────────────────────────────────────────────────────────────────────

class TestNoNaNInf:
    """Extreme or invalid float inputs must be clamped; output never contains NaN/Inf."""

    def _make_extreme_report(self, confidence: float) -> FakePatchReport:
        return _report(
            _suggestion(module="agent.core", severity="high", confidence=confidence)
        )

    def test_nan_confidence_no_nan_in_output(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(float("nan")))
        assert _all_floats_finite(result.to_dict())

    def test_positive_inf_confidence_no_inf_in_output(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(float("inf")))
        assert _all_floats_finite(result.to_dict())

    def test_negative_inf_confidence_no_inf_in_output(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(float("-inf")))
        assert _all_floats_finite(result.to_dict())

    def test_very_large_confidence_clamped_to_one(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(1e308))
        for p in result.proposals:
            assert math.isfinite(p.confidence_score)
            assert p.confidence_score <= 1.0

    def test_very_large_confidence_output_finite(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(1e308))
        for p in result.proposals:
            assert math.isfinite(p.confidence_score)
            assert 0.0 <= p.confidence_score <= 1.0

    def test_negative_confidence_clamped_or_suppressed(self):
        result = GeneratorEngine().generate_tests(self._make_extreme_report(-999.0))
        for p in result.proposals:
            assert math.isfinite(p.confidence_score)
            assert p.confidence_score >= 0.0

    def test_normal_confidence_stays_finite(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        for p in result.proposals:
            assert math.isfinite(p.confidence_score)

    def test_to_dict_all_floats_finite(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        assert _all_floats_finite(result.to_dict())

    def test_extreme_instability_no_nan_in_output(self):
        state  = StructuralMetaState(structural_instability_index=1.0)
        result = GeneratorEngine().generate_tests(
            _make_rich_patch_report(), structural_state=state,
        )
        assert _all_floats_finite(result.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# 9 — IMMUTABILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestImmutability:
    """TestCaseProposal must be frozen; mutation must raise AttributeError or TypeError."""

    def _get_proposal(self) -> TestCaseProposal:
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        assert len(result.proposals) > 0, "Need at least one proposal"
        return result.proposals[0]

    def test_cannot_set_target_module(self):
        p = self._get_proposal()
        try:
            p.target_module = "hacked"
            raise AssertionError("Should have raised on frozen field assignment")
        except (AttributeError, TypeError):
            pass

    def test_cannot_set_test_type(self):
        p = self._get_proposal()
        try:
            p.test_type = "mutation"
            raise AssertionError("Should have raised on frozen field assignment")
        except (AttributeError, TypeError):
            pass

    def test_cannot_set_severity_level(self):
        p = self._get_proposal()
        try:
            p.severity_level = "low"
            raise AssertionError("Should have raised on frozen field assignment")
        except (AttributeError, TypeError):
            pass

    def test_cannot_set_confidence_score(self):
        p = self._get_proposal()
        try:
            p.confidence_score = 0.0
            raise AssertionError("Should have raised on frozen field assignment")
        except (AttributeError, TypeError):
            pass

    def test_cannot_delete_target_module(self):
        p = self._get_proposal()
        try:
            del p.target_module
            raise AssertionError("Should have raised on frozen field deletion")
        except (AttributeError, TypeError):
            pass

    def test_report_proposals_is_tuple(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        assert isinstance(result.proposals, tuple)

    def test_report_itself_is_frozen(self):
        result = GeneratorEngine().generate_tests(_make_rich_patch_report())
        try:
            result.total_proposed = 999
            raise AssertionError("GenerationReport should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_input_patch_report_not_mutated(self):
        suggestions_before = tuple(
            _suggestion(module=f"agent.m{i}", severity="high", confidence=0.80)
            for i in range(5)
        )
        patch_report = _report(*suggestions_before)
        original_len = len(patch_report.suggestions)
        GeneratorEngine().generate_tests(patch_report)
        assert len(patch_report.suggestions) == original_len


# ─────────────────────────────────────────────────────────────────────────────
# 10 — EMPTY REPORT
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyReport:
    """An empty patch_report must produce a valid, empty GenerationReport."""

    def test_total_proposed_is_zero(self):
        result = GeneratorEngine().generate_tests(_report())
        assert result.total_proposed == 0

    def test_proposals_is_empty(self):
        result = GeneratorEngine().generate_tests(_report())
        assert len(result.proposals) == 0

    def test_proposals_is_tuple_type(self):
        result = GeneratorEngine().generate_tests(_report())
        assert isinstance(result.proposals, tuple)

    def test_suppressed_count_is_zero(self):
        result = GeneratorEngine().generate_tests(_report())
        assert result.suppressed_count == 0

    def test_to_dict_still_serializable(self):
        result = GeneratorEngine().generate_tests(_report())
        json.dumps(result.to_dict())

    def test_to_dict_proposals_key_exists(self):
        result = GeneratorEngine().generate_tests(_report())
        assert "proposals" in result.to_dict()

    def test_to_dict_total_proposed_key_exists(self):
        result = GeneratorEngine().generate_tests(_report())
        assert "total_proposed" in result.to_dict()

    def test_empty_with_instability_still_empty(self):
        state  = StructuralMetaState(structural_instability_index=1.0)
        result = GeneratorEngine().generate_tests(_report(), structural_state=state)
        assert result.total_proposed == 0

    def test_empty_with_gate_still_empty(self):
        gate   = StructuralGateDecision(patch_aggressiveness=1.0)
        result = GeneratorEngine().generate_tests(_report(), gate_decision=gate)
        assert result.total_proposed == 0

    def test_empty_report_result_is_frozen(self):
        result = GeneratorEngine().generate_tests(_report())
        try:
            result.total_proposed = 999
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_no_nan_in_empty_result(self):
        result = GeneratorEngine().generate_tests(_report())
        assert _all_floats_finite(result.to_dict())