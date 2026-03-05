"""
A.C.C.E.S.S. — System-Level Integration Tests (Phase 5.2)

Full pipeline coherence across:
    StructuralMetaState → StructuralGateDecision
    → PatchReport (fake, deterministic)
    → GeneratorEngine
    → HumanApprovalGateEngine

All tests use real engines and deterministic fake inputs.
No randomness. No logging. No print. No external deps beyond stdlib + pytest.
"""

import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.structural_meta import (
    StructuralMetaState,
    StructuralGateDecision,
)
from agent.test_generator import (
    GeneratorEngine,
    GenerationConfig,
    GenerationReport,
)
from agent.human_gate import (
    HumanApprovalGateEngine,
    HumanApprovalDecision,
    ApprovalAction,
    ApprovalValidationError,
    PatchBundle,
    DiffView,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE PATCH LAYER — deterministic stubs, no production PatchProposalEngine
# ─────────────────────────────────────────────────────────────────────────────

_VALID_ISSUE_TYPES = (
    "high_complexity",
    "deep_nesting",
    "god_class",
    "circular_dependency",
    "layer_violation",
    "high_io_density",
    "high_state_density",
    "oversized_module",
    "over_parameterized",
    "long_function",
)


@dataclass(frozen=True)
class FakeSuggestion:
    """Deterministic duck-typed stub for PatchSuggestion."""
    module: str = "agent.core"
    location: str = "fn"
    issue_type: str = "high_complexity"
    severity: str = "high"
    rationale: str = "Structural issue detected."
    suggested_refactor_strategy: str = "Decompose into smaller units."
    confidence_score: float = 0.80
    risk_reduction_estimate: float = 0.10
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
    """Deterministic duck-typed stub for ProposalReport."""
    suggestions: tuple = field(default_factory=tuple)
    total_suggestions_emitted: int = 0
    suggestions_suppressed: int = 0
    aggressiveness_level: float = 1.0
    structural_instability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "total_suggestions_emitted": self.total_suggestions_emitted,
            "suggestions_suppressed": self.suggestions_suppressed,
            "aggressiveness_level": self.aggressiveness_level,
            "structural_instability": self.structural_instability,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _s(
    module: str = "agent.core",
    severity: str = "high",
    confidence: float = 0.80,
    location: str = "fn",
    issue_type: str = "high_complexity",
) -> FakeSuggestion:
    return FakeSuggestion(
        module=module,
        location=location,
        issue_type=issue_type,
        severity=severity,
        confidence_score=confidence,
    )


def _patch_report(*suggestions: FakeSuggestion) -> FakePatchReport:
    return FakePatchReport(
        suggestions=tuple(suggestions),
        total_suggestions_emitted=len(suggestions),
    )


def _rich_patch_report() -> FakePatchReport:
    return _patch_report(
        _s(module="agent.alpha", severity="critical",  confidence=0.92,
           issue_type="high_complexity"),
        _s(module="agent.beta",  severity="high",      confidence=0.85,
           issue_type="deep_nesting"),
        _s(module="agent.gamma", severity="medium",    confidence=0.70,
           issue_type="god_class"),
        _s(module="agent.delta", severity="low",       confidence=0.55,
           issue_type="long_function"),
        _s(module="agent.alpha", severity="high",      confidence=0.80,
           issue_type="layer_violation"),
        _s(module="agent.beta",  severity="medium",    confidence=0.65,
           issue_type="high_io_density"),
    )


def _large_patch_report(n: int = 50, severity: str = "high") -> FakePatchReport:
    issue_types = _VALID_ISSUE_TYPES
    suggestions = tuple(
        _s(
            module=f"agent.mod_{i:03d}",
            severity=severity,
            confidence=0.90,
            location=f"fn_{i}",
            issue_type=issue_types[i % len(issue_types)],
        )
        for i in range(n)
    )
    return _patch_report(*suggestions)


def _all_floats_finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_floats_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_floats_finite(v) for v in obj)
    return True


def _run_full_pipeline(patch_report, structural_state=None, gate_decision=None):
    test_engine = GeneratorEngine()
    gate_engine = HumanApprovalGateEngine()

    test_report = test_engine.generate_tests(
        patch_report,
        structural_state=structural_state,
        gate_decision=gate_decision,
    )
    bundle = gate_engine.create_patch_bundle(patch_report, test_report)

    return test_report, bundle


# ─────────────────────────────────────────────────────────────────────────────
# TEST SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineDeterministic:

    def test_full_pipeline_deterministic(self):
        patch_report     = _rich_patch_report()
        structural_state = StructuralMetaState(structural_instability_index=0.3)
        gate_decision    = StructuralGateDecision(patch_aggressiveness=0.8)

        r1, b1 = _run_full_pipeline(patch_report, structural_state, gate_decision)
        r2, b2 = _run_full_pipeline(patch_report, structural_state, gate_decision)

        assert r1.total_proposed    == r2.total_proposed
        assert r1.suppressed_count  == r2.suppressed_count
        assert r1.highest_severity  == r2.highest_severity
        assert len(r1.proposals)    == len(r2.proposals)

        for p1, p2 in zip(r1.proposals, r2.proposals):
            assert p1.test_type        == p2.test_type
            assert p1.target_module    == p2.target_module
            assert p1.severity_level   == p2.severity_level
            assert p1.confidence_score == p2.confidence_score
            assert p1.invariant_targeted == p2.invariant_targeted

        assert len(b1.diff_views) == len(b2.diff_views)
        for dv1, dv2 in zip(b1.diff_views, b2.diff_views):
            assert dv1.module_name            == dv2.module_name
            assert dv1.location               == dv2.location
            assert dv1.severity_level         == dv2.severity_level
            assert dv1.confidence_score       == dv2.confidence_score
            assert dv1.structural_risk_level  == dv2.structural_risk_level
            assert dv1.original_summary       == dv2.original_summary
            assert dv1.proposed_change_summary == dv2.proposed_change_summary

        gate_engine = HumanApprovalGateEngine()
        module_names = tuple(dv.module_name for dv in b1.diff_views)[:2]
        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=module_names,
        )
        s1 = gate_engine.evaluate_decision(b1, decision)
        s2 = gate_engine.evaluate_decision(b2, decision)

        for key in ("action", "approved_count", "rejected_count",
                    "requires_followup", "safe_to_execute"):
            assert s1[key] == s2[key]

    def test_input_objects_not_mutated(self):
        patch_report     = _rich_patch_report()
        structural_state = StructuralMetaState(structural_instability_index=0.2)
        gate_decision    = StructuralGateDecision(patch_aggressiveness=1.0)

        original_suggestions_len = len(patch_report.suggestions)
        original_instability     = structural_state.structural_instability_index
        original_aggressiveness  = gate_decision.patch_aggressiveness

        _run_full_pipeline(patch_report, structural_state, gate_decision)

        assert len(patch_report.suggestions)                  == original_suggestions_len
        assert structural_state.structural_instability_index  == original_instability
        assert gate_decision.patch_aggressiveness             == original_aggressiveness


class TestStructuralGateReducesPipelineOutput:

    def test_gate_reduces_test_proposals(self):
        patch_report = _large_patch_report(30, severity="critical")
        gate_zero    = StructuralGateDecision(patch_aggressiveness=0.0)

        r_normal, _ = _run_full_pipeline(patch_report)
        r_gated,  _ = _run_full_pipeline(patch_report, gate_decision=gate_zero)

        assert r_gated.total_proposed <= r_normal.total_proposed

    def test_gate_does_not_change_diff_views_count(self):
        patch_report = _large_patch_report(20, severity="high")
        gate_zero    = StructuralGateDecision(patch_aggressiveness=0.0)

        _, b_normal = _run_full_pipeline(patch_report)
        _, b_gated  = _run_full_pipeline(patch_report, gate_decision=gate_zero)

        assert len(b_normal.diff_views) == len(patch_report.suggestions)
        assert len(b_gated.diff_views)  == len(patch_report.suggestions)

    def test_safe_to_execute_false_with_gate(self):
        patch_report = _rich_patch_report()
        gate_zero    = StructuralGateDecision(patch_aggressiveness=0.0)

        _, bundle = _run_full_pipeline(patch_report, gate_decision=gate_zero)
        decision = HumanApprovalDecision(action=ApprovalAction.DEFER)
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)

        assert summary["safe_to_execute"] is False


class TestHighInstabilityReducesTestGeneration:

    def test_high_instability_reduces_proposals(self):
        patch_report = _large_patch_report(25, severity="high")
        high_state   = StructuralMetaState(structural_instability_index=1.0)

        r_baseline, _ = _run_full_pipeline(patch_report)
        r_unstable, _ = _run_full_pipeline(patch_report, structural_state=high_state)

        assert r_unstable.total_proposed <= r_baseline.total_proposed

    def test_high_instability_leaves_diff_views_unchanged(self):
        patch_report = _large_patch_report(15)
        high_state   = StructuralMetaState(structural_instability_index=1.0)

        _, b_baseline = _run_full_pipeline(patch_report)
        _, b_unstable = _run_full_pipeline(patch_report, structural_state=high_state)

        assert len(b_unstable.diff_views) == len(b_baseline.diff_views)
        assert len(b_unstable.diff_views) == len(patch_report.suggestions)

    def test_high_instability_no_crash(self):
        patch_report = _rich_patch_report()
        high_state   = StructuralMetaState(structural_instability_index=1.0)

        r, b = _run_full_pipeline(patch_report, structural_state=high_state)
        assert isinstance(r, GenerationReport)
        assert isinstance(b, PatchBundle)

    def test_high_instability_no_nan_in_output(self):
        patch_report = _rich_patch_report()
        high_state   = StructuralMetaState(structural_instability_index=1.0)

        r, b = _run_full_pipeline(patch_report, structural_state=high_state)
        assert _all_floats_finite(r.to_dict())
        assert _all_floats_finite(b.to_dict())


class TestHumanRejectBlocksPipeline:

    def test_reject_approved_count_zero(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        first_module = bundle.diff_views[0].module_name
        decision = HumanApprovalDecision(
            action=ApprovalAction.REJECT,
            rejected_modules=(first_module,),
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["approved_count"] == 0

    def test_reject_rejected_count_positive(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        modules = tuple(dv.module_name for dv in bundle.diff_views[:2])
        decision = HumanApprovalDecision(
            action=ApprovalAction.REJECT,
            rejected_modules=modules,
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["rejected_count"] > 0

    def test_reject_safe_to_execute_false(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        first_module = bundle.diff_views[0].module_name
        decision = HumanApprovalDecision(
            action=ApprovalAction.REJECT,
            rejected_modules=(first_module,),
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["safe_to_execute"] is False

    def test_reject_does_not_mutate_bundle(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        original_diff_count = len(bundle.diff_views)
        first_module = bundle.diff_views[0].module_name
        decision = HumanApprovalDecision(
            action=ApprovalAction.REJECT,
            rejected_modules=(first_module,),
        )
        HumanApprovalGateEngine().evaluate_decision(bundle, decision)

        assert len(bundle.diff_views) == original_diff_count

    def test_reject_bundle_immutable_after_decision(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        try:
            bundle.diff_views = ()
            raise AssertionError("PatchBundle should be frozen")
        except (AttributeError, TypeError):
            pass


class TestRequestChangesRequiresNotes:

    def test_request_changes_without_notes_raises(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.REQUEST_CHANGES,
            reviewer_notes="",
        )
        with pytest.raises(ApprovalValidationError):
            HumanApprovalGateEngine().evaluate_decision(bundle, decision)

    def test_request_changes_whitespace_only_raises(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.REQUEST_CHANGES,
            reviewer_notes="   ",
        )
        with pytest.raises(ApprovalValidationError):
            HumanApprovalGateEngine().evaluate_decision(bundle, decision)

    def test_request_changes_with_notes_succeeds(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.REQUEST_CHANGES,
            reviewer_notes="Please add unit tests before merging.",
            requires_followup=True,
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["action"] == "request_changes"
        assert summary["safe_to_execute"] is False
        assert summary["requires_followup"] is True

    def test_request_changes_approved_count_zero(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.REQUEST_CHANGES,
            reviewer_notes="Needs more tests.",
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["approved_count"] == 0
        assert summary["rejected_count"] == 0


class TestPartialAcceptRequiresExplicitModules:

    def test_accept_without_modules_raises(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(action=ApprovalAction.ACCEPT)
        with pytest.raises(ApprovalValidationError):
            HumanApprovalGateEngine().evaluate_decision(bundle, decision)

    def test_accept_with_approved_modules_succeeds(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        first_module = bundle.diff_views[0].module_name
        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=(first_module,),
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["action"] == "accept"
        assert summary["approved_count"] == 1
        assert summary["safe_to_execute"] is False

    def test_accept_with_rejected_modules_also_raises(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        first_module  = bundle.diff_views[0].module_name
        second_module = bundle.diff_views[1].module_name
        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=(first_module,),
            rejected_modules=(second_module,),
        )
        with pytest.raises(ApprovalValidationError):
            HumanApprovalGateEngine().evaluate_decision(bundle, decision)

    def test_accept_unknown_module_raises(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=("agent.does_not_exist_anywhere",),
        )
        with pytest.raises(ApprovalValidationError):
            HumanApprovalGateEngine().evaluate_decision(bundle, decision)

    def test_accept_all_modules_count_matches(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        all_modules = tuple({dv.module_name for dv in bundle.diff_views})
        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=all_modules,
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["approved_count"] == len(all_modules)


class TestNoAutoExecutionPathExists:

    def test_safe_to_execute_false_on_accept(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        first_module = bundle.diff_views[0].module_name

        decision = HumanApprovalDecision(
            action=ApprovalAction.ACCEPT,
            approved_modules=(first_module,),
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["safe_to_execute"] is False

    def test_safe_to_execute_false_on_reject(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        first_module = bundle.diff_views[0].module_name

        decision = HumanApprovalDecision(
            action=ApprovalAction.REJECT,
            rejected_modules=(first_module,),
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["safe_to_execute"] is False

    def test_safe_to_execute_false_on_request_changes(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(
            action=ApprovalAction.REQUEST_CHANGES,
            reviewer_notes="Fix and resubmit.",
        )
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["safe_to_execute"] is False

    def test_safe_to_execute_false_on_defer(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)

        decision = HumanApprovalDecision(action=ApprovalAction.DEFER)
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["safe_to_execute"] is False

    def test_no_subprocess_in_human_gate_module(self):
        import agent.human_gate as hg_module
        source = inspect.getsource(hg_module)
        forbidden = ("subprocess", "os.system", "exec(", "eval(", "compile(")
        for token in forbidden:
            assert token not in source, (
                f"Forbidden token {token!r} found in agent/human_gate.py"
            )

    def test_no_file_write_in_human_gate_module(self):
        import agent.human_gate as hg_module
        source = inspect.getsource(hg_module)
        for bad in ('open(', 'write(', 'writelines('):
            assert bad not in source, (
                f"File-write primitive {bad!r} found in agent/human_gate.py"
            )

    def test_evaluate_decision_returns_dict(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        decision = HumanApprovalDecision(action=ApprovalAction.DEFER)
        result = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert isinstance(result, dict)
        assert "safe_to_execute" in result
        assert result["safe_to_execute"] is False


class TestPipelineHandlesEmptyPatchReport:

    def test_empty_patch_report_test_generation_empty(self):
        empty = _patch_report()
        r, _ = _run_full_pipeline(empty)
        assert r.total_proposed == 0
        assert len(r.proposals) == 0

    def test_empty_patch_report_diff_views_empty(self):
        empty = _patch_report()
        _, bundle = _run_full_pipeline(empty)
        assert len(bundle.diff_views) == 0
        assert isinstance(bundle.diff_views, tuple)

    def test_empty_patch_report_defer_valid(self):
        empty = _patch_report()
        _, bundle = _run_full_pipeline(empty)
        decision = HumanApprovalDecision(action=ApprovalAction.DEFER)
        summary = HumanApprovalGateEngine().evaluate_decision(bundle, decision)
        assert summary["action"] == "defer"
        assert summary["safe_to_execute"] is False

    def test_empty_patch_report_no_crash(self):
        empty = _patch_report()
        r, bundle = _run_full_pipeline(
            empty,
            structural_state=StructuralMetaState(structural_instability_index=1.0),
            gate_decision=StructuralGateDecision(patch_aggressiveness=0.0),
        )
        assert isinstance(r, GenerationReport)
        assert isinstance(bundle, PatchBundle)

    def test_empty_patch_report_json_serializable(self):
        empty = _patch_report()
        r, bundle = _run_full_pipeline(empty)
        json.dumps(r.to_dict())
        json.dumps(bundle.to_dict())

    def test_empty_patch_report_suppressed_count_zero(self):
        empty = _patch_report()
        r, _ = _run_full_pipeline(empty)
        assert r.suppressed_count == 0


class TestSeverityPropagationConsistency:

    def test_highest_severity_is_critical_when_present(self):
        patch_report = _patch_report(
            _s(module="agent.a", severity="critical", confidence=0.95,
               issue_type="high_complexity"),
            _s(module="agent.b", severity="high",     confidence=0.85,
               issue_type="deep_nesting"),
            _s(module="agent.c", severity="medium",   confidence=0.70,
               issue_type="long_function"),
        )
        r, _ = _run_full_pipeline(patch_report)
        if r.total_proposed > 0:
            assert r.highest_severity == "critical"

    def test_highest_severity_is_high_when_no_critical(self):
        patch_report = _patch_report(
            _s(module="agent.a", severity="high",   confidence=0.88,
               issue_type="deep_nesting"),
            _s(module="agent.b", severity="medium", confidence=0.75,
               issue_type="god_class"),
        )
        r, _ = _run_full_pipeline(patch_report)
        if r.total_proposed > 0:
            assert r.highest_severity in ("high", "critical")

    def test_diff_view_severity_matches_suggestion_severity(self):
        severities  = ["critical", "high", "medium", "low"]
        issue_types = ["high_complexity", "deep_nesting", "god_class", "long_function"]
        suggestions = tuple(
            _s(
                module=f"agent.m{i}",
                severity=severities[i],
                confidence=0.80,
                issue_type=issue_types[i],
            )
            for i in range(4)
        )
        patch_report = _patch_report(*suggestions)
        _, bundle = _run_full_pipeline(patch_report)

        severity_map = {s.module: s.severity for s in patch_report.suggestions}
        for dv in bundle.diff_views:
            expected = severity_map.get(dv.module_name)
            if expected is not None:
                assert dv.severity_level == expected

    def test_diff_view_count_matches_suggestion_count(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        assert len(bundle.diff_views) == len(patch_report.suggestions)

    def test_all_proposals_severity_levels_are_known(self):
        patch_report = _rich_patch_report()
        r, _ = _run_full_pipeline(patch_report)
        known = {"critical", "high", "medium", "low"}
        for p in r.proposals:
            assert p.severity_level in known

    def test_no_severity_mismatch_in_diff_views(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        known = {"critical", "high", "medium", "low"}
        for dv in bundle.diff_views:
            assert dv.severity_level in known


class TestGlobalBoundednessUnderLargeInput:

    def test_proposal_caps_respected_at_200_suggestions(self):
        patch_report = _large_patch_report(200, severity="critical")
        cfg = GenerationConfig(
            max_unit_tests=15,
            max_regression_tests=10,
            max_mutation_tests=8,
        )
        r = GeneratorEngine(config=cfg).generate_tests(patch_report)

        unit_count = sum(1 for p in r.proposals if p.test_type == "unit")
        reg_count  = sum(1 for p in r.proposals if p.test_type == "regression")
        mut_count  = sum(1 for p in r.proposals if p.test_type == "mutation")

        assert unit_count <= 15
        assert reg_count  <= 10
        assert mut_count  <= 8

    def test_no_nan_inf_in_test_report_at_200_suggestions(self):
        patch_report = _large_patch_report(200)
        r, _ = _run_full_pipeline(patch_report)
        assert _all_floats_finite(r.to_dict())

    def test_no_nan_inf_in_bundle_at_200_suggestions(self):
        patch_report = _large_patch_report(200)
        _, bundle = _run_full_pipeline(patch_report)
        assert _all_floats_finite(bundle.to_dict())

    def test_diff_views_count_equals_suggestion_count_at_200(self):
        patch_report = _large_patch_report(200)
        _, bundle = _run_full_pipeline(patch_report)
        assert len(bundle.diff_views) == 200

    def test_total_proposed_bounded_at_200_suggestions(self):
        patch_report = _large_patch_report(200)
        r, _ = _run_full_pipeline(patch_report)
        assert r.total_proposed <= 33

    def test_pipeline_returns_in_bounded_time_at_200_suggestions(self):
        patch_report = _large_patch_report(200)
        t0 = time.monotonic()
        _run_full_pipeline(patch_report)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0

    def test_json_serializable_at_200_suggestions(self):
        patch_report = _large_patch_report(200)
        r, bundle = _run_full_pipeline(patch_report)
        json.dumps(r.to_dict())
        json.dumps(bundle.to_dict())

    def test_instability_gate_combined_at_200_suggestions(self):
        patch_report  = _large_patch_report(200)
        high_state    = StructuralMetaState(structural_instability_index=1.0)
        gate_zero     = StructuralGateDecision(patch_aggressiveness=0.0)

        r, bundle = _run_full_pipeline(
            patch_report,
            structural_state=high_state,
            gate_decision=gate_zero,
        )
        assert _all_floats_finite(r.to_dict())
        assert _all_floats_finite(bundle.to_dict())
        assert len(bundle.diff_views) == 200


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CUTTING INVARIANT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalInvariants:

    def test_p1_deterministic_proposals_tuple(self):
        patch_report = _rich_patch_report()
        r1 = GeneratorEngine().generate_tests(patch_report)
        r2 = GeneratorEngine().generate_tests(patch_report)
        assert r1.proposals == r2.proposals

    def test_p2_patch_report_not_mutated(self):
        patch_report = _rich_patch_report()
        snap = len(patch_report.suggestions)
        _run_full_pipeline(patch_report)
        assert len(patch_report.suggestions) == snap

    def test_p3_structural_state_not_mutated(self):
        state = StructuralMetaState(structural_instability_index=0.6)
        snap  = state.structural_instability_index
        _run_full_pipeline(_rich_patch_report(), structural_state=state)
        assert state.structural_instability_index == snap

    def test_p4_no_auto_commit_in_any_module(self):
        import agent.human_gate  as hg
        import agent.test_generator as tg
        forbidden = ("git commit", "subprocess.run", "os.system", ".commit(")
        for mod in (hg, tg):
            src = inspect.getsource(mod)
            for token in forbidden:
                assert token not in src

    def test_p5_all_floats_finite_normal_run(self):
        patch_report = _rich_patch_report()
        r, bundle = _run_full_pipeline(patch_report)
        assert _all_floats_finite(r.to_dict())
        assert _all_floats_finite(bundle.to_dict())

    def test_p5_all_floats_finite_high_instability(self):
        state = StructuralMetaState(structural_instability_index=1.0)
        r, bundle = _run_full_pipeline(_rich_patch_report(), structural_state=state)
        assert _all_floats_finite(r.to_dict())
        assert _all_floats_finite(bundle.to_dict())

    def test_p6_safe_to_execute_false_every_action(self):
        patch_report = _rich_patch_report()
        _, bundle = _run_full_pipeline(patch_report)
        engine = HumanApprovalGateEngine()

        first_module = bundle.diff_views[0].module_name

        decisions = [
            HumanApprovalDecision(
                action=ApprovalAction.ACCEPT,
                approved_modules=(first_module,),
            ),
            HumanApprovalDecision(
                action=ApprovalAction.REJECT,
                rejected_modules=(first_module,),
            ),
            HumanApprovalDecision(
                action=ApprovalAction.REQUEST_CHANGES,
                reviewer_notes="Please revise.",
            ),
            HumanApprovalDecision(action=ApprovalAction.DEFER),
        ]

        for d in decisions:
            summary = engine.evaluate_decision(bundle, d)
            assert summary["safe_to_execute"] is False