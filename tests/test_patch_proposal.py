"""
A.C.C.E.S.S. — Patch Proposal Engine Test Suite (Phase 6.2)

Coverage:
    TestPatchProposalConfig       — config immutability, bounds
    TestPatchSuggestion           — frozen, serialization, repr
    TestProposalReport            — collection, serialization
    TestComplexityRules           — high CC → split suggestions
    TestNestingRules              — deep nesting → extract method
    TestGodClassRules             — large class → split suggestion
    TestCircularDepRules          — cycles → dependency inversion
    TestLayerViolationRules       — violations → adapter suggestion
    TestFileLevelRules            — IO/state density, oversized module
    TestAggressivenessModulation  — instability → filtering
    TestBoundedOutput             — hard cap, no infinite loops
    TestDeterminism               — same input → same output
    TestEdgeCases                 — empty report, no files, extreme values
    TestColdStart                 — no structural state → full output
    TestSeverityClassification    — correct tier mapping
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.patch_proposal import (
    PatchProposalConfig,
    PatchProposalEngine,
    PatchSuggestion,
    ProposalReport,
    Severity,
    IssueType,
)
from agent.structural_meta import (
    StructuralMetaState,
    StructuralGateDecision,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE DATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeFunc:
    name: str = "my_func"
    cyclomatic_complexity: int = 5
    max_nesting: int = 2
    parameter_count: int = 3
    lines: int = 30
    is_complex: bool = False
    is_deeply_nested: bool = False
    is_over_parameterized: bool = False
    is_long: bool = False


@dataclass
class FakeClass:
    name: str = "MyClass"
    method_count: int = 5
    attribute_count: int = 3
    is_god_class: bool = False
    lines: int = 50


@dataclass
class FakeFileMetrics:
    path: str = "agent/core.py"
    lines_total: int = 200
    lines_code: int = 150
    function_count: int = 5
    class_count: int = 1
    import_count: int = 5
    imports: tuple = ()
    functions: tuple = field(default_factory=tuple)
    classes: tuple = field(default_factory=tuple)
    smells: tuple = ()
    io_density: float = 0.1
    state_density: float = 0.1
    parse_error: bool = False
    is_oversized: bool = False


@dataclass
class FakeReport:
    files: tuple = field(default_factory=tuple)
    import_graph: tuple = ()
    cycles: tuple = ()
    layer_violations: tuple = ()
    fan_in: tuple = ()
    fan_out: tuple = ()
    smells: tuple = ()
    risk_scores: tuple = ()
    composite_risk: float = 0.3
    health_grade: str = "B"
    files_analyzed: int = 0
    lines_analyzed: int = 0


def _make_complex_report():
    """Create a report with various issues for testing."""
    complex_func = FakeFunc(name="process_all", cyclomatic_complexity=20, max_nesting=6, parameter_count=10, lines=100)
    simple_func = FakeFunc(name="helper", cyclomatic_complexity=3, max_nesting=1, parameter_count=2, lines=10)
    god_class = FakeClass(name="MegaClass", method_count=25, attribute_count=15, is_god_class=True)
    normal_class = FakeClass(name="SmallClass", method_count=3, attribute_count=2)
    heavy_io_file = FakeFileMetrics(
        path="agent/io_heavy.py", lines_code=600, io_density=0.55, state_density=0.2,
        functions=(complex_func, simple_func), classes=(god_class,), is_oversized=True,
    )
    clean_file = FakeFileMetrics(
        path="agent/clean.py", lines_code=100, io_density=0.05, state_density=0.1,
        functions=(simple_func,), classes=(normal_class,),
    )
    return FakeReport(
        files=(heavy_io_file, clean_file),
        cycles=(("a", "b", "c"),),
        layer_violations=(("persistence", "agent_core"),),
        smells=("smell1", "smell2", "smell3"),
        composite_risk=0.55,
        health_grade="C",
        files_analyzed=2,
        lines_analyzed=700,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchProposalConfig:
    def test_config_frozen(self):
        cfg = PatchProposalConfig()
        try:
            cfg.max_suggestions = 100
            assert False
        except (AttributeError, TypeError):
            pass

    def test_custom_config(self):
        cfg = PatchProposalConfig(max_suggestions=10, complexity_medium=15)
        assert cfg.max_suggestions == 10
        assert cfg.complexity_medium == 15

    def test_severity_thresholds_ordered(self):
        cfg = PatchProposalConfig()
        assert cfg.complexity_medium < cfg.complexity_high < cfg.complexity_critical
        assert cfg.nesting_medium < cfg.nesting_high < cfg.nesting_critical
        assert cfg.module_size_medium < cfg.module_size_high < cfg.module_size_critical

    def test_confidence_floor_reasonable(self):
        cfg = PatchProposalConfig()
        assert 0.0 < cfg.confidence_floor < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTION MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchSuggestion:
    def test_suggestion_frozen(self):
        s = PatchSuggestion(module="core", location="func", issue_type="test", severity="high", rationale="r", suggested_refactor_strategy="s")
        try:
            s.module = "other"
            assert False
        except (AttributeError, TypeError):
            pass

    def test_requires_human_review_always_true(self):
        s = PatchSuggestion(module="m", location="l", issue_type="t", severity="s", rationale="r", suggested_refactor_strategy="s")
        assert s.requires_human_review is True

    def test_to_dict_json_serializable(self):
        s = PatchSuggestion(
            module="core", location="func", issue_type="high_complexity",
            severity="high", rationale="test", suggested_refactor_strategy="split",
            confidence_score=0.8, metric_evidence=(("cc", 15),),
        )
        d = s.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["module"] == "core"
        assert parsed["confidence_score"] == 0.8
        assert len(parsed["metric_evidence"]) == 1

    def test_repr_informative(self):
        s = PatchSuggestion(module="x", location="y", issue_type="high_complexity", severity="high", rationale="r", suggested_refactor_strategy="s")
        r = repr(s)
        assert "PatchSuggestion" in r
        assert "high_complexity" in r


class TestProposalReport:
    def test_report_frozen(self):
        r = ProposalReport()
        try:
            r.total_suggestions_emitted = 999
            assert False
        except (AttributeError, TypeError):
            pass

    def test_report_to_dict(self):
        r = ProposalReport(total_issues_detected=5, total_suggestions_emitted=3)
        d = r.to_dict()
        assert d["total_issues_detected"] == 5
        assert isinstance(json.dumps(d), str)

    def test_report_repr(self):
        r = ProposalReport(total_suggestions_emitted=3, suggestions_suppressed=2)
        assert "emitted=3" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY RULE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexityRules:
    def test_high_cc_generates_suggestion(self):
        func = FakeFunc(name="complex_fn", cyclomatic_complexity=20)
        fm = FakeFileMetrics(path="agent/core.py", functions=(func,))
        report = FakeReport(files=(fm,))
        engine = PatchProposalEngine()
        result = engine.propose(report)
        cc_suggestions = [s for s in result.suggestions if s.issue_type == IssueType.HIGH_COMPLEXITY.value]
        assert len(cc_suggestions) >= 1
        assert cc_suggestions[0].module == "core"
        assert cc_suggestions[0].location == "complex_fn"

    def test_low_cc_no_suggestion(self):
        func = FakeFunc(name="simple_fn", cyclomatic_complexity=5)
        fm = FakeFileMetrics(path="agent/simple.py", functions=(func,))
        report = FakeReport(files=(fm,))
        engine = PatchProposalEngine()
        result = engine.propose(report)
        cc_suggestions = [s for s in result.suggestions if s.issue_type == IssueType.HIGH_COMPLEXITY.value]
        assert len(cc_suggestions) == 0

    def test_critical_cc_severity(self):
        func = FakeFunc(name="mega_fn", cyclomatic_complexity=30)
        fm = FakeFileMetrics(path="agent/mega.py", functions=(func,))
        report = FakeReport(files=(fm,))
        engine = PatchProposalEngine()
        result = engine.propose(report)
        cc_suggestions = [s for s in result.suggestions if s.issue_type == IssueType.HIGH_COMPLEXITY.value]
        assert len(cc_suggestions) >= 1
        assert cc_suggestions[0].severity == Severity.CRITICAL.value


class TestNestingRules:
    def test_deep_nesting_generates_suggestion(self):
        func = FakeFunc(name="nested_fn", max_nesting=7)
        fm = FakeFileMetrics(path="agent/deep.py", functions=(func,))
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        nesting = [s for s in result.suggestions if s.issue_type == IssueType.DEEP_NESTING.value]
        assert len(nesting) >= 1

    def test_shallow_nesting_no_suggestion(self):
        func = FakeFunc(name="flat_fn", max_nesting=2)
        fm = FakeFileMetrics(path="agent/flat.py", functions=(func,))
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        nesting = [s for s in result.suggestions if s.issue_type == IssueType.DEEP_NESTING.value]
        assert len(nesting) == 0


class TestGodClassRules:
    def test_god_class_generates_suggestion(self):
        cls = FakeClass(name="BigClass", method_count=30, attribute_count=20, is_god_class=True)
        fm = FakeFileMetrics(path="agent/god.py", classes=(cls,))
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        gods = [s for s in result.suggestions if s.issue_type == IssueType.GOD_CLASS.value]
        assert len(gods) >= 1
        assert "BigClass" in gods[0].location

    def test_small_class_no_suggestion(self):
        cls = FakeClass(name="SmallClass", method_count=3, is_god_class=False)
        fm = FakeFileMetrics(path="agent/small.py", classes=(cls,))
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        gods = [s for s in result.suggestions if s.issue_type == IssueType.GOD_CLASS.value]
        assert len(gods) == 0


class TestCircularDepRules:
    def test_cycle_generates_suggestion(self):
        report = FakeReport(cycles=(("a", "b", "c"),))
        result = PatchProposalEngine().propose(report)
        cycles = [s for s in result.suggestions if s.issue_type == IssueType.CIRCULAR_DEPENDENCY.value]
        assert len(cycles) >= 1
        assert "a" in cycles[0].rationale

    def test_no_cycles_no_suggestion(self):
        report = FakeReport(cycles=())
        result = PatchProposalEngine().propose(report)
        cycles = [s for s in result.suggestions if s.issue_type == IssueType.CIRCULAR_DEPENDENCY.value]
        assert len(cycles) == 0

    def test_long_cycle_critical_severity(self):
        report = FakeReport(cycles=(("a", "b", "c", "d"),))
        result = PatchProposalEngine().propose(report)
        cycles = [s for s in result.suggestions if s.issue_type == IssueType.CIRCULAR_DEPENDENCY.value]
        assert cycles[0].severity == Severity.CRITICAL.value


class TestLayerViolationRules:
    def test_violation_generates_suggestion(self):
        report = FakeReport(layer_violations=(("persistence", "agent_core"),))
        result = PatchProposalEngine().propose(report)
        violations = [s for s in result.suggestions if s.issue_type == IssueType.LAYER_VIOLATION.value]
        assert len(violations) >= 1
        assert violations[0].module == "persistence"

    def test_no_violations_no_suggestion(self):
        report = FakeReport(layer_violations=())
        result = PatchProposalEngine().propose(report)
        violations = [s for s in result.suggestions if s.issue_type == IssueType.LAYER_VIOLATION.value]
        assert len(violations) == 0


class TestFileLevelRules:
    def test_high_io_density_generates_suggestion(self):
        fm = FakeFileMetrics(path="agent/io.py", io_density=0.6)
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        io = [s for s in result.suggestions if s.issue_type == IssueType.HIGH_IO_DENSITY.value]
        assert len(io) >= 1

    def test_high_state_density_generates_suggestion(self):
        fm = FakeFileMetrics(path="agent/state.py", state_density=0.65)
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        state = [s for s in result.suggestions if s.issue_type == IssueType.HIGH_STATE_DENSITY.value]
        assert len(state) >= 1

    def test_oversized_module_generates_suggestion(self):
        fm = FakeFileMetrics(path="agent/big.py", lines_code=900, is_oversized=True)
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        oversized = [s for s in result.suggestions if s.issue_type == IssueType.OVERSIZED_MODULE.value]
        assert len(oversized) >= 1

    def test_clean_file_no_suggestion(self):
        fm = FakeFileMetrics(path="agent/clean.py", io_density=0.05, state_density=0.05, lines_code=50)
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        assert result.total_suggestions_emitted == 0


# ─────────────────────────────────────────────────────────────────────────────
# AGGRESSIVENESS MODULATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAggressivenessModulation:
    def test_no_state_full_aggressiveness(self):
        report = _make_complex_report()
        result = PatchProposalEngine().propose(report)
        assert result.aggressiveness_level == 1.0

    def test_low_instability_full_aggressiveness(self):
        report = _make_complex_report()
        state = StructuralMetaState(structural_instability_index=0.1)
        result = PatchProposalEngine().propose(report, structural_state=state)
        assert result.aggressiveness_level > 0.95

    def test_high_instability_reduced_suggestions(self):
        report = _make_complex_report()
        state = StructuralMetaState(structural_instability_index=0.8)
        result = PatchProposalEngine().propose(report, structural_state=state)
        assert result.aggressiveness_level < 0.5
        # Only high/critical should survive
        for s in result.suggestions:
            assert s.severity in (Severity.HIGH.value, Severity.CRITICAL.value)

    def test_gate_decision_overrides_state(self):
        report = _make_complex_report()
        gate = StructuralGateDecision(patch_aggressiveness=0.3, max_suggestions=5)
        result = PatchProposalEngine().propose(report, gate_decision=gate)
        assert result.max_suggestions_allowed == 5
        assert result.total_suggestions_emitted <= 5

    def test_aggressiveness_affects_count(self):
        report = _make_complex_report()
        full = PatchProposalEngine().propose(report)
        reduced = PatchProposalEngine().propose(
            report, structural_state=StructuralMetaState(structural_instability_index=0.9),
        )
        assert reduced.total_suggestions_emitted <= full.total_suggestions_emitted


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDED OUTPUT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedOutput:
    def test_hard_cap_respected(self):
        # Create many issues
        funcs = tuple(FakeFunc(name=f"fn_{i}", cyclomatic_complexity=25, max_nesting=8, parameter_count=12, lines=150) for i in range(30))
        fm = FakeFileMetrics(path="agent/huge.py", functions=funcs, lines_code=3000, is_oversized=True)
        report = FakeReport(files=(fm,))
        cfg = PatchProposalConfig(max_suggestions=10)
        result = PatchProposalEngine(config=cfg).propose(report)
        assert result.total_suggestions_emitted <= 10

    def test_suggestions_suppressed_counted(self):
        funcs = tuple(FakeFunc(name=f"fn_{i}", cyclomatic_complexity=25) for i in range(30))
        fm = FakeFileMetrics(path="agent/many.py", functions=funcs)
        report = FakeReport(files=(fm,))
        cfg = PatchProposalConfig(max_suggestions=5)
        result = PatchProposalEngine(config=cfg).propose(report)
        assert result.suggestions_suppressed > 0

    def test_confidence_floor_filters(self):
        cfg = PatchProposalConfig(confidence_floor=0.99)
        report = _make_complex_report()
        result = PatchProposalEngine(config=cfg).propose(report)
        # Very high floor should suppress most suggestions
        for s in result.suggestions:
            assert s.confidence_score >= 0.99


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISM TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_runs(self):
        report = _make_complex_report()
        engine = PatchProposalEngine()
        r1 = engine.propose(report)
        r2 = engine.propose(report)
        assert r1.total_suggestions_emitted == r2.total_suggestions_emitted
        for s1, s2 in zip(r1.suggestions, r2.suggestions):
            assert s1.module == s2.module
            assert s1.location == s2.location
            assert s1.issue_type == s2.issue_type
            assert s1.severity == s2.severity

    def test_ordering_by_severity_then_module(self):
        report = _make_complex_report()
        result = PatchProposalEngine().propose(report)
        severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        for i in range(1, len(result.suggestions)):
            prev = result.suggestions[i - 1]
            curr = result.suggestions[i]
            prev_rank = severity_rank.get(prev.severity, 0)
            curr_rank = severity_rank.get(curr.severity, 0)
            if prev_rank == curr_rank:
                assert prev.module <= curr.module or prev.location <= curr.location
            else:
                assert prev_rank >= curr_rank


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_report(self):
        report = FakeReport()
        result = PatchProposalEngine().propose(report)
        assert result.total_suggestions_emitted == 0
        assert result.total_issues_detected == 0

    def test_no_files(self):
        report = FakeReport(files=())
        result = PatchProposalEngine().propose(report)
        assert result.total_suggestions_emitted == 0

    def test_file_with_no_functions(self):
        fm = FakeFileMetrics(path="agent/empty.py", functions=(), classes=())
        report = FakeReport(files=(fm,))
        result = PatchProposalEngine().propose(report)
        # Only file-level rules could fire
        for s in result.suggestions:
            assert "(file-level)" in s.location or "cycle" in s.location or "import" in s.location

    def test_report_json_roundtrip(self):
        report = _make_complex_report()
        result = PatchProposalEngine().propose(report)
        d = result.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["total_suggestions_emitted"] == result.total_suggestions_emitted

    def test_suggestion_metric_evidence_serializable(self):
        report = _make_complex_report()
        result = PatchProposalEngine().propose(report)
        for s in result.suggestions:
            d = s.to_dict()
            json.dumps(d)  # Must not raise


# ─────────────────────────────────────────────────────────────────────────────
# COLD START TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestColdStart:
    def test_no_structural_state_works(self):
        report = _make_complex_report()
        result = PatchProposalEngine().propose(report, structural_state=None)
        assert result.aggressiveness_level == 1.0
        assert result.total_suggestions_emitted > 0

    def test_default_state_full_output(self):
        report = _make_complex_report()
        state = StructuralMetaState()
        result = PatchProposalEngine().propose(report, structural_state=state)
        assert result.aggressiveness_level > 0.95


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY CLASSIFICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityClassification:
    def test_severity_enum_ranking(self):
        assert Severity.LOW.rank < Severity.MEDIUM.rank
        assert Severity.MEDIUM.rank < Severity.HIGH.rank
        assert Severity.HIGH.rank < Severity.CRITICAL.rank

    def test_issue_type_values(self):
        assert IssueType.HIGH_COMPLEXITY.value == "high_complexity"
        assert IssueType.GOD_CLASS.value == "god_class"

    def test_all_issue_types_have_strategy(self):
        from agent.patch_proposal import _REFACTOR_STRATEGIES
        for issue in IssueType:
            assert issue in _REFACTOR_STRATEGIES, f"Missing strategy for {issue}"


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestPatchProposalConfig, TestPatchSuggestion, TestProposalReport,
        TestComplexityRules, TestNestingRules, TestGodClassRules,
        TestCircularDepRules, TestLayerViolationRules, TestFileLevelRules,
        TestAggressivenessModulation, TestBoundedOutput, TestDeterminism,
        TestEdgeCases, TestColdStart, TestSeverityClassification,
    ]
    passed, failed = [], []
    for cls in test_classes:
        instance = cls()
        for method_name in sorted(m for m in dir(cls) if m.startswith("test_")):
            label = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed.append(label)
                print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e))
                print(f"  ❌ {label}: {e}")
                traceback.print_exc()
    total = len(passed) + len(failed)
    print(f"\n{'='*60}\nResults: {len(passed)}/{total} passed")
    if failed:
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")