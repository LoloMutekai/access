"""
A.C.C.E.S.S. — Generator Engine (Phase 6.3)

Deterministic, bounded, risk-aware test proposal generator.
Consumes PatchSuggestion inputs and produces structured TestCaseProposal
objects suitable for human review and implementation.

This module PROPOSES tests. It DESCRIBES invariants.
It does NOT run tests. It does NOT execute code. It does NOT import pytest.

Design:
    - TestCaseProposal is FROZEN — immutable, JSON-serializable
    - GeneratorEngine is pure: (patch_report, state, gate) → report
    - All outputs deterministic and bounded
    - No randomness, no side effects, no dynamic imports
    - No external dependencies beyond stdlib
    - requires_human_review is ALWAYS True — no auto-execution
    - Integrates with StructuralGate for instability-aware throttling

Rule mapping:
    Each issue_type maps to a set of invariant risks, which drive the
    generation of unit, regression, and mutation test proposals.

    issue_type              invariant risk               test focus
    ───────────────────────────────────────────────────────────────────
    high_complexity       → branch explosion risk       → path coverage
    deep_nesting          → path coverage risk          → depth traversal
    circular_dependency   → state entanglement risk     → isolation
    god_class             → cohesion violation risk      → responsibility
    layer_violation       → boundary breach risk        → import constraints
    high_io_density       → side-effect propagation     → purity verification
    high_state_density    → mutation coupling risk      → immutability
    oversized_module      → scope explosion risk        → modularity
    over_parameterized    → interface bloat risk        → contract narrowing
    long_function         → step entanglement risk      → decomposition

Mathematical properties:
    P1: Deterministic output — same inputs → same GenerationReport
    P2: No mutation of input objects
    P3: No infinite proposal growth — hard-capped
    P4: All floats bounded to [0.0, 1.0]
    P5: No NaN / Inf in any output field
    P6: Proposals ordered by (severity desc, module asc, test_type rank)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverity(str, Enum):
    """Severity ranking for test proposals."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class TestType(str, Enum):
    """Category of proposed test."""
    UNIT = "unit"
    REGRESSION = "regression"
    MUTATION = "mutation"

    @property
    def rank(self) -> int:
        return {"unit": 0, "regression": 1, "mutation": 2}[self.value]


class IssueType(str, Enum):
    """Mirror of patch_proposal.IssueType for invariant mapping."""
    HIGH_COMPLEXITY = "high_complexity"
    DEEP_NESTING = "deep_nesting"
    GOD_CLASS = "god_class"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    LAYER_VIOLATION = "layer_violation"
    HIGH_IO_DENSITY = "high_io_density"
    HIGH_STATE_DENSITY = "high_state_density"
    OVERSIZED_MODULE = "oversized_module"
    OVER_PARAMETERIZED = "over_parameterized"
    LONG_FUNCTION = "long_function"


# ─────────────────────────────────────────────────────────────────────────────
# INVARIANT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InvariantTemplate:
    """Describes the testing invariant associated with a structural issue."""
    risk_name: str
    unit_description: str
    regression_description: str
    mutation_description: str


_INVARIANT_MAP: dict[str, InvariantTemplate] = {
    IssueType.HIGH_COMPLEXITY.value: InvariantTemplate(
        risk_name="branch explosion risk",
        unit_description=(
            "Verify each extracted sub-function produces correct output "
            "for representative inputs covering all major branch paths."
        ),
        regression_description=(
            "Confirm that the refactored function yields identical output "
            "to the original for a fixed set of canonical inputs."
        ),
        mutation_description=(
            "Perturb branch conditions at decision boundaries and verify "
            "that output changes detectably — no silent path collapse."
        ),
    ),
    IssueType.DEEP_NESTING.value: InvariantTemplate(
        risk_name="path coverage risk",
        unit_description=(
            "Test each extracted helper at every nesting level "
            "with inputs that exercise the deepest code path."
        ),
        regression_description=(
            "Ensure flattened control flow preserves behavior for inputs "
            "that previously required traversal to maximum depth."
        ),
        mutation_description=(
            "Inject off-by-one guard clause conditions and verify "
            "that edge cases at nesting boundaries are caught."
        ),
    ),
    IssueType.CIRCULAR_DEPENDENCY.value: InvariantTemplate(
        risk_name="state entanglement risk",
        unit_description=(
            "Test each module in the former cycle in isolation using "
            "mock/stub interfaces at the decoupled boundary."
        ),
        regression_description=(
            "Verify that cross-module call sequences produce identical "
            "results after dependency inversion is applied."
        ),
        mutation_description=(
            "Swap mock implementations at the interface boundary and "
            "confirm that module behavior remains independent."
        ),
    ),
    IssueType.GOD_CLASS.value: InvariantTemplate(
        risk_name="cohesion violation risk",
        unit_description=(
            "Test each extracted cohesive class independently, verifying "
            "that its public interface satisfies its single responsibility."
        ),
        regression_description=(
            "Confirm that the original god class's full API contract is "
            "preserved across the set of extracted classes."
        ),
        mutation_description=(
            "Remove one extracted class from the composition and verify "
            "that dependent tests fail — no hidden redundancy."
        ),
    ),
    IssueType.LAYER_VIOLATION.value: InvariantTemplate(
        risk_name="boundary breach risk",
        unit_description=(
            "Verify the adapter or bridge module correctly translates "
            "between architectural layers without leaking abstractions."
        ),
        regression_description=(
            "Confirm that moving the import path does not alter behavior "
            "for all callers that previously used the violating import."
        ),
        mutation_description=(
            "Re-introduce the direct cross-layer import and verify "
            "that a static analysis check detects the violation."
        ),
    ),
    IssueType.HIGH_IO_DENSITY.value: InvariantTemplate(
        risk_name="side-effect propagation risk",
        unit_description=(
            "Test the pure logic core extracted from the IO-heavy module "
            "with synthetic inputs — no filesystem or network access."
        ),
        regression_description=(
            "Verify that the ports-and-adapters boundary produces "
            "identical observable side effects as the original."
        ),
        mutation_description=(
            "Inject a failing IO adapter and confirm the pure logic "
            "core remains unaffected — fault isolation holds."
        ),
    ),
    IssueType.HIGH_STATE_DENSITY.value: InvariantTemplate(
        risk_name="mutation coupling risk",
        unit_description=(
            "Test each method with frozen/immutable input objects and "
            "verify that return values carry all state changes."
        ),
        regression_description=(
            "Confirm that replacing mutable attributes with immutable "
            "return values preserves observable behavior."
        ),
        mutation_description=(
            "Inject unexpected attribute mutations between method calls "
            "and verify the refactored code is resilient."
        ),
    ),
    IssueType.OVERSIZED_MODULE.value: InvariantTemplate(
        risk_name="scope explosion risk",
        unit_description=(
            "Test each extracted sub-module's public API in isolation, "
            "verifying that internal concerns do not leak."
        ),
        regression_description=(
            "Verify that the __init__.py re-export surface matches "
            "the original module's public API exactly."
        ),
        mutation_description=(
            "Remove one sub-module from the package and verify that "
            "only its specific dependents fail — blast radius bounded."
        ),
    ),
    IssueType.OVER_PARAMETERIZED.value: InvariantTemplate(
        risk_name="interface bloat risk",
        unit_description=(
            "Test the parameter object's construction with all valid "
            "combinations and verify field-level constraints hold."
        ),
        regression_description=(
            "Confirm that replacing positional parameters with the "
            "parameter object does not change any call site behavior."
        ),
        mutation_description=(
            "Omit one field from the parameter object and verify "
            "that the function raises or uses the documented default."
        ),
    ),
    IssueType.LONG_FUNCTION.value: InvariantTemplate(
        risk_name="step entanglement risk",
        unit_description=(
            "Test each decomposed helper step independently with "
            "its own input/output assertions."
        ),
        regression_description=(
            "Verify that composing the helper steps in sequence "
            "produces identical output to the original monolith."
        ),
        mutation_description=(
            "Reorder two adjacent decomposed steps and verify "
            "that the test suite detects the semantic change."
        ),
    ),
}

# Severity weights for confidence scaling
_SEVERITY_WEIGHTS: dict[str, float] = {
    TestSeverity.LOW.value: 0.25,
    TestSeverity.MEDIUM.value: 0.50,
    TestSeverity.HIGH.value: 0.75,
    TestSeverity.CRITICAL.value: 1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationConfig:
    """
    Frozen configuration for test proposal generation.
    All limits are hard caps — never exceeded regardless of input size.
    """

    # Per-type caps
    max_unit_tests: int = 15
    max_regression_tests: int = 10
    max_mutation_tests: int = 8

    # Severity weighting: multiplier applied to confidence when
    # computing the final confidence_score of a proposal.
    # Higher severity → higher confidence boost.
    severity_weighting: float = 0.20

    # Proposals below this confidence are suppressed entirely.
    confidence_floor: float = 0.30

    # Instability reduction factor: at instability=1.0,
    # max proposals reduced by this fraction.
    instability_reduction: float = 0.60



# ─────────────────────────────────────────────────────────────────────────────
# TEST CASE PROPOSAL (immutable output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TestCaseProposal:
    """
    A single test proposal. Immutable, JSON-serializable.
    Always requires human review — no auto-execution.
    """
    test_type: str
    target_module: str
    target_function: str
    description: str
    invariant_targeted: str
    risk_justification: str
    severity_level: str
    confidence_score: float
    requires_human_review: bool = True

    def __post_init__(self):
        """Enforce float bounds and prevent NaN/Inf."""
        score = self.confidence_score
        if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
            object.__setattr__(self, "confidence_score", 0.0)
        object.__setattr__(
            self, "confidence_score",
            max(0.0, min(1.0, self.confidence_score)),
        )

    def to_dict(self) -> dict:
        return {
            "test_type": self.test_type,
            "target_module": self.target_module,
            "target_function": self.target_function,
            "description": self.description,
            "invariant_targeted": self.invariant_targeted,
            "risk_justification": self.risk_justification,
            "severity_level": self.severity_level,
            "confidence_score": round(self.confidence_score, 4),
            "requires_human_review": self.requires_human_review,
        }

    def __repr__(self) -> str:
        return (
            f"TestCaseProposal("
            f"{self.test_type} {self.severity_level} "
            f"→ {self.target_module}::{self.target_function}, "
            f"conf={self.confidence_score:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION REPORT (collection output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationReport:
    """
    Complete output of a test generation run.
    Immutable, JSON-serializable, bounded.
    """
    total_proposed: int = 0
    suppressed_count: int = 0
    highest_severity: str = "low"
    proposals: tuple[TestCaseProposal, ...] = field(default_factory=tuple)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "total_proposed": self.total_proposed,
            "suppressed_count": self.suppressed_count,
            "highest_severity": self.highest_severity,
            "proposals": [p.to_dict() for p in self.proposals],
            "generated_at": self.generated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"GenerationReport("
            f"proposed={self.total_proposed}, "
            f"suppressed={self.suppressed_count}, "
            f"highest={self.highest_severity})"
        )



# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class GeneratorEngine:
    """
    Main entry point for generating test proposals from patch suggestions.

    Usage:
        engine = GeneratorEngine()
        report = engine.generate_tests(patch_report)
        for p in report.proposals:
            print(p.to_dict())

    Pure function semantics: no state, no side effects.
    Deterministic ordering. Bounded output.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self._cfg = config or GenerationConfig()

    @property
    def config(self) -> GenerationConfig:
        return self._cfg

    def generate_tests(
        self,
        patch_report,
        structural_state=None,
        gate_decision=None,
    ) -> GenerationReport:
        """
        Generate bounded, deterministic test proposals from patch suggestions.

        Args:
            patch_report: ProposalReport from PatchProposalEngine.
                Must have: suggestions (iterable of PatchSuggestion).
            structural_state: Optional StructuralMetaState.
                If provided and instability is high, reduces max proposals.
            gate_decision: Optional StructuralGateDecision.
                If provided, overrides instability-based reduction.

        Returns:
            GenerationReport with ordered, bounded proposals.
        """
        cfg = self._cfg

        # ── Compute effective caps ────────────────────────────────────────
        cap_unit = cfg.max_unit_tests
        cap_regression = cfg.max_regression_tests
        cap_mutation = cfg.max_mutation_tests

        reduction = self._compute_reduction(structural_state, gate_decision)
        if reduction > 0.0:
            cap_unit = max(1, int(cap_unit * (1.0 - reduction)))
            cap_regression = max(1, int(cap_regression * (1.0 - reduction)))
            cap_mutation = max(1, int(cap_mutation * (1.0 - reduction)))

        # ── Extract suggestions from patch report ─────────────────────────
        suggestions = getattr(patch_report, "suggestions", ())
        if not suggestions:
            return GenerationReport()

        # ── Generate candidate proposals ──────────────────────────────────
        candidates: list[TestCaseProposal] = []

        for suggestion in suggestions:
            module = getattr(suggestion, "module", "unknown")
            location = getattr(suggestion, "location", "unknown")
            issue_type = getattr(suggestion, "issue_type", "")
            severity = getattr(suggestion, "severity", "low")
            base_confidence = getattr(suggestion, "confidence_score", 0.5)

            template = _INVARIANT_MAP.get(issue_type)
            if template is None:
                continue

            sev_weight = _SEVERITY_WEIGHTS.get(severity, 0.25)
            boosted_confidence = _clamp(
                base_confidence + cfg.severity_weighting * sev_weight
            )

            risk_justification = (
                f"Structural issue '{issue_type}' detected in "
                f"{module}::{location} at severity={severity}. "
                f"Risk: {template.risk_name}."
            )

            # ── Unit test proposal ────────────────────────────────────
            candidates.append(TestCaseProposal(
                test_type=TestType.UNIT.value,
                target_module=module,
                target_function=location,
                description=template.unit_description,
                invariant_targeted=template.risk_name,
                risk_justification=risk_justification,
                severity_level=severity,
                confidence_score=round(boosted_confidence, 4),
            ))

            # ── Regression test proposal ──────────────────────────────
            candidates.append(TestCaseProposal(
                test_type=TestType.REGRESSION.value,
                target_module=module,
                target_function=location,
                description=template.regression_description,
                invariant_targeted=template.risk_name,
                risk_justification=risk_justification,
                severity_level=severity,
                confidence_score=round(
                    _clamp(boosted_confidence * 0.90), 4
                ),
            ))

            # ── Mutation test proposal ────────────────────────────────
            candidates.append(TestCaseProposal(
                test_type=TestType.MUTATION.value,
                target_module=module,
                target_function=location,
                description=template.mutation_description,
                invariant_targeted=template.risk_name,
                risk_justification=risk_justification,
                severity_level=severity,
                confidence_score=round(
                    _clamp(boosted_confidence * 0.80), 4
                ),
            ))

        total_generated = len(candidates)

        # ── Filter by confidence floor ────────────────────────────────────
        candidates = [
            c for c in candidates
            if c.confidence_score >= cfg.confidence_floor
        ]

        # ── Deterministic ordering ────────────────────────────────────────
        severity_rank = {
            TestSeverity.CRITICAL.value: 3,
            TestSeverity.HIGH.value: 2,
            TestSeverity.MEDIUM.value: 1,
            TestSeverity.LOW.value: 0,
        }
        type_rank = {
            TestType.UNIT.value: 0,
            TestType.REGRESSION.value: 1,
            TestType.MUTATION.value: 2,
        }
        candidates.sort(key=lambda p: (
            -severity_rank.get(p.severity_level, 0),
            p.target_module,
            type_rank.get(p.test_type, 9),
        ))

        # ── Apply per-type caps ───────────────────────────────────────────
        unit_count = 0
        regression_count = 0
        mutation_count = 0
        emitted: list[TestCaseProposal] = []

        for candidate in candidates:
            if candidate.test_type == TestType.UNIT.value:
                if unit_count >= cap_unit:
                    continue
                unit_count += 1
            elif candidate.test_type == TestType.REGRESSION.value:
                if regression_count >= cap_regression:
                    continue
                regression_count += 1
            elif candidate.test_type == TestType.MUTATION.value:
                if mutation_count >= cap_mutation:
                    continue
                mutation_count += 1
            emitted.append(candidate)

        suppressed = total_generated - len(emitted)

        # ── Compute highest severity ──────────────────────────────────────
        highest = TestSeverity.LOW.value
        if emitted:
            highest_rank = max(
                severity_rank.get(p.severity_level, 0) for p in emitted
            )
            for sev_val, sev_rank in severity_rank.items():
                if sev_rank == highest_rank:
                    highest = sev_val
                    break

        return GenerationReport(
            total_proposed=len(emitted),
            suppressed_count=max(0, suppressed),
            highest_severity=highest,
            proposals=tuple(emitted),
        )

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _compute_reduction(
        self,
        structural_state,
        gate_decision,
    ) -> float:
        """
        Compute proportional reduction of max test proposals.

        If gate_decision is provided, use its patch_aggressiveness
        (lower aggressiveness → higher reduction).
        Otherwise, if structural_state is provided, use instability.
        Otherwise, no reduction (0.0).

        Returns:
            float ∈ [0.0, cfg.instability_reduction]
        """
        cfg = self._cfg

        if gate_decision is not None:
            aggressiveness = getattr(gate_decision, "patch_aggressiveness", 1.0)
            aggressiveness = max(0.0, min(1.0, aggressiveness))
            return cfg.instability_reduction * (1.0 - aggressiveness)

        if structural_state is not None:
            instability = getattr(
                structural_state, "structural_instability_index", 0.0
            )
            instability = max(0.0, min(1.0, instability))
            return cfg.instability_reduction * instability

        return 0.0



# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]. Returns lo on NaN, Inf, or non-numeric."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, v))