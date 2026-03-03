"""
A.C.C.E.S.S. — Patch Proposal Engine (Phase 6.2)

Generates deterministic, bounded, explainable refactoring suggestions
based on Static Inspection (6.1) + Structural Meta-State (6.1→Meta).

This module PROPOSES. It EXPLAINS. It STRUCTURES.
It does NOT apply patches. It does NOT modify files. It does NOT execute code.

Design:
    - PatchSuggestion is FROZEN — immutable, JSON-serializable
    - PatchProposalEngine is pure: (report, meta_state, config) → suggestions
    - All rules are explicit, deterministic, and bounded
    - Aggressiveness modulated by structural_instability_index
    - Hard cap on total suggestions per run
    - No LLM-based generation — purely rule-driven
    - No speculative rewrites — every suggestion backed by metric
    - requires_human_review is ALWAYS True

Rule categories:
    1. HIGH_COMPLEXITY    → split function / extract method
    2. DEEP_NESTING       → extract method / flatten
    3. GOD_CLASS          → split class / extract concerns
    4. CIRCULAR_DEP       → dependency inversion / interface extraction
    5. LAYER_VIOLATION    → move module / introduce adapter
    6. HIGH_IO_DENSITY    → isolate side effects / ports & adapters
    7. HIGH_STATE_DENSITY → reduce mutation scope / immutable data
    8. OVERSIZED_MODULE   → split module / extract submodules
    9. OVER_PARAMETERIZED → introduce parameter object / builder
    10. LONG_FUNCTION     → extract method / decompose steps

Aggressiveness modulation:
    instability < 0.30  → micro-improvements only (severity ≤ medium)
    0.30 ≤ inst < 0.55  → standard suggestions (all severities)
    inst ≥ 0.55         → structural refactors prioritized (high/critical first)

Safety:
    - Hard cap: max_suggestions (default 20, reduced by gate)
    - Deterministic ordering: by severity desc, then module asc
    - No suggestion without supporting metric evidence
    - No hallucinated file paths — all paths from InspectionReport
    - confidence_score ∈ [0.0, 1.0] — based on metric strength
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class IssueType(str, Enum):
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
# REFACTOR STRATEGY TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_REFACTOR_STRATEGIES: dict[IssueType, str] = {
    IssueType.HIGH_COMPLEXITY: (
        "Split function into smaller sub-functions, each handling one logical path. "
        "Extract complex boolean conditions into named predicates."
    ),
    IssueType.DEEP_NESTING: (
        "Extract deeply nested blocks into named helper methods. "
        "Consider guard clauses (early returns) to flatten control flow."
    ),
    IssueType.GOD_CLASS: (
        "Identify cohesive method clusters and extract them into focused classes. "
        "Apply Single Responsibility Principle — one reason to change per class."
    ),
    IssueType.CIRCULAR_DEPENDENCY: (
        "Introduce an interface/protocol at the dependency boundary. "
        "Consider dependency inversion: both modules depend on an abstraction."
    ),
    IssueType.LAYER_VIOLATION: (
        "Move the import to respect the architectural layer hierarchy. "
        "If the dependency is necessary, introduce an adapter or event bridge."
    ),
    IssueType.HIGH_IO_DENSITY: (
        "Isolate I/O operations behind a ports-and-adapters boundary. "
        "Pure logic should accept data, not fetch it — inject dependencies."
    ),
    IssueType.HIGH_STATE_DENSITY: (
        "Reduce mutable state scope. Prefer frozen/immutable data structures. "
        "Replace attribute mutation with method return values where possible."
    ),
    IssueType.OVERSIZED_MODULE: (
        "Split into focused sub-modules grouped by concern. "
        "Create a package directory with an __init__.py that re-exports the public API."
    ),
    IssueType.OVER_PARAMETERIZED: (
        "Introduce a parameter object (dataclass/NamedTuple) to group related parameters. "
        "Consider a builder pattern if parameters have complex defaults."
    ),
    IssueType.LONG_FUNCTION: (
        "Decompose into sequential helper functions, each performing one logical step. "
        "Name each step clearly — the function should read like a summary."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatchProposalConfig:
    """
    Thresholds and limits for the patch proposal engine.
    Frozen — never mutated.

    max_suggestions (20):
        Hard cap. The StructuralGate can reduce this further.
        No run ever produces more suggestions than this.

    severity_thresholds:
        When to classify an issue as low/medium/high/critical.
        Based on how far the metric exceeds its inspection threshold.

    confidence_floor (0.30):
        Below this confidence, suggestions are suppressed entirely.
        Prevents noise from marginal detections.
    """

    # Hard limits
    max_suggestions: int = 20

    # Complexity severity mapping (cyclomatic complexity thresholds)
    complexity_medium: int = 12
    complexity_high: int = 18
    complexity_critical: int = 25

    # Nesting severity mapping
    nesting_medium: int = 4
    nesting_high: int = 6
    nesting_critical: int = 8

    # God class: method count thresholds
    god_class_medium: int = 15
    god_class_high: int = 25
    god_class_critical: int = 40

    # Module size: lines thresholds
    module_size_medium: int = 500
    module_size_high: int = 800
    module_size_critical: int = 1200

    # Function length: lines thresholds
    function_length_medium: int = 50
    function_length_high: int = 80
    function_length_critical: int = 120

    # Parameter count thresholds
    params_medium: int = 6
    params_high: int = 9
    params_critical: int = 12

    # IO density thresholds
    io_density_medium: float = 0.30
    io_density_high: float = 0.50
    io_density_critical: float = 0.70

    # State density thresholds
    state_density_medium: float = 0.40
    state_density_high: float = 0.60
    state_density_critical: float = 0.80

    # Confidence floor: below this → suppress suggestion
    confidence_floor: float = 0.30

    # Risk reduction estimates (conservative)
    risk_reduction_low: float = 0.02
    risk_reduction_medium: float = 0.05
    risk_reduction_high: float = 0.10
    risk_reduction_critical: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# PATCH SUGGESTION (immutable output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatchSuggestion:
    """
    A single refactoring suggestion. Immutable, JSON-serializable.

    Fields:
        module:                  Module name (stem of the .py file)
        location:                Function/class name or line range
        issue_type:              Category of the detected issue
        severity:                Severity classification
        rationale:               Human-readable explanation with metric evidence
        suggested_refactor_strategy: Actionable refactoring advice
        risk_reduction_estimate: Conservative estimate of risk reduction if applied
        confidence_score:        How confident the engine is (0.0–1.0)
        requires_human_review:   ALWAYS True — no auto-apply
        metric_evidence:         Dict of supporting metrics
    """
    module: str
    location: str
    issue_type: str
    severity: str
    rationale: str
    suggested_refactor_strategy: str
    risk_reduction_estimate: float = 0.05
    confidence_score: float = 0.5
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
            "risk_reduction_estimate": round(self.risk_reduction_estimate, 4),
            "confidence_score": round(self.confidence_score, 4),
            "requires_human_review": self.requires_human_review,
            "metric_evidence": [
                {"key": k, "value": v} for k, v in self.metric_evidence
            ],
        }

    def __repr__(self) -> str:
        return (
            f"PatchSuggestion("
            f"{self.severity} {self.issue_type} "
            f"in {self.module}::{self.location}, "
            f"conf={self.confidence_score:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROPOSAL REPORT (collection output)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProposalReport:
    """
    Complete output of a patch proposal run.
    Immutable, JSON-serializable, bounded.
    """
    suggestions: tuple[PatchSuggestion, ...] = field(default_factory=tuple)
    total_issues_detected: int = 0
    total_suggestions_emitted: int = 0
    suggestions_suppressed: int = 0
    aggressiveness_level: float = 1.0
    max_suggestions_allowed: int = 20
    structural_instability: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "total_issues_detected": self.total_issues_detected,
            "total_suggestions_emitted": self.total_suggestions_emitted,
            "suggestions_suppressed": self.suggestions_suppressed,
            "aggressiveness_level": round(self.aggressiveness_level, 4),
            "max_suggestions_allowed": self.max_suggestions_allowed,
            "structural_instability": round(self.structural_instability, 4),
            "generated_at": self.generated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"ProposalReport("
            f"emitted={self.total_suggestions_emitted}, "
            f"suppressed={self.suggestions_suppressed}, "
            f"aggr={self.aggressiveness_level:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PatchProposalEngine:
    """
    Main entry point for generating patch suggestions.

    Usage:
        engine = PatchProposalEngine()
        report = engine.propose(inspection_report, structural_meta_state)
        for s in report.suggestions:
            print(s.to_dict())

    Pure function semantics: no state, no side effects.
    All randomness eliminated. Deterministic ordering.
    """

    def __init__(self, config: Optional[PatchProposalConfig] = None):
        self._cfg = config or PatchProposalConfig()

    @property
    def config(self) -> PatchProposalConfig:
        return self._cfg

    def propose(
        self,
        report,
        structural_state=None,
        gate_decision=None,
    ) -> ProposalReport:
        """
        Generate bounded, deterministic patch suggestions.

        Args:
            report: InspectionReport from StaticInspector
            structural_state: Optional StructuralMetaState for aggressiveness modulation
            gate_decision: Optional StructuralGateDecision for limit override

        Returns:
            ProposalReport with bounded, ordered suggestions.
        """
        cfg = self._cfg

        # ── Determine aggressiveness and limits ───────────────────────────
        if gate_decision is not None:
            aggressiveness = gate_decision.patch_aggressiveness
            max_sugg = gate_decision.max_suggestions
        elif structural_state is not None:
            inst = structural_state.structural_instability_index
            aggressiveness = max(0.0, 1.0 - inst ** 1.5)
            max_sugg = int(cfg.max_suggestions * aggressiveness)
            max_sugg = max(3, max_sugg)
        else:
            aggressiveness = 1.0
            max_sugg = cfg.max_suggestions

        instability = 0.0
        if structural_state is not None:
            instability = structural_state.structural_instability_index

        # ── Collect all candidate suggestions ─────────────────────────────
        candidates: list[PatchSuggestion] = []

        # Extract file data
        files = getattr(report, "files", ())
        cycles = getattr(report, "cycles", ())
        violations = getattr(report, "layer_violations", ())

        # Rule 1-2: Function-level issues (complexity, nesting, params, length)
        for fm in files:
            module = Path(getattr(fm, "path", "unknown.py")).stem
            for func in getattr(fm, "functions", ()):
                candidates.extend(self._function_rules(module, func))

        # Rule 3: God classes
        for fm in files:
            module = Path(getattr(fm, "path", "unknown.py")).stem
            for cls in getattr(fm, "classes", ()):
                cand = self._god_class_rule(module, cls)
                if cand:
                    candidates.append(cand)

        # Rule 4: Circular dependencies
        for cycle in cycles:
            cand = self._circular_dep_rule(cycle)
            if cand:
                candidates.append(cand)

        # Rule 5: Layer violations
        for violation in violations:
            cand = self._layer_violation_rule(violation)
            if cand:
                candidates.append(cand)

        # Rule 6-7: File-level issues (IO density, state density, oversized)
        for fm in files:
            module = Path(getattr(fm, "path", "unknown.py")).stem
            candidates.extend(self._file_level_rules(module, fm))

        total_detected = len(candidates)

        # ── Filter by confidence floor ────────────────────────────────────
        candidates = [c for c in candidates if c.confidence_score >= cfg.confidence_floor]

        # ── Aggressiveness filtering ──────────────────────────────────────
        if aggressiveness < 0.5:
            # Low aggressiveness: only high/critical severity
            candidates = [
                c for c in candidates
                if c.severity in (Severity.HIGH.value, Severity.CRITICAL.value)
            ]
        elif aggressiveness < 0.8:
            # Medium aggressiveness: medium and above
            candidates = [
                c for c in candidates
                if c.severity in (Severity.MEDIUM.value, Severity.HIGH.value, Severity.CRITICAL.value)
            ]
        # else: full aggressiveness — keep all

        # ── Deterministic ordering ────────────────────────────────────────
        severity_rank = {
            Severity.CRITICAL.value: 3,
            Severity.HIGH.value: 2,
            Severity.MEDIUM.value: 1,
            Severity.LOW.value: 0,
        }
        candidates.sort(key=lambda s: (
            -severity_rank.get(s.severity, 0),
            s.module,
            s.location,
            s.issue_type,
        ))

        # ── Apply hard cap ────────────────────────────────────────────────
        emitted = candidates[:max_sugg]
        suppressed = total_detected - len(emitted)

        return ProposalReport(
            suggestions=tuple(emitted),
            total_issues_detected=total_detected,
            total_suggestions_emitted=len(emitted),
            suggestions_suppressed=max(0, suppressed),
            aggressiveness_level=aggressiveness,
            max_suggestions_allowed=max_sugg,
            structural_instability=instability,
        )

    # ─────────────────────────────────────────────────────────────────────
    # RULE IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────────

    def _function_rules(self, module: str, func) -> list[PatchSuggestion]:
        """Generate suggestions for function-level issues."""
        suggestions = []
        cfg = self._cfg

        name = getattr(func, "name", "<unknown>")
        cc = getattr(func, "cyclomatic_complexity", 1)
        nesting = getattr(func, "max_nesting", 0)
        params = getattr(func, "parameter_count", 0)
        lines = getattr(func, "lines", 0)

        # ── High Complexity ───────────────────────────────────────────
        if cc > cfg.complexity_medium:
            severity = self._classify_severity(
                cc, cfg.complexity_medium, cfg.complexity_high, cfg.complexity_critical,
            )
            confidence = min(1.0, (cc - cfg.complexity_medium) / max(1, cfg.complexity_critical - cfg.complexity_medium) * 0.7 + 0.3)
            suggestions.append(PatchSuggestion(
                module=module,
                location=name,
                issue_type=IssueType.HIGH_COMPLEXITY.value,
                severity=severity.value,
                rationale=f"Cyclomatic complexity of {cc} exceeds threshold ({cfg.complexity_medium}). "
                          f"High complexity correlates with bug density and maintenance cost.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.HIGH_COMPLEXITY],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("cyclomatic_complexity", cc), ("threshold", cfg.complexity_medium)),
            ))

        # ── Deep Nesting ──────────────────────────────────────────────
        if nesting > cfg.nesting_medium:
            severity = self._classify_severity(
                nesting, cfg.nesting_medium, cfg.nesting_high, cfg.nesting_critical,
            )
            confidence = min(1.0, (nesting - cfg.nesting_medium) / max(1, cfg.nesting_critical - cfg.nesting_medium) * 0.6 + 0.35)
            suggestions.append(PatchSuggestion(
                module=module,
                location=name,
                issue_type=IssueType.DEEP_NESTING.value,
                severity=severity.value,
                rationale=f"Nesting depth of {nesting} exceeds threshold ({cfg.nesting_medium}). "
                          f"Deep nesting reduces readability and increases cognitive load.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.DEEP_NESTING],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("max_nesting", nesting), ("threshold", cfg.nesting_medium)),
            ))

        # ── Over-Parameterized ────────────────────────────────────────
        if params > cfg.params_medium:
            severity = self._classify_severity(
                params, cfg.params_medium, cfg.params_high, cfg.params_critical,
            )
            confidence = min(1.0, 0.4 + (params - cfg.params_medium) * 0.1)
            suggestions.append(PatchSuggestion(
                module=module,
                location=name,
                issue_type=IssueType.OVER_PARAMETERIZED.value,
                severity=severity.value,
                rationale=f"Function has {params} parameters (threshold: {cfg.params_medium}). "
                          f"Many parameters suggest the function is doing too much.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.OVER_PARAMETERIZED],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(min(1.0, confidence), 4),
                metric_evidence=(("parameter_count", params), ("threshold", cfg.params_medium)),
            ))

        # ── Long Function ─────────────────────────────────────────────
        if lines > cfg.function_length_medium:
            severity = self._classify_severity(
                lines, cfg.function_length_medium, cfg.function_length_high, cfg.function_length_critical,
            )
            confidence = min(1.0, 0.35 + (lines - cfg.function_length_medium) / max(1, cfg.function_length_critical) * 0.5)
            suggestions.append(PatchSuggestion(
                module=module,
                location=name,
                issue_type=IssueType.LONG_FUNCTION.value,
                severity=severity.value,
                rationale=f"Function is {lines} lines (threshold: {cfg.function_length_medium}). "
                          f"Long functions are harder to test and maintain.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.LONG_FUNCTION],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("lines", lines), ("threshold", cfg.function_length_medium)),
            ))

        return suggestions

    def _god_class_rule(self, module: str, cls) -> Optional[PatchSuggestion]:
        """Generate suggestion for god class detection."""
        cfg = self._cfg
        is_god = getattr(cls, "is_god_class", False)
        if not is_god:
            return None

        methods = getattr(cls, "method_count", 0)
        attrs = getattr(cls, "attribute_count", 0)
        name = getattr(cls, "name", "<unknown>")

        severity = self._classify_severity(
            methods, cfg.god_class_medium, cfg.god_class_high, cfg.god_class_critical,
        )
        confidence = min(1.0, 0.5 + methods / 50.0)

        return PatchSuggestion(
            module=module,
            location=name,
            issue_type=IssueType.GOD_CLASS.value,
            severity=severity.value,
            rationale=f"Class '{name}' has {methods} methods and {attrs} attributes. "
                      f"God classes violate SRP and create maintenance bottlenecks.",
            suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.GOD_CLASS],
            risk_reduction_estimate=self._risk_estimate(severity),
            confidence_score=round(confidence, 4),
            metric_evidence=(("method_count", methods), ("attribute_count", attrs)),
        )

    def _circular_dep_rule(self, cycle) -> Optional[PatchSuggestion]:
        """Generate suggestion for a circular dependency."""
        if not cycle:
            return None

        modules = list(cycle) if hasattr(cycle, "__iter__") else [str(cycle)]
        cycle_str = " → ".join(modules)
        length = len(modules)

        # Longer cycles are harder to break → higher severity
        if length >= 4:
            severity = Severity.CRITICAL
        elif length >= 3:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        confidence = min(1.0, 0.6 + length * 0.1)

        return PatchSuggestion(
            module=modules[0] if modules else "unknown",
            location=f"cycle({cycle_str})",
            issue_type=IssueType.CIRCULAR_DEPENDENCY.value,
            severity=severity.value,
            rationale=f"Circular dependency detected: {cycle_str}. "
                      f"Cycles prevent clean module boundaries and complicate testing.",
            suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.CIRCULAR_DEPENDENCY],
            risk_reduction_estimate=self._risk_estimate(severity),
            confidence_score=round(confidence, 4),
            metric_evidence=(("cycle_length", length), ("modules", cycle_str)),
        )

    def _layer_violation_rule(self, violation) -> Optional[PatchSuggestion]:
        """Generate suggestion for an architectural layer violation."""
        if not violation or len(violation) < 2:
            return None

        source, target = violation[0], violation[1]

        return PatchSuggestion(
            module=source,
            location=f"import({target})",
            issue_type=IssueType.LAYER_VIOLATION.value,
            severity=Severity.HIGH.value,
            rationale=f"Module '{source}' imports from '{target}', violating the layer hierarchy. "
                      f"Lower layers should not depend on higher layers.",
            suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.LAYER_VIOLATION],
            risk_reduction_estimate=self._risk_estimate(Severity.HIGH),
            confidence_score=0.85,
            metric_evidence=(("source", source), ("target", target)),
        )

    def _file_level_rules(self, module: str, fm) -> list[PatchSuggestion]:
        """Generate suggestions for file-level issues."""
        suggestions = []
        cfg = self._cfg

        io_density = getattr(fm, "io_density", 0.0)
        state_density = getattr(fm, "state_density", 0.0)
        lines_code = getattr(fm, "lines_code", 0)
        is_oversized = getattr(fm, "is_oversized", False)

        # ── High IO Density ───────────────────────────────────────────
        if io_density > cfg.io_density_medium:
            severity = self._classify_severity_float(
                io_density, cfg.io_density_medium, cfg.io_density_high, cfg.io_density_critical,
            )
            confidence = min(1.0, 0.4 + io_density)
            suggestions.append(PatchSuggestion(
                module=module,
                location="(file-level)",
                issue_type=IssueType.HIGH_IO_DENSITY.value,
                severity=severity.value,
                rationale=f"IO density of {io_density:.2f} exceeds threshold ({cfg.io_density_medium:.2f}). "
                          f"Mixed IO and logic reduces testability.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.HIGH_IO_DENSITY],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("io_density", round(io_density, 4)), ("threshold", cfg.io_density_medium)),
            ))

        # ── High State Density ────────────────────────────────────────
        if state_density > cfg.state_density_medium:
            severity = self._classify_severity_float(
                state_density, cfg.state_density_medium, cfg.state_density_high, cfg.state_density_critical,
            )
            confidence = min(1.0, 0.35 + state_density * 0.5)
            suggestions.append(PatchSuggestion(
                module=module,
                location="(file-level)",
                issue_type=IssueType.HIGH_STATE_DENSITY.value,
                severity=severity.value,
                rationale=f"State mutation density of {state_density:.2f} exceeds threshold ({cfg.state_density_medium:.2f}). "
                          f"High mutation density increases coupling and makes testing harder.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.HIGH_STATE_DENSITY],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("state_density", round(state_density, 4)), ("threshold", cfg.state_density_medium)),
            ))

        # ── Oversized Module ──────────────────────────────────────────
        if is_oversized and lines_code > cfg.module_size_medium:
            severity = self._classify_severity(
                lines_code, cfg.module_size_medium, cfg.module_size_high, cfg.module_size_critical,
            )
            confidence = min(1.0, 0.4 + lines_code / 2000.0)
            suggestions.append(PatchSuggestion(
                module=module,
                location="(file-level)",
                issue_type=IssueType.OVERSIZED_MODULE.value,
                severity=severity.value,
                rationale=f"Module has {lines_code} code lines (threshold: {cfg.module_size_medium}). "
                          f"Large modules are harder to navigate and maintain.",
                suggested_refactor_strategy=_REFACTOR_STRATEGIES[IssueType.OVERSIZED_MODULE],
                risk_reduction_estimate=self._risk_estimate(severity),
                confidence_score=round(confidence, 4),
                metric_evidence=(("lines_code", lines_code), ("threshold", cfg.module_size_medium)),
            ))

        return suggestions

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _classify_severity(self, value: int, medium: int, high: int, critical: int) -> Severity:
        """Classify integer metric into severity tier."""
        if value >= critical:
            return Severity.CRITICAL
        elif value >= high:
            return Severity.HIGH
        elif value >= medium:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _classify_severity_float(self, value: float, medium: float, high: float, critical: float) -> Severity:
        """Classify float metric into severity tier."""
        if value >= critical:
            return Severity.CRITICAL
        elif value >= high:
            return Severity.HIGH
        elif value >= medium:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _risk_estimate(self, severity: Severity) -> float:
        """Map severity to conservative risk reduction estimate."""
        cfg = self._cfg
        return {
            Severity.LOW: cfg.risk_reduction_low,
            Severity.MEDIUM: cfg.risk_reduction_medium,
            Severity.HIGH: cfg.risk_reduction_high,
            Severity.CRITICAL: cfg.risk_reduction_critical,
        }[severity]