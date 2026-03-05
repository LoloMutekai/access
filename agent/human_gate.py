"""
A.C.C.E.S.S. — Human Approval Gate (Phase 6.4)

Deterministic human approval control layer.
Sits between PatchProposalEngine/GeneratorEngine output and any
downstream action. This module is a PURE DATA ORCHESTRATION layer.

It:
    - Bundles patch and test reports into a reviewable PatchBundle
    - Presents structured DiffView summaries (no raw code diffs)
    - Records human decisions via HumanApprovalDecision
    - Validates decision integrity against the bundle
    - NEVER executes patches
    - NEVER writes files
    - NEVER mutates input objects
    - NEVER auto-approves anything

Security properties:
    P1: Deterministic — same inputs → same PatchBundle
    P2: Immutable bundle — frozen dataclasses throughout
    P3: No hidden mutation — all state in return values only
    P4: No auto-execution path — safe_to_execute is hardcoded False
    P5: Explicit human consent — ACCEPT requires approved_modules listing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ApprovalAction(str, Enum):
    """Human decision action for a PatchBundle."""
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    DEFER = "defer"


# ─────────────────────────────────────────────────────────────────────────────
# DIFF VIEW (immutable summary, no executable content)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiffView:
    """
    Structured human-readable summary of a single patch suggestion.
    Contains NO executable code. Contains NO raw diffs.
    Only structured metadata suitable for human review.
    """
    module_name: str
    location: str
    original_summary: str
    proposed_change_summary: str
    structural_risk_level: str
    severity_level: str
    confidence_score: float

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "location": self.location,
            "original_summary": self.original_summary,
            "proposed_change_summary": self.proposed_change_summary,
            "structural_risk_level": self.structural_risk_level,
            "severity_level": self.severity_level,
            "confidence_score": round(self.confidence_score, 4),
        }

    def __repr__(self) -> str:
        return (
            f"DiffView("
            f"module={self.module_name!r}, "
            f"location={self.location!r}, "
            f"severity={self.severity_level!r}, "
            f"conf={self.confidence_score:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PATCH BUNDLE (immutable review unit)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatchBundle:
    """
    Immutable bundle of patch and test reports ready for human review.
    Contains NO executable code. Safe to serialise and log.

    patch_report : ProposalReport from PatchProposalEngine (duck-typed)
    test_report  : GenerationReport from GeneratorEngine (duck-typed)
    diff_views   : Tuple of DiffView summaries, one per patch suggestion
    generated_at : UTC timestamp of bundle creation
    """
    patch_report: object
    test_report: object
    diff_views: tuple
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        patch_dict = (
            self.patch_report.to_dict()
            if hasattr(self.patch_report, "to_dict")
            else {}
        )
        test_dict = (
            self.test_report.to_dict()
            if hasattr(self.test_report, "to_dict")
            else {}
        )
        return {
            "patch_report": patch_dict,
            "test_report": test_dict,
            "diff_views": [dv.to_dict() for dv in self.diff_views],
            "generated_at": self.generated_at.isoformat(),
            "diff_view_count": len(self.diff_views),
        }

    def __repr__(self) -> str:
        return (
            f"PatchBundle("
            f"diff_views={len(self.diff_views)}, "
            f"generated_at={self.generated_at.isoformat()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN APPROVAL DECISION (immutable record of human intent)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HumanApprovalDecision:
    """
    Immutable record of a human reviewer's decision on a PatchBundle.

    Default action is DEFER — no silent approval is ever possible.
    approved_modules and rejected_modules identify individual patch targets
    by their module_name, allowing partial approval.

    This record is purely descriptive. It cannot trigger any execution.
    """
    action: ApprovalAction = ApprovalAction.DEFER
    reviewed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewer_notes: str = ""
    approved_modules: tuple = field(default_factory=tuple)
    rejected_modules: tuple = field(default_factory=tuple)
    requires_followup: bool = False

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reviewed_at": self.reviewed_at.isoformat(),
            "reviewer_notes": self.reviewer_notes,
            "approved_modules": list(self.approved_modules),
            "rejected_modules": list(self.rejected_modules),
            "requires_followup": self.requires_followup,
        }

    def __repr__(self) -> str:
        return (
            f"HumanApprovalDecision("
            f"action={self.action.value!r}, "
            f"approved={len(self.approved_modules)}, "
            f"rejected={len(self.rejected_modules)}, "
            f"followup={self.requires_followup})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION ERRORS
# ─────────────────────────────────────────────────────────────────────────────

class ApprovalValidationError(ValueError):
    """Raised when a HumanApprovalDecision violates safety rules."""


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN APPROVAL GATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class HumanApprovalGateEngine:
    """
    Orchestrates the human review workflow between patch/test output and
    any downstream action.

    Lifecycle:
        1. bundle = engine.create_patch_bundle(patch_report, test_report)
        2. Human reviews bundle.to_dict() / bundle.diff_views
        3. decision = HumanApprovalDecision(action=..., approved_modules=(...), ...)
        4. summary = engine.evaluate_decision(bundle, decision)
        5. summary["safe_to_execute"] is always False — no auto-execution

    No state is held between calls. All results are in return values only.
    """

    # ── Bundle creation ───────────────────────────────────────────────────────

    def create_patch_bundle(
        self,
        patch_report,
        test_report,
    ) -> PatchBundle:
        """
        Build an immutable PatchBundle from patch and test reports.

        For each PatchSuggestion in patch_report.suggestions, creates one
        DiffView containing a structured summary (no raw code, no diffs).

        Args:
            patch_report : ProposalReport with .suggestions iterable
            test_report  : GenerationReport with .proposals iterable

        Returns:
            PatchBundle — frozen, JSON-serializable, safe to log and review.
        """
        suggestions = getattr(patch_report, "suggestions", ()) or ()
        diff_views = tuple(
            self._build_diff_view(s) for s in suggestions
        )

        return PatchBundle(
            patch_report=patch_report,
            test_report=test_report,
            diff_views=diff_views,
            generated_at=datetime.now(UTC),
        )

    # ── Decision evaluation ───────────────────────────────────────────────────

    def evaluate_decision(
        self,
        bundle: PatchBundle,
        decision: HumanApprovalDecision,
    ) -> dict:
        """
        Validate and summarise a human decision against the bundle.

        Safety rules enforced:
            ACCEPT         → approved_modules non-empty; rejected_modules empty
            REJECT         → rejected_modules non-empty
            REQUEST_CHANGES→ reviewer_notes non-empty
            DEFER          → no modules approved

        Module names in approved_modules / rejected_modules must exist in the
        bundle's diff_views. Unknown module names are rejected.

        Args:
            bundle   : PatchBundle produced by create_patch_bundle()
            decision : HumanApprovalDecision from the human reviewer

        Returns:
            dict with keys:
                action, approved_count, rejected_count,
                requires_followup, safe_to_execute (always False), timestamp

        Raises:
            ApprovalValidationError on any safety rule violation.
        """
        self._validate_decision(bundle, decision)

        return {
            "action": decision.action.value,
            "approved_count": len(decision.approved_modules),
            "rejected_count": len(decision.rejected_modules),
            "requires_followup": decision.requires_followup,
            "safe_to_execute": False,
            "timestamp": decision.reviewed_at.isoformat(),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_diff_view(self, suggestion) -> DiffView:
        """
        Construct a DiffView from a single PatchSuggestion.
        No executable content. Summaries only.
        """
        module_name = getattr(suggestion, "module", "unknown")
        location    = getattr(suggestion, "location", "unknown")
        issue_type  = getattr(suggestion, "issue_type", "unknown")
        severity    = getattr(suggestion, "severity", "low")
        rationale   = getattr(suggestion, "rationale", "")
        strategy    = getattr(suggestion, "suggested_refactor_strategy", "")
        confidence  = float(getattr(suggestion, "confidence_score", 0.0))
        risk_est    = getattr(suggestion, "risk_reduction_estimate", 0.0)

        confidence = max(0.0, min(1.0, confidence))

        original_summary = (
            f"Issue detected: {issue_type} in {module_name}::{location}. "
            f"Rationale: {rationale}"
        ).strip()

        proposed_change_summary = (
            f"Suggested refactor strategy: {strategy} "
            f"(estimated risk reduction: {float(risk_est):.2%})"
        ).strip()

        structural_risk_level = _severity_to_risk(severity)

        return DiffView(
            module_name=module_name,
            location=location,
            original_summary=original_summary,
            proposed_change_summary=proposed_change_summary,
            structural_risk_level=structural_risk_level,
            severity_level=severity,
            confidence_score=round(confidence, 4),
        )

    def _validate_decision(
        self,
        bundle: PatchBundle,
        decision: HumanApprovalDecision,
    ) -> None:
        """
        Enforce all safety rules for a decision.
        Raises ApprovalValidationError on any violation.
        """
        action = decision.action
        known_modules = {dv.module_name for dv in bundle.diff_views}

        # ── Rule 1: ACCEPT requires approved_modules, forbids rejected_modules
        if action == ApprovalAction.ACCEPT:
            if not decision.approved_modules:
                raise ApprovalValidationError(
                    "ACCEPT action requires at least one module in approved_modules. "
                    "Silent auto-approval is not permitted."
                )
            if decision.rejected_modules:
                raise ApprovalValidationError(
                    "ACCEPT action must have an empty rejected_modules list. "
                    "Use REQUEST_CHANGES or DEFER for partial review."
                )

        # ── Rule 2: REJECT requires rejected_modules
        if action == ApprovalAction.REJECT:
            if not decision.rejected_modules:
                raise ApprovalValidationError(
                    "REJECT action requires at least one module in rejected_modules."
                )

        # ── Rule 3: REQUEST_CHANGES requires reviewer_notes
        if action == ApprovalAction.REQUEST_CHANGES:
            if not decision.reviewer_notes or not decision.reviewer_notes.strip():
                raise ApprovalValidationError(
                    "REQUEST_CHANGES action requires non-empty reviewer_notes."
                )

        # ── Rule 4: DEFER must not approve any modules
        if action == ApprovalAction.DEFER:
            if decision.approved_modules:
                raise ApprovalValidationError(
                    "DEFER action must not list any approved_modules."
                )

        # ── Rule 5: All named modules must exist in the bundle
        all_named = set(decision.approved_modules) | set(decision.rejected_modules)
        if known_modules and all_named:
            unknown = all_named - known_modules
            if unknown:
                raise ApprovalValidationError(
                    f"The following modules are not present in the bundle "
                    f"and cannot be approved or rejected: {sorted(unknown)}"
                )

        # ── Rule 6: No overlap between approved and rejected
        overlap = set(decision.approved_modules) & set(decision.rejected_modules)
        if overlap:
            raise ApprovalValidationError(
                f"Modules cannot appear in both approved_modules and "
                f"rejected_modules: {sorted(overlap)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _severity_to_risk(severity: str) -> str:
    """Map patch severity string to a human-readable structural risk label."""
    _MAP = {
        "critical": "high structural risk — immediate attention required",
        "high":     "elevated structural risk — review before merge",
        "medium":   "moderate structural risk — review recommended",
        "low":      "low structural risk — cosmetic or minor concern",
    }
    return _MAP.get(severity.lower(), "unknown risk level")