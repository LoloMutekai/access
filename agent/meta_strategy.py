"""
A.C.C.E.S.S. — Strategic Adjustment Layer (Phase 4.5)

Observes MetaDiagnostics and returns suggested corrections.
NEVER mutates Phase 4 state directly.
Returns StrategicAdjustment — the caller (CognitiveIdentityManager) decides what to apply.

Design:
    - Pure function: diagnostics in → adjustments out
    - No side effects, no state, no external calls
    - Adjustments are SUGGESTIONS, not commands
    - Meta-goals are injected via GoalQueue (not directly into Phase 4 state)
    - All thresholds configurable

Adjustment types:
    1. Drift cap reduction     — slow personality changes when coherence is low
    2. Dependency mitigation   — increase relief weight when dependency is rising
    3. Goal injection throttle — reduce new goal creation when queue is failing
    4. Guidance mode hint      — suggest structured mode during instability
    5. Meta-goal injection     — specific corrective goals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from .meta_diagnostics import MetaDiagnostics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ADJUSTMENT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetaGoal:
    """A corrective goal suggested by the meta-cognitive layer."""
    description: str
    priority: float = 0.7
    origin: str = "meta_cognition"

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "priority": round(self.priority, 3),
            "origin": self.origin,
        }


@dataclass(frozen=True)
class StrategicAdjustment:
    """
    Immutable set of suggested corrections. Applied by CognitiveIdentityManager.
    All fields are suggestions — None means "no change recommended".
    """

    # Personality drift cap override (None = keep current, e.g. 0.005 = halved)
    suggested_drift_cap: Optional[float] = None

    # Dependency mitigation weight multiplier (1.0 = normal, 2.0 = doubled)
    dependency_mitigation_multiplier: float = 1.0

    # Goal injection allowed this turn? (False = skip goal creation)
    allow_goal_injection: bool = True

    # Suggested guidance mode hint (None = no suggestion)
    suggested_guidance_mode: Optional[str] = None

    # Meta-goals to inject into GoalQueue
    meta_goals: tuple[MetaGoal, ...] = field(default_factory=tuple)

    # Source diagnostics coherence (for logging/telemetry)
    coherence_at_evaluation: float = 1.0

    # Was any adjustment actually triggered?
    any_adjustment: bool = False

    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "suggested_drift_cap": self.suggested_drift_cap,
            "dependency_mitigation_multiplier": round(self.dependency_mitigation_multiplier, 3),
            "allow_goal_injection": self.allow_goal_injection,
            "suggested_guidance_mode": self.suggested_guidance_mode,
            "meta_goals": [g.to_dict() for g in self.meta_goals],
            "coherence_at_evaluation": round(self.coherence_at_evaluation, 4),
            "any_adjustment": self.any_adjustment,
            "generated_at": self.generated_at.isoformat(),
        }

    def __repr__(self) -> str:
        parts = [f"coherence={self.coherence_at_evaluation:.2f}"]
        if self.suggested_drift_cap is not None:
            parts.append(f"drift_cap={self.suggested_drift_cap}")
        if self.dependency_mitigation_multiplier != 1.0:
            parts.append(f"dep_mult={self.dependency_mitigation_multiplier:.1f}")
        if not self.allow_goal_injection:
            parts.append("goals=BLOCKED")
        if self.suggested_guidance_mode:
            parts.append(f"mode_hint={self.suggested_guidance_mode}")
        if self.meta_goals:
            parts.append(f"meta_goals={len(self.meta_goals)}")
        return f"StrategicAdjustment({', '.join(parts)})"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Thresholds and parameters for strategic adjustments."""

    # Coherence threshold below which adjustments activate
    coherence_threshold: float = 0.6

    # Drift cap reduction when coherence is low
    reduced_drift_cap: float = 0.005       # half of normal 0.01

    # Dependency mitigation multiplier when trend is rising
    dependency_boost_multiplier: float = 2.0

    # Goal injection blocked when resolution rate is below this
    goal_injection_min_resolution: float = 0.20

    # Dependency trend threshold for meta-goal injection
    dependency_meta_goal_threshold: float = 0.015

    # Goal failure threshold for cognitive load meta-goal
    goal_failure_meta_goal_threshold: float = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class StrategicAdjustmentEngine:
    """
    Pure function: MetaDiagnostics → StrategicAdjustment.
    No state. No side effects. No mutations.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self._cfg = config or StrategyConfig()

    def evaluate(
        self,
        diagnostics: MetaDiagnostics,
        adapted_context=None,
    ) -> StrategicAdjustment:
        """
        Analyze diagnostics and return suggested adjustments.
        Returns a no-op StrategicAdjustment if everything is healthy.

        Args:
            diagnostics: MetaDiagnostics from the diagnostics engine.
            adapted_context: Optional AdaptedContext from Phase 5.
                If provided:
                    - Uses adapted_threshold instead of static threshold
                    - Applies dampening_factor to adjustment intensity
                    - Gates non-safety adjustments via stability_gate_open
                    - Blocks ALL adjustments via circuit_breaker_active
                If None: Phase 4.6 behavior exactly.
        """
        cfg = self._cfg
        any_triggered = False

        # ── Phase 5: Extract adaptive parameters (fallback to static) ─────
        if adapted_context is not None:
            threshold = adapted_context.adapted_threshold
            dampening = adapted_context.dampening_factor
            gate_open = adapted_context.stability_gate_open
            circuit_break = adapted_context.circuit_breaker_active
        else:
            threshold = cfg.coherence_threshold
            dampening = 1.0
            gate_open = True
            circuit_break = False

        # ── Phase 5: Circuit breaker — block ALL adjustments ──────────────
        if circuit_break:
            return StrategicAdjustment(
                coherence_at_evaluation=diagnostics.coherence_score,
                any_adjustment=False,
            )

        # ── 1. Drift cap reduction ────────────────────────────────────────
        drift_cap = None
        if diagnostics.coherence_score < threshold:
            drift_cap = cfg.reduced_drift_cap
            # Phase 5: dampening relaxes the cap toward normal
            if dampening < 1.0:
                normal_cap = 0.01  # default max_drift_per_update
                drift_cap = drift_cap + (normal_cap - drift_cap) * (1.0 - dampening)
            any_triggered = True

        # ── Phase 5: Gate non-safety adjustments when S² too low ──────────
        # Safety flags always pass through regardless of gate
        safety_flags_active = bool(
            set(diagnostics.risk_flags) & {"dependency_rising", "goals_failing"}
        )

        # ── 2. Dependency mitigation ──────────────────────────────────────
        dep_mult = 1.0
        if diagnostics.dependency_trend > 0 and "dependency_rising" in diagnostics.risk_flags:
            # Safety mechanism — never gated, never dampened
            dep_mult = cfg.dependency_boost_multiplier
            any_triggered = True

        # ── 3. Goal injection throttle ────────────────────────────────────
        allow_goals = True
        if diagnostics.goal_resolution_rate < cfg.goal_injection_min_resolution:
            # Safety mechanism — never gated
            allow_goals = False
            any_triggered = True

        # ── 4. Guidance mode hint ─────────────────────────────────────────
        guidance_hint = None
        if diagnostics.coherence_score < threshold and gate_open:
            guidance_hint = "structured"
            any_triggered = True

        # ── 5. Meta-goal injection ────────────────────────────────────────
        meta_goals = []

        if diagnostics.dependency_trend > cfg.dependency_meta_goal_threshold:
            # Safety — bypasses gate
            meta_goals.append(MetaGoal(
                description="Encourage independent real-world action",
                priority=0.8,
            ))
            any_triggered = True

        if diagnostics.goal_resolution_rate < cfg.goal_failure_meta_goal_threshold:
            # Safety — bypasses gate
            meta_goals.append(MetaGoal(
                description="Reduce cognitive load — prioritize fewer tasks",
                priority=0.75,
            ))
            any_triggered = True

        if "mode_unstable" in diagnostics.risk_flags and gate_open:
            # Non-safety — gated by S²
            meta_goals.append(MetaGoal(
                description="Stabilize interaction approach before adapting further",
                priority=0.6,
            ))
            any_triggered = True

        # ── Phase 5: If gate is closed and no safety flags, suppress ──────
        if not gate_open and not safety_flags_active:
            # Volatility too high — suppress non-safety adjustments
            drift_cap = None
            guidance_hint = None
            meta_goals = [g for g in meta_goals if g.priority >= 0.75]
            # Re-check if anything is still triggered
            any_triggered = bool(
                dep_mult > 1.0
                or not allow_goals
                or meta_goals
            )

        return StrategicAdjustment(
            suggested_drift_cap=drift_cap,
            dependency_mitigation_multiplier=dep_mult,
            allow_goal_injection=allow_goals,
            suggested_guidance_mode=guidance_hint,
            meta_goals=tuple(meta_goals),
            coherence_at_evaluation=diagnostics.coherence_score,
            any_adjustment=any_triggered,
        )