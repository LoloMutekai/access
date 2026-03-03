"""
A.C.C.E.S.S. — Relationship State (Phase 4)

Models the evolving companion-user relationship.

Design:
    - RelationshipState is FROZEN — every update returns a new instance
    - RelationshipEngine is PURE — no DB, no side effects, no external calls
    - All values clamped to [0.0, 1.0]
    - Dependency risk is a safety mechanism — prevents unhealthy reliance

Dimensions:
    trust_level         : built through consistent helpful interactions, eroded by errors
    respect_level       : built through appropriate challenge, eroded by sycophancy
    challenge_tolerance : how much push the user can handle, adapts per outcome
    dependency_risk     : rises with high frequency + emotional reliance patterns

Update triggers:
    - After each finalized turn (via reflection + trajectory + emotional state)
    - Time-based decay (between sessions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelationshipConfig:
    """All thresholds for relationship updates. No magic numbers in engine code."""

    # Per-turn increments (before clamping)
    trust_positive_step: float = 0.015
    trust_negative_step: float = 0.025         # asymmetric: easier to lose than build
    respect_positive_step: float = 0.012
    respect_negative_step: float = 0.008
    challenge_tolerance_step: float = 0.010

    # Dependency risk
    dependency_base_increment: float = 0.005   # passive rise per interaction
    dependency_negative_bonus: float = 0.020   # extra if user relies on emotional support
    dependency_positive_relief: float = 0.010  # decrease when user shows independence
    dependency_alarm_threshold: float = 0.7

    # Time-based decay
    trust_decay_per_day: float = 0.005
    respect_decay_per_day: float = 0.003
    dependency_decay_per_day: float = 0.015    # dependency dissipates faster (healthy)

    # Baseline gravity — values drift here during inactivity
    trust_baseline: float = 0.3
    respect_baseline: float = 0.3
    challenge_tolerance_baseline: float = 0.35

    # Safety clamp — no single update exceeds this
    max_single_step: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RelationshipState:
    """Immutable snapshot. Every update returns a new instance."""
    trust_level: float = 0.3
    respect_level: float = 0.3
    challenge_tolerance: float = 0.35
    dependency_risk: float = 0.0
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        for attr in ("trust_level", "respect_level", "challenge_tolerance", "dependency_risk"):
            object.__setattr__(self, attr, _clamp(getattr(self, attr)))

    def to_dict(self) -> dict:
        return {
            "trust_level": round(self.trust_level, 4),
            "respect_level": round(self.respect_level, 4),
            "challenge_tolerance": round(self.challenge_tolerance, 4),
            "dependency_risk": round(self.dependency_risk, 4),
            "interaction_count": self.interaction_count,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "last_interaction_at": self.last_interaction_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipState":
        return cls(
            trust_level=d.get("trust_level", 0.3),
            respect_level=d.get("respect_level", 0.3),
            challenge_tolerance=d.get("challenge_tolerance", 0.35),
            dependency_risk=d.get("dependency_risk", 0.0),
            interaction_count=d.get("interaction_count", 0),
            positive_interactions=d.get("positive_interactions", 0),
            negative_interactions=d.get("negative_interactions", 0),
            last_interaction_at=_parse_dt(d.get("last_interaction_at")),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    @property
    def dependency_alarm(self) -> bool:
        return self.dependency_risk >= 0.7

    @property
    def trust_ratio(self) -> float:
        total = self.positive_interactions + self.negative_interactions
        return self.positive_interactions / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"RelationshipState(trust={self.trust_level:.2f}, "
            f"respect={self.respect_level:.2f}, "
            f"challenge={self.challenge_tolerance:.2f}, "
            f"dependency={self.dependency_risk:.2f}, "
            f"interactions={self.interaction_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipEngine:
    """Pure computation engine. No side effects, no DB, fully deterministic."""

    def __init__(self, config: Optional[RelationshipConfig] = None):
        self._cfg = config or RelationshipConfig()

    def update_from_turn(
        self,
        current: RelationshipState,
        reflection_result,
        trajectory_state,
        emotional_state,
        dependency_mitigation_multiplier: float = 1.0,
    ) -> RelationshipState:
        """
        Update relationship state after a finalized turn.

        Args:
            dependency_mitigation_multiplier: Phase 4.6 scoped override.
                Values > 1.0 amplify the dependency_positive_relief (reduces dependency faster).
                Applied only for this call — no permanent state mutation.
                Default 1.0 = no override.
        """
        cfg = self._cfg

        # Extract signals (duck-typed)
        goal = _getattr(reflection_result, "goal_signal", None)
        traj = _getattr(trajectory_state, "dominant_trajectory", None)
        drift = _getattr(trajectory_state, "drift_score", 0.0)
        is_positive = _getattr(emotional_state, "is_positive", False)
        is_negative = _getattr(emotional_state, "is_negative", False)
        emotion = _getattr(emotional_state, "primary_emotion", "neutral")

        # ── Trust ─────────────────────────────────────────────────────────
        trust_delta = 0.0
        if traj in ("progressing", "escalating"):
            trust_delta += cfg.trust_positive_step
        if goal in ("push_forward", "execute"):
            trust_delta += cfg.trust_positive_step * 0.5
        if traj == "declining":
            trust_delta -= cfg.trust_negative_step
        if drift > 0.5:
            trust_delta -= cfg.trust_negative_step * 0.3

        # ── Respect ───────────────────────────────────────────────────────
        respect_delta = 0.0
        if goal in ("execute", "push_forward") and is_positive:
            respect_delta += cfg.respect_positive_step
        if goal in ("stabilize", "recover") and not is_negative:
            respect_delta -= cfg.respect_negative_step

        # ── Challenge tolerance ───────────────────────────────────────────
        challenge_delta = 0.0
        if is_positive and goal in ("push_forward", "execute"):
            challenge_delta += cfg.challenge_tolerance_step
        if emotion in ("fatigue", "frustration") and traj == "declining":
            challenge_delta -= cfg.challenge_tolerance_step * 1.5

        # ── Dependency risk ───────────────────────────────────────────────
        # Phase 4.6: scoped mitigation multiplier amplifies relief
        effective_relief = cfg.dependency_positive_relief * dependency_mitigation_multiplier
        dep_delta = cfg.dependency_base_increment
        if is_negative and goal in ("stabilize", "recover"):
            dep_delta += cfg.dependency_negative_bonus
        if is_positive and goal in ("push_forward", "execute"):
            dep_delta -= effective_relief

        # ── Classify interaction ──────────────────────────────────────────
        positive = is_positive or traj in ("progressing", "escalating")
        negative = is_negative and traj == "declining"

        max_step = cfg.max_single_step
        return RelationshipState(
            trust_level=current.trust_level + _bounded(trust_delta, max_step),
            respect_level=current.respect_level + _bounded(respect_delta, max_step),
            challenge_tolerance=current.challenge_tolerance + _bounded(challenge_delta, max_step),
            dependency_risk=current.dependency_risk + _bounded(dep_delta, max_step),
            interaction_count=current.interaction_count + 1,
            positive_interactions=current.positive_interactions + (1 if positive else 0),
            negative_interactions=current.negative_interactions + (1 if negative else 0),
            last_interaction_at=datetime.now(UTC),
        )

    def decay_over_time(
        self,
        current: RelationshipState,
        hours_elapsed: float,
    ) -> RelationshipState:
        """Apply time-based drift when user is absent."""
        if hours_elapsed <= 0:
            return current
        cfg = self._cfg
        days = hours_elapsed / 24.0

        return RelationshipState(
            trust_level=_drift_toward(current.trust_level, cfg.trust_baseline,
                                      cfg.trust_decay_per_day * days),
            respect_level=_drift_toward(current.respect_level, cfg.respect_baseline,
                                        cfg.respect_decay_per_day * days),
            challenge_tolerance=current.challenge_tolerance,
            dependency_risk=max(0.0, current.dependency_risk - cfg.dependency_decay_per_day * days),
            interaction_count=current.interaction_count,
            positive_interactions=current.positive_interactions,
            negative_interactions=current.negative_interactions,
            last_interaction_at=current.last_interaction_at,
        )

    def detect_dependency_risk(self, state: RelationshipState) -> dict:
        """Analyze dependency signals. Returns diagnostic dict."""
        risk_level = "safe"
        if state.dependency_risk >= 0.7:
            risk_level = "critical"
        elif state.dependency_risk >= 0.4:
            risk_level = "elevated"
        elif state.dependency_risk >= 0.2:
            risk_level = "mild"

        recommendations = {
            "safe": "No action needed.",
            "mild": "Monitor interaction patterns.",
            "elevated": "Consider encouraging independent action.",
            "critical": "Actively redirect user toward real-world support systems.",
        }
        return {
            "risk_level": risk_level,
            "dependency_risk": round(state.dependency_risk, 4),
            "interaction_count": state.interaction_count,
            "trust_ratio": round(state.trust_ratio, 4),
            "alarm": state.dependency_alarm,
            "recommendation": recommendations.get(risk_level, "Unknown."),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _bounded(delta: float, max_step: float) -> float:
    return max(-max_step, min(max_step, delta))

def _drift_toward(current: float, target: float, rate: float) -> float:
    if current > target:
        return max(target, current - rate)
    elif current < target:
        return min(target, current + rate)
    return current

def _getattr(obj, attr: str, default):
    """Safe getattr that handles None objects."""
    if obj is None:
        return default
    return getattr(obj, attr, default)

def _parse_dt(val) -> datetime:
    if val is None:
        return datetime.now(UTC)
    if isinstance(val, datetime):
        return val
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return datetime.now(UTC)