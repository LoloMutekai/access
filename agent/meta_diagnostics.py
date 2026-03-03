"""
A.C.C.E.S.S. — Meta-Diagnostics Engine (Phase 4.5)

Structural self-monitoring layer.
Observes Phase 4 state history and computes coherence metrics.

This is NOT emotional reflection.
This is internal systems integrity monitoring.

Design:
    - MetaDiagnostics is FROZEN — pure diagnostic snapshot
    - MetaDiagnosticsEngine is PURE — no side effects, no mutations
    - Operates on history buffers (lists of past snapshots)
    - All computations deterministic
    - Never modifies Phase 4 state

Metrics:
    personality_volatility  : mean trait distance between consecutive snapshots
    relationship_volatility : mean change in trust/respect between consecutive snapshots
    goal_resolution_rate    : completed / (completed + expired) goals
    dependency_trend        : slope of dependency_risk over last K snapshots
    mode_instability        : fraction of mode changes in recent self-model history
    coherence_score         : weighted composite ∈ [0.0, 1.0]
    risk_flags              : triggered when thresholds are exceeded
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from .relationship_state import RelationshipState, _clamp
from .personality_state import PersonalityTraits
from .self_model import SelfModel
from .goal_queue import GoalQueue

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaDiagnosticsConfig:
    """Thresholds for meta-cognitive diagnostics."""

    # Volatility thresholds (above these → risk flag)
    personality_volatility_threshold: float = 0.15
    relationship_volatility_threshold: float = 0.10

    # Dependency trend threshold (positive slope above this → flag)
    dependency_trend_threshold: float = 0.01

    # Mode instability threshold (fraction of turns with mode change)
    mode_instability_threshold: float = 0.30

    # Goal resolution minimum (below this → flag)
    goal_resolution_min: float = 0.25

    # Coherence scoring weights (must sum to ~1.0)
    w_personality_stability: float = 0.20
    w_relationship_stability: float = 0.20
    w_mode_stability: float = 0.20
    w_dependency_health: float = 0.25
    w_goal_health: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetaDiagnostics:
    """Immutable diagnostic snapshot. Pure observation."""
    personality_volatility: float = 0.0
    relationship_volatility: float = 0.0
    goal_resolution_rate: float = 1.0
    dependency_trend: float = 0.0
    mode_instability: float = 0.0
    coherence_score: float = 1.0
    risk_flags: tuple[str, ...] = field(default_factory=tuple)
    data_points: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        object.__setattr__(self, "coherence_score", _clamp(self.coherence_score))

    def to_dict(self) -> dict:
        return {
            "personality_volatility": round(self.personality_volatility, 4),
            "relationship_volatility": round(self.relationship_volatility, 4),
            "goal_resolution_rate": round(self.goal_resolution_rate, 4),
            "dependency_trend": round(self.dependency_trend, 4),
            "mode_instability": round(self.mode_instability, 4),
            "coherence_score": round(self.coherence_score, 4),
            "risk_flags": list(self.risk_flags),
            "data_points": self.data_points,
            "generated_at": self.generated_at.isoformat(),
        }

    @property
    def is_healthy(self) -> bool:
        return len(self.risk_flags) == 0

    def __repr__(self) -> str:
        flags = f", flags={list(self.risk_flags)}" if self.risk_flags else ""
        return (
            f"MetaDiagnostics(coherence={self.coherence_score:.2f}, "
            f"p_vol={self.personality_volatility:.3f}, "
            f"r_vol={self.relationship_volatility:.3f}, "
            f"dep_trend={self.dependency_trend:+.4f}, "
            f"pts={self.data_points}{flags})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY SNAPSHOT (what gets buffered each turn)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IdentitySnapshot:
    """Lightweight snapshot of Phase 4 state at one point in time."""
    personality: PersonalityTraits
    relationship: RelationshipState
    dominant_mode: str
    dependency_risk: float
    active_goals: int
    completed_goals: int
    expired_goals: int
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "personality": self.personality.to_dict(),
            "relationship": self.relationship.to_dict(),
            "dominant_mode": self.dominant_mode,
            "dependency_risk": round(self.dependency_risk, 4),
            "active_goals": self.active_goals,
            "completed_goals": self.completed_goals,
            "expired_goals": self.expired_goals,
            "captured_at": self.captured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IdentitySnapshot":
        """Deserialize from JSON dict. Returns defaults on malformed input."""
        from .relationship_state import _parse_dt
        return cls(
            personality=PersonalityTraits.from_dict(d.get("personality", {})),
            relationship=RelationshipState.from_dict(d.get("relationship", {})),
            dominant_mode=d.get("dominant_mode", "collaborating"),
            dependency_risk=d.get("dependency_risk", 0.0),
            active_goals=d.get("active_goals", 0),
            completed_goals=d.get("completed_goals", 0),
            expired_goals=d.get("expired_goals", 0),
            captured_at=_parse_dt(d.get("captured_at")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MetaDiagnosticsEngine:
    """
    Pure computation engine. Analyzes history buffers, returns MetaDiagnostics.
    Never mutates state. Never raises.
    """

    def __init__(self, config: Optional[MetaDiagnosticsConfig] = None):
        self._cfg = config or MetaDiagnosticsConfig()

    def analyze(self, history: list[IdentitySnapshot]) -> MetaDiagnostics:
        """
        Compute diagnostics from a history buffer of identity snapshots.

        Requires >= 2 snapshots for meaningful metrics.
        Returns neutral diagnostics if insufficient data.
        """
        if len(history) < 2:
            return MetaDiagnostics(data_points=len(history))

        p_vol = self._personality_volatility(history)
        r_vol = self._relationship_volatility(history)
        dep_trend = self._dependency_trend(history)
        mode_inst = self._mode_instability(history)
        goal_rate = self._goal_resolution_rate(history)

        coherence = self._compute_coherence(
            p_vol, r_vol, dep_trend, mode_inst, goal_rate,
        )

        flags = self._detect_risk_flags(
            p_vol, r_vol, dep_trend, mode_inst, goal_rate,
        )

        return MetaDiagnostics(
            personality_volatility=p_vol,
            relationship_volatility=r_vol,
            goal_resolution_rate=goal_rate,
            dependency_trend=dep_trend,
            mode_instability=mode_inst,
            coherence_score=coherence,
            risk_flags=tuple(flags),
            data_points=len(history),
        )

    # ── Metric computations ───────────────────────────────────────────────

    def _personality_volatility(self, history: list[IdentitySnapshot]) -> float:
        """Mean Euclidean trait distance between consecutive snapshots."""
        if len(history) < 2:
            return 0.0
        distances = []
        for i in range(1, len(history)):
            prev = history[i - 1].personality
            curr = history[i].personality
            d = curr.distance_from(prev)
            distances.append(d)
        return sum(distances) / len(distances) if distances else 0.0

    def _relationship_volatility(self, history: list[IdentitySnapshot]) -> float:
        """Mean absolute change in trust + respect between consecutive snapshots."""
        if len(history) < 2:
            return 0.0
        deltas = []
        for i in range(1, len(history)):
            prev = history[i - 1].relationship
            curr = history[i].relationship
            trust_d = abs(curr.trust_level - prev.trust_level)
            respect_d = abs(curr.respect_level - prev.respect_level)
            deltas.append(trust_d + respect_d)
        return sum(deltas) / len(deltas) if deltas else 0.0

    def _dependency_trend(self, history: list[IdentitySnapshot]) -> float:
        """Linear slope of dependency_risk over snapshots. Positive = rising."""
        if len(history) < 2:
            return 0.0
        values = [s.dependency_risk for s in history]
        return _linear_slope(values)

    def _mode_instability(self, history: list[IdentitySnapshot]) -> float:
        """Fraction of turns where dominant_mode changed from previous turn."""
        if len(history) < 2:
            return 0.0
        changes = sum(
            1 for i in range(1, len(history))
            if history[i].dominant_mode != history[i - 1].dominant_mode
        )
        return changes / (len(history) - 1)

    def _goal_resolution_rate(self, history: list[IdentitySnapshot]) -> float:
        """Completed / (completed + expired) from latest snapshot. 1.0 if no terminal goals."""
        if not history:
            return 1.0
        latest = history[-1]
        total_terminal = latest.completed_goals + latest.expired_goals
        if total_terminal == 0:
            return 1.0
        return latest.completed_goals / total_terminal

    # ── Coherence scoring ─────────────────────────────────────────────────

    def _compute_coherence(
        self,
        p_vol: float,
        r_vol: float,
        dep_trend: float,
        mode_inst: float,
        goal_rate: float,
    ) -> float:
        """
        Weighted coherence score ∈ [0.0, 1.0].

        Higher = more stable/coherent system.
        Monotonically decreases with instability.
        Never returns NaN.
        """
        cfg = self._cfg

        # Inverse volatilities → stability scores [0, 1]
        p_stability = 1.0 / (1.0 + 10.0 * p_vol)
        r_stability = 1.0 / (1.0 + 10.0 * r_vol)
        mode_stability = 1.0 - _clamp(mode_inst)

        # Dependency health: 1.0 when dep_trend <= 0, decays as trend rises
        dep_health = 1.0 / (1.0 + max(0.0, dep_trend) * 50.0)

        # Goal health = resolution rate directly
        goal_health = _clamp(goal_rate)

        # Weighted sum
        score = (
            cfg.w_personality_stability * p_stability
            + cfg.w_relationship_stability * r_stability
            + cfg.w_mode_stability * mode_stability
            + cfg.w_dependency_health * dep_health
            + cfg.w_goal_health * goal_health
        )

        if math.isnan(score) or math.isinf(score):
            return 0.0
        return _clamp(score)

    # ── Risk flag detection ───────────────────────────────────────────────

    def _detect_risk_flags(
        self,
        p_vol: float,
        r_vol: float,
        dep_trend: float,
        mode_inst: float,
        goal_rate: float,
    ) -> list[str]:
        """Return list of triggered risk flag identifiers."""
        cfg = self._cfg
        flags = []

        if p_vol > cfg.personality_volatility_threshold:
            flags.append("personality_volatile")
        if r_vol > cfg.relationship_volatility_threshold:
            flags.append("relationship_volatile")
        if dep_trend > cfg.dependency_trend_threshold:
            flags.append("dependency_rising")
        if mode_inst > cfg.mode_instability_threshold:
            flags.append("mode_unstable")
        if goal_rate < cfg.goal_resolution_min:
            flags.append("goals_failing")

        return flags


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _linear_slope(values: list[float]) -> float:
    """OLS slope for a sequence of values indexed 0..N-1. Returns 0.0 if < 2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if abs(denominator) < 1e-10:
        return 0.0
    return numerator / denominator