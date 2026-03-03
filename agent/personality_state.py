"""
A.C.C.E.S.S. — Personality State (Phase 4)

Models the companion's evolving personality traits.

Design:
    - PersonalityTraits is FROZEN — drift returns a new instance
    - TraitDriftEngine computes adjustments from trajectory + relationship signals
    - Maximum drift per update: 0.01 (glacial — personality is inherently stable)
    - Traits drift toward baseline during inactivity (homeostasis)
    - Fully deterministic, no external calls

Trait dimensions:
    assertiveness      : [0.0 passive → 1.0 confrontational]
    warmth             : [0.0 clinical → 1.0 nurturing]
    analytical_bias    : [0.0 intuitive → 1.0 rigidly structured]
    motivational_drive : [0.0 passive → 1.0 relentless]
    emotional_stability: [0.0 reactive → 1.0 rock-solid]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from .relationship_state import _clamp, _bounded, _drift_toward, _getattr, _parse_dt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonalityConfig:
    """Drift thresholds. Personality changes are deliberately slow."""
    max_drift_per_update: float = 0.01
    drift_step: float = 0.005

    # Baselines — traits regress here during inactivity
    baseline_assertiveness: float = 0.45
    baseline_warmth: float = 0.55
    baseline_analytical_bias: float = 0.50
    baseline_motivational_drive: float = 0.50
    baseline_emotional_stability: float = 0.55

    # Homeostasis: how fast traits drift back per day of inactivity
    homeostasis_rate_per_day: float = 0.003

    # Maturation: emotional_stability slowly increases over time
    maturation_per_100_interactions: float = 0.005


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

TRAIT_NAMES = ("assertiveness", "warmth", "analytical_bias",
               "motivational_drive", "emotional_stability")

@dataclass(frozen=True)
class PersonalityTraits:
    """Immutable personality snapshot. All values ∈ [0.0, 1.0]."""
    assertiveness: float = 0.45
    warmth: float = 0.55
    analytical_bias: float = 0.50
    motivational_drive: float = 0.50
    emotional_stability: float = 0.55
    total_drift_events: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        for attr in TRAIT_NAMES:
            object.__setattr__(self, attr, _clamp(getattr(self, attr)))

    def to_dict(self) -> dict:
        return {
            "assertiveness": round(self.assertiveness, 4),
            "warmth": round(self.warmth, 4),
            "analytical_bias": round(self.analytical_bias, 4),
            "motivational_drive": round(self.motivational_drive, 4),
            "emotional_stability": round(self.emotional_stability, 4),
            "total_drift_events": self.total_drift_events,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityTraits":
        return cls(
            assertiveness=d.get("assertiveness", 0.45),
            warmth=d.get("warmth", 0.55),
            analytical_bias=d.get("analytical_bias", 0.50),
            motivational_drive=d.get("motivational_drive", 0.50),
            emotional_stability=d.get("emotional_stability", 0.55),
            total_drift_events=d.get("total_drift_events", 0),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    @property
    def trait_vector(self) -> tuple[float, ...]:
        return tuple(getattr(self, n) for n in TRAIT_NAMES)

    def distance_from(self, other: "PersonalityTraits") -> float:
        return sum((a - b) ** 2 for a, b in zip(self.trait_vector, other.trait_vector)) ** 0.5

    def __repr__(self) -> str:
        return (
            f"PersonalityTraits("
            f"assert={self.assertiveness:.2f}, "
            f"warmth={self.warmth:.2f}, "
            f"analytic={self.analytical_bias:.2f}, "
            f"drive={self.motivational_drive:.2f}, "
            f"stability={self.emotional_stability:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TraitDriftEngine:
    """Computes personality drift from interaction signals. Max ±0.01 per trait per update."""

    def __init__(self, config: Optional[PersonalityConfig] = None):
        self._cfg = config or PersonalityConfig()

    def compute_drift(
        self,
        current: PersonalityTraits,
        reflection_result,
        trajectory_state,
        emotional_state,
        modulation,
        drift_cap_override: Optional[float] = None,
    ) -> PersonalityTraits:
        """
        Apply one step of personality drift.

        Args:
            drift_cap_override: Phase 4.6 scoped override.
                If not None, replaces max_drift_per_update for this call only.
                No permanent mutation of engine config.
        """
        cfg = self._cfg
        step = cfg.drift_step
        cap = drift_cap_override if drift_cap_override is not None else cfg.max_drift_per_update

        goal = _getattr(reflection_result, "goal_signal", None)
        traj = _getattr(trajectory_state, "dominant_trajectory", None)
        is_positive = _getattr(emotional_state, "is_positive", False)
        is_negative = _getattr(emotional_state, "is_negative", False)
        emotion = _getattr(emotional_state, "primary_emotion", "neutral")
        validation = _getattr(modulation, "emotional_validation", False)
        structure = _getattr(modulation, "structure_bias", "conversational")

        # Assertiveness: ↑ push/execute, ↓ stabilize/recover
        a_delta = 0.0
        if goal in ("push_forward", "execute"):
            a_delta += step
        if goal in ("stabilize", "recover"):
            a_delta -= step * 0.5

        # Warmth: ↑ emotional validation, ↓ pure analytical
        w_delta = 0.0
        if validation:
            w_delta += step
        if not validation and structure == "structured":
            w_delta -= step * 0.3

        # Analytical bias: ↑ structured success, ↓ conversational preferred
        ab_delta = 0.0
        if structure == "structured":
            ab_delta += step * 0.5
        if structure == "conversational":
            ab_delta -= step * 0.3

        # Motivational drive: ↑ positive momentum, ↓ burnout signals
        md_delta = 0.0
        if is_positive and goal in ("push_forward", "execute"):
            md_delta += step
        if emotion in ("fatigue", "frustration") and traj == "declining":
            md_delta -= step * 1.2

        # Emotional stability: slow maturation, slowed by volatility
        es_delta = cfg.maturation_per_100_interactions / 100.0
        if traj == "declining" and is_negative:
            es_delta -= step * 0.3

        return PersonalityTraits(
            assertiveness=current.assertiveness + _bounded(a_delta, cap),
            warmth=current.warmth + _bounded(w_delta, cap),
            analytical_bias=current.analytical_bias + _bounded(ab_delta, cap),
            motivational_drive=current.motivational_drive + _bounded(md_delta, cap),
            emotional_stability=current.emotional_stability + _bounded(es_delta, cap),
            total_drift_events=current.total_drift_events + 1,
        )

    def apply_homeostasis(
        self,
        current: PersonalityTraits,
        hours_elapsed: float,
    ) -> PersonalityTraits:
        """Drift traits toward baselines during inactivity."""
        if hours_elapsed <= 0:
            return current
        cfg = self._cfg
        rate = cfg.homeostasis_rate_per_day * (hours_elapsed / 24.0)

        return PersonalityTraits(
            assertiveness=_drift_toward(current.assertiveness, cfg.baseline_assertiveness, rate),
            warmth=_drift_toward(current.warmth, cfg.baseline_warmth, rate),
            analytical_bias=_drift_toward(current.analytical_bias, cfg.baseline_analytical_bias, rate),
            motivational_drive=_drift_toward(current.motivational_drive, cfg.baseline_motivational_drive, rate),
            emotional_stability=_drift_toward(current.emotional_stability, cfg.baseline_emotional_stability, rate),
            total_drift_events=current.total_drift_events,
        )