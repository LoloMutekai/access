"""
A.C.C.E.S.S. — Self Model (Phase 4)

The companion's internal model of itself and the user's patterns.

Design:
    - SelfModel is FROZEN — updates return new instances
    - SelfModelEngine heuristically infers patterns from turn signals
    - Requires sustained signal before shifting (hysteresis prevents noise)
    - All inferences tagged with vote counts
    - Serializable to JSON

Components:
    dominant_interaction_mode  : coaching / collaborating / supporting / challenging
    preferred_guidance_style   : socratic / directive / exploratory / structured
    long_term_goal_pattern     : achievement / exploration / stability / recovery
    recurring_user_patterns    : behavioral tags observed across sessions

Mode detection uses weighted voting with hysteresis:
    Each turn casts a vote. The dominant mode must exceed a threshold
    AND lead the runner-up by ≥2 votes before a shift is allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

from .relationship_state import _parse_dt, _getattr

logger = logging.getLogger(__name__)

# ── Valid modes ──────────────────────────────────────────────────────────────
INTERACTION_MODES = frozenset({"coaching", "collaborating", "supporting", "challenging"})
GUIDANCE_STYLES = frozenset({"socratic", "directive", "exploratory", "structured"})
GOAL_PATTERNS = frozenset({"achievement", "exploration", "stability", "recovery"})


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelfModelConfig:
    mode_shift_threshold: int = 5       # min votes before shift allowed
    max_pattern_tags: int = 50          # max tracked patterns
    recurring_tag_min_count: int = 3    # min appearances to be "recurring"
    vote_window: int = 50              # votes trimmed beyond 2× this


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelfModel:
    """Immutable snapshot of the companion's self-understanding."""
    dominant_interaction_mode: str = "collaborating"
    preferred_guidance_style: str = "socratic"
    long_term_goal_pattern: Optional[str] = None
    recurring_user_patterns: tuple[str, ...] = field(default_factory=tuple)

    # Internal vote accumulators (carried forward between updates)
    _mode_votes: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    _style_votes: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    _goal_votes: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    _pattern_counts: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    total_observations: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "dominant_interaction_mode": self.dominant_interaction_mode,
            "preferred_guidance_style": self.preferred_guidance_style,
            "long_term_goal_pattern": self.long_term_goal_pattern,
            "recurring_user_patterns": list(self.recurring_user_patterns),
            "mode_votes": dict(self._mode_votes),
            "style_votes": dict(self._style_votes),
            "goal_votes": dict(self._goal_votes),
            "pattern_counts": dict(self._pattern_counts),
            "total_observations": self.total_observations,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        return cls(
            dominant_interaction_mode=d.get("dominant_interaction_mode", "collaborating"),
            preferred_guidance_style=d.get("preferred_guidance_style", "socratic"),
            long_term_goal_pattern=d.get("long_term_goal_pattern"),
            recurring_user_patterns=tuple(d.get("recurring_user_patterns", [])),
            _mode_votes=tuple(sorted(d.get("mode_votes", {}).items())),
            _style_votes=tuple(sorted(d.get("style_votes", {}).items())),
            _goal_votes=tuple(sorted(d.get("goal_votes", {}).items())),
            _pattern_counts=tuple(sorted(d.get("pattern_counts", {}).items())),
            total_observations=d.get("total_observations", 0),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    def __repr__(self) -> str:
        return (
            f"SelfModel(mode={self.dominant_interaction_mode!r}, "
            f"style={self.preferred_guidance_style!r}, "
            f"goal={self.long_term_goal_pattern!r}, "
            f"patterns={len(self.recurring_user_patterns)}, "
            f"obs={self.total_observations})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SelfModelEngine:
    """Observes turn signals and gradually builds the companion's self-model."""

    def __init__(self, config: Optional[SelfModelConfig] = None):
        self._cfg = config or SelfModelConfig()

    def observe_turn(
        self,
        current: SelfModel,
        reflection_result,
        trajectory_state,
        emotional_state,
        modulation,
    ) -> SelfModel:
        """Process one turn's signals and return an updated SelfModel."""
        cfg = self._cfg

        mode_vote = self._infer_mode(reflection_result, trajectory_state, emotional_state)
        style_vote = self._infer_style(modulation, reflection_result)
        goal_vote = self._infer_goal(reflection_result, trajectory_state)
        pattern_tags = self._extract_patterns(emotional_state, reflection_result)

        mode_votes = _add_vote(dict(current._mode_votes), mode_vote, cfg.vote_window)
        style_votes = _add_vote(dict(current._style_votes), style_vote, cfg.vote_window)
        goal_votes = _add_vote(dict(current._goal_votes), goal_vote, cfg.vote_window)
        pattern_counts = _merge_tags(dict(current._pattern_counts), pattern_tags, cfg.max_pattern_tags)

        new_mode = _resolve_dominant(mode_votes, current.dominant_interaction_mode, cfg.mode_shift_threshold)
        new_style = _resolve_dominant(style_votes, current.preferred_guidance_style, cfg.mode_shift_threshold)
        new_goal = _resolve_dominant(goal_votes, current.long_term_goal_pattern, cfg.mode_shift_threshold)

        recurring = tuple(
            tag for tag, count in sorted(pattern_counts.items(), key=lambda x: -x[1])
            if count >= cfg.recurring_tag_min_count
        )

        return SelfModel(
            dominant_interaction_mode=new_mode or current.dominant_interaction_mode,
            preferred_guidance_style=new_style or current.preferred_guidance_style,
            long_term_goal_pattern=new_goal,
            recurring_user_patterns=recurring,
            _mode_votes=tuple(sorted(mode_votes.items())),
            _style_votes=tuple(sorted(style_votes.items())),
            _goal_votes=tuple(sorted(goal_votes.items())),
            _pattern_counts=tuple(sorted(pattern_counts.items())),
            total_observations=current.total_observations + 1,
        )

    # ── Signal inference ──────────────────────────────────────────────────

    def _infer_mode(self, reflection, trajectory, emotional_state) -> Optional[str]:
        goal = _getattr(reflection, "goal_signal", None)
        traj = _getattr(trajectory, "dominant_trajectory", None)
        is_neg = _getattr(emotional_state, "is_negative", False)

        if goal == "push_forward" and traj == "progressing":
            return "coaching"
        if goal == "execute":
            return "collaborating"
        if goal in ("stabilize", "recover") and is_neg:
            return "supporting"
        if goal == "push_forward" and traj == "escalating":
            return "challenging"
        return None

    def _infer_style(self, modulation, reflection) -> Optional[str]:
        structure = _getattr(modulation, "structure_bias", None)
        tone = _getattr(modulation, "tone", None)
        importance = _getattr(reflection, "importance_score", 0.5)

        if structure == "structured" and importance > 0.6:
            return "structured"
        if tone == "challenging" and importance > 0.5:
            return "directive"
        if tone in ("calm", "grounding"):
            return "socratic"
        if tone == "energizing":
            return "exploratory"
        return None

    def _infer_goal(self, reflection, trajectory) -> Optional[str]:
        goal = _getattr(reflection, "goal_signal", None)
        if goal in ("push_forward", "execute"):
            return "achievement"
        if goal == "explore":
            return "exploration"
        if goal == "stabilize":
            return "stability"
        if goal == "recover":
            return "recovery"
        return None

    def _extract_patterns(self, emotional_state, reflection) -> list[str]:
        tags = []
        emotion = _getattr(emotional_state, "primary_emotion", None)
        if emotion:
            tags.append(f"emotion:{emotion}")
        goal = _getattr(reflection, "goal_signal", None)
        if goal:
            tags.append(f"goal:{goal}")
        return tags


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _add_vote(votes: dict, candidate: Optional[str], window: int) -> dict:
    if candidate is None:
        return votes
    votes[candidate] = votes.get(candidate, 0) + 1
    total = sum(votes.values())
    if total > window * 2:
        factor = window / total
        votes = {k: max(1, int(v * factor)) for k, v in votes.items()}
    return votes

def _merge_tags(counts: dict, new_tags: list[str], max_tags: int) -> dict:
    for tag in new_tags:
        counts[tag] = counts.get(tag, 0) + 1
    if len(counts) > max_tags:
        counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:max_tags])
    return counts

def _resolve_dominant(votes: dict, current: Optional[str], threshold: int) -> Optional[str]:
    """Top-voted candidate, but only if it exceeds threshold AND leads by ≥2 (hysteresis)."""
    if not votes:
        return current
    ranked = sorted(votes.items(), key=lambda x: -x[1])
    top_name, top_count = ranked[0]
    if top_count < threshold:
        return current
    if len(ranked) >= 2 and top_count - ranked[1][1] < 2:
        return current
    return top_name