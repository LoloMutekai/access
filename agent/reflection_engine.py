"""
A.C.C.E.S.S. — Self-Reflection Engine (Phase 3)

After each finalized turn, the agent performs internal reflection on the interaction.

Design principles:
    - PURE — no external API calls, no database writes, no side effects
    - DETERMINISTIC — same inputs always produce same outputs (testable)
    - HEURISTIC-BASED — uses simple scoring rules, not another LLM call

ReflectionResult:
    Frozen dataclass capturing what the turn "meant" from a cognitive perspective.
    Used by AgentCore to:
        (a) Adapt the importance score for memory write-back
        (b) Feed the TrajectoryTracker for goal-drift detection
        (c) Surface metadata in AgentResponse for external inspection

Heuristic Scoring Algorithm:
    Base importance: 0.4

    +0.20  if emotional intensity > high_intensity_threshold  (emotionally significant turn)
    +0.15  if any tools were used                              (action was taken)
    +0.15  if |motivational_bias| > strong_bias_threshold     (strong push or brake)
    +0.10  if len(assistant_output) > long_output_threshold   (substantive response)
    +0.10  if emotional state is negative (frustration, doubt, fatigue)  (stress flagging)
    +0.05  if emotional validation was active in modulation    (empathic moment)
    clamp → [0.0, 1.0]

Goal Signal Detection:
    Extracted from the emotional state + modulation combination.
    Used by TrajectoryTracker to detect goal drift over time.

    "push_forward"   → high motivational bias + positive emotion
    "stabilize"      → calm/grounding tone + negative emotion
    "explore"        → doubt/questioning state
    "recover"        → burnout/fatigue detected
    "execute"        → drive + high arousal
    None             → neutral, no detectable signal

Trajectory Signal:
    A broader hint about the session arc — complements goal_signal.
    Detected from trend + dominant pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionConfig:
    """Thresholds for reflection heuristics. No magic numbers in engine code."""

    # Base importance before any bonuses
    base_importance: float = 0.4

    # Emotional intensity above which the turn is flagged as significant
    high_intensity_threshold: float = 0.65

    # |motivational_bias| above which the turn is "strongly directed"
    strong_bias_threshold: float = 0.35

    # assistant_output length above which the turn is considered substantive
    long_output_threshold: int = 400

    # Importance bonuses (sum clamped to [0.0, 1.0])
    bonus_high_intensity: float = 0.20
    bonus_tool_used: float = 0.15
    bonus_strong_bias: float = 0.15
    bonus_long_output: float = 0.10
    bonus_negative_emotion: float = 0.10
    bonus_validation: float = 0.05

    # Importance floor/ceiling
    min_importance: float = 0.0
    max_importance: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReflectionResult:
    """
    Immutable output of a reflection pass on a single turn.

    summary:
        A short machine-readable description of what happened this turn.
        Used as the memory summary instead of raw assistant_output when
        use_reflection_summary_for_memory=True.

    importance_score:
        Heuristic estimate of how important this turn is for long-term memory.
        Range [0.0, 1.0]. Used by AgentCore for adaptive memory write-back.

    emotional_tags:
        List of emotion labels relevant to this turn (from emotional_state).
        Stored alongside the memory for future alignment-aware retrieval.

    goal_signal:
        Optional categorical signal about the user's current goal orientation.
        One of: "push_forward", "stabilize", "explore", "recover", "execute", None.

    trajectory_signal:
        Optional broader arc signal for the session.
        One of: "progressing", "declining", "stable", "escalating", None.

    reflected_at:
        UTC timestamp of reflection computation.
    """
    summary: str
    importance_score: float
    emotional_tags: list[str]
    goal_signal: Optional[str]
    trajectory_signal: Optional[str]
    reflected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_log_dict(self) -> dict:
        return {
            "summary":           self.summary,
            "importance_score":  round(self.importance_score, 3),
            "emotional_tags":    self.emotional_tags,
            "goal_signal":       self.goal_signal,
            "trajectory_signal": self.trajectory_signal,
            "reflected_at":      self.reflected_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"ReflectionResult("
            f"importance={self.importance_score:.2f}, "
            f"goal={self.goal_signal!r}, "
            f"trajectory={self.trajectory_signal!r}, "
            f"summary='{self.summary[:60]}')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class ReflectionEngineProtocol(Protocol):
    """
    Structural contract for any reflection engine implementation.
    Allows easy substitution in tests (FakeReflectionEngine).
    """

    def reflect(
        self,
        user_input: str,
        assistant_output: str,
        emotional_state,
        modulation,
        tool_results: tuple,
        session_id: Optional[str] = None,
    ) -> ReflectionResult:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionEngine:
    """
    Pure heuristic reflection engine.

    No API calls. No database access. No side effects.
    Fully deterministic — same turn → same ReflectionResult.

    Usage:
        engine = ReflectionEngine()
        result = engine.reflect(
            user_input="I can't figure this out.",
            assistant_output="Let's break this down step by step...",
            emotional_state=state,
            modulation=mod,
            tool_results=(),
            session_id="session_1",
        )
        print(result.importance_score)  # e.g. 0.65
        print(result.goal_signal)       # e.g. "stabilize"
    """

    def __init__(self, config: Optional[ReflectionConfig] = None):
        self._config = config or ReflectionConfig()

    def reflect(
        self,
        user_input: str,
        assistant_output: str,
        emotional_state,
        modulation,
        tool_results: tuple,
        session_id: Optional[str] = None,
    ) -> ReflectionResult:
        """
        Reflect on a completed turn.

        Args:
            user_input:       Original user message.
            assistant_output: Final LLM response (post tool-use, post strip).
            emotional_state:  EmotionalState (duck-typed). May be None on failure.
            modulation:       ResponseModulation (duck-typed). May be None on failure.
            tool_results:     Tuple of ToolResult objects from this turn.
            session_id:       For contextual logging.

        Returns:
            ReflectionResult — always succeeds (uses safe defaults on error).
        """
        try:
            importance = self._compute_importance(
                emotional_state, modulation, tool_results, assistant_output
            )
            summary = self._build_summary(
                user_input, emotional_state, modulation, tool_results, importance
            )
            emotional_tags = self._extract_tags(emotional_state)
            goal_signal = self._detect_goal_signal(emotional_state, modulation)
            trajectory_signal = self._detect_trajectory_signal(emotional_state, modulation)

            result = ReflectionResult(
                summary=summary,
                importance_score=importance,
                emotional_tags=emotional_tags,
                goal_signal=goal_signal,
                trajectory_signal=trajectory_signal,
            )

            logger.debug(
                f"Reflection — session={session_id}, "
                f"importance={importance:.2f}, "
                f"goal={goal_signal}, "
                f"traj={trajectory_signal}"
            )
            return result

        except Exception as exc:
            # Reflection failure must NEVER crash AgentCore
            logger.error(f"ReflectionEngine error (returning defaults): {exc}", exc_info=True)
            return self._fallback_result(user_input, emotional_state)

    # ─────────────────────────────────────────────────────────────────────────
    # IMPORTANCE SCORING
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_importance(
        self,
        emotional_state,
        modulation,
        tool_results: tuple,
        assistant_output: str,
    ) -> float:
        cfg = self._config
        score = cfg.base_importance

        intensity = getattr(emotional_state, "intensity", 0.0)
        if intensity > cfg.high_intensity_threshold:
            score += cfg.bonus_high_intensity

        if tool_results:
            score += cfg.bonus_tool_used

        bias = abs(getattr(modulation, "motivational_bias", 0.0))
        if bias > cfg.strong_bias_threshold:
            score += cfg.bonus_strong_bias

        if len(assistant_output) > cfg.long_output_threshold:
            score += cfg.bonus_long_output

        # Negative emotions are more important to flag
        is_negative = getattr(emotional_state, "is_negative", False)
        if is_negative:
            score += cfg.bonus_negative_emotion

        if getattr(modulation, "emotional_validation", False):
            score += cfg.bonus_validation

        return max(cfg.min_importance, min(cfg.max_importance, score))

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        user_input: str,
        emotional_state,
        modulation,
        tool_results: tuple,
        importance: float,
    ) -> str:
        """
        Build a compact machine-readable summary of the turn.
        Format: [emotion|intensity] user_fragment [TOOL:name,...] → [tone]
        """
        parts: list[str] = []

        # Emotion prefix
        emotion = getattr(emotional_state, "primary_emotion", "unknown")
        intensity = getattr(emotional_state, "intensity", 0.0)
        parts.append(f"[{emotion}|{intensity:.2f}]")

        # User input fragment
        user_fragment = user_input[:80].strip().replace("\n", " ")
        parts.append(user_fragment)

        # Tools used
        if tool_results:
            tool_names = [getattr(r, "tool_name", "?") for r in tool_results]
            parts.append(f"[TOOLS:{','.join(tool_names)}]")

        # Modulation tone
        tone = getattr(modulation, "tone", "neutral")
        parts.append(f"→[{tone}]")

        summary = " ".join(parts)

        # Ensure reasonable length
        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # TAG EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_tags(self, emotional_state) -> list[str]:
        """Extract emotion tags from the emotional state for memory annotation."""
        if emotional_state is None:
            return []
        emotion = getattr(emotional_state, "primary_emotion", None)
        tags = [emotion] if emotion else []

        is_negative = getattr(emotional_state, "is_negative", False)
        is_positive = getattr(emotional_state, "is_positive", False)
        if is_negative:
            tags.append("negative_state")
        if is_positive:
            tags.append("positive_state")

        return tags

    # ─────────────────────────────────────────────────────────────────────────
    # GOAL SIGNAL DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_goal_signal(self, emotional_state, modulation) -> Optional[str]:
        """
        Infer the user's goal orientation from emotional + modulation signals.

        Signal taxonomy:
            push_forward : positive emotion + high motivational bias
            execute      : drive + high arousal
            stabilize    : negative emotion + calm/grounding tone
            recover      : fatigue detected
            explore      : doubt + low motivational bias
            None         : neutral / ambiguous
        """
        if emotional_state is None or modulation is None:
            return None

        emotion = getattr(emotional_state, "primary_emotion", "")
        is_positive = getattr(emotional_state, "is_positive", False)
        is_negative = getattr(emotional_state, "is_negative", False)
        is_high_arousal = getattr(emotional_state, "is_high_arousal", False)
        m_bias = getattr(modulation, "motivational_bias", 0.0)
        tone = getattr(modulation, "tone", "neutral")

        # Recover: fatigue always wins
        if emotion == "fatigue":
            return "recover"

        # Execute: drive state with high arousal
        if emotion == "drive" and is_high_arousal:
            return "execute"

        # Push forward: positive + motivated
        if is_positive and m_bias > 0.25:
            return "push_forward"

        # Stabilize: negative + calming/grounding response
        if is_negative and tone in ("calm", "grounding", "reassuring"):
            return "stabilize"

        # Explore: doubt state
        if emotion == "doubt":
            return "explore"

        # Default: no clear signal
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # TRAJECTORY SIGNAL
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_trajectory_signal(self, emotional_state, modulation) -> Optional[str]:
        """
        Detect the session arc trajectory from single-turn signals.

        Trajectory signals:
            progressing : positive + drive/confidence
            declining   : negative + fatigue/frustration + high intensity
            escalating  : high arousal + high intensity (positive)
            stable      : moderate intensity, neutral-ish
            None        : ambiguous
        """
        if emotional_state is None:
            return None

        emotion = getattr(emotional_state, "primary_emotion", "")
        intensity = getattr(emotional_state, "intensity", 0.0)
        is_positive = getattr(emotional_state, "is_positive", False)
        is_negative = getattr(emotional_state, "is_negative", False)
        is_high_arousal = getattr(emotional_state, "is_high_arousal", False)
        cfg = self._config

        # Declining: strong negative + fatigue or frustration
        if is_negative and intensity > cfg.high_intensity_threshold and emotion in ("fatigue", "frustration"):
            return "declining"

        # Escalating: high arousal + positive + high intensity
        if is_positive and is_high_arousal and intensity > cfg.high_intensity_threshold:
            return "escalating"

        # Progressing: positive, moderate intensity
        if is_positive and emotion in ("confidence", "drive", "flow"):
            return "progressing"

        # Stable: low-moderate intensity
        if intensity < 0.45:
            return "stable"

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_result(self, user_input: str, emotional_state) -> ReflectionResult:
        """Safe defaults when reflection computation fails."""
        return ReflectionResult(
            summary=f"[unknown] {user_input[:80]}",
            importance_score=self._config.base_importance,
            emotional_tags=[],
            goal_signal=None,
            trajectory_signal=None,
        )

    def explain(
        self,
        user_input: str,
        assistant_output: str,
        emotional_state,
        modulation,
        tool_results: tuple = (),
    ) -> str:
        """Human-readable breakdown of the reflection scoring for debugging."""
        cfg = self._config
        lines = ["ReflectionEngine — scoring trace:"]

        score = cfg.base_importance
        lines.append(f"  Base importance: {score:.2f}")

        intensity = getattr(emotional_state, "intensity", 0.0)
        if intensity > cfg.high_intensity_threshold:
            score += cfg.bonus_high_intensity
            lines.append(f"  +{cfg.bonus_high_intensity:.2f} high intensity ({intensity:.2f})")

        if tool_results:
            score += cfg.bonus_tool_used
            lines.append(f"  +{cfg.bonus_tool_used:.2f} tool used ({len(tool_results)} tools)")

        bias = abs(getattr(modulation, "motivational_bias", 0.0))
        if bias > cfg.strong_bias_threshold:
            score += cfg.bonus_strong_bias
            lines.append(f"  +{cfg.bonus_strong_bias:.2f} strong bias (|{bias:.2f}|)")

        if len(assistant_output) > cfg.long_output_threshold:
            score += cfg.bonus_long_output
            lines.append(f"  +{cfg.bonus_long_output:.2f} long output ({len(assistant_output)} chars)")

        if getattr(emotional_state, "is_negative", False):
            score += cfg.bonus_negative_emotion
            lines.append(f"  +{cfg.bonus_negative_emotion:.2f} negative emotion")

        if getattr(modulation, "emotional_validation", False):
            score += cfg.bonus_validation
            lines.append(f"  +{cfg.bonus_validation:.2f} validation active")

        clamped = max(cfg.min_importance, min(cfg.max_importance, score))
        lines.append(f"  → Final importance: {clamped:.2f}")
        lines.append(f"  Goal signal: {self._detect_goal_signal(emotional_state, modulation)!r}")
        lines.append(f"  Trajectory:  {self._detect_trajectory_signal(emotional_state, modulation)!r}")

        return "\n".join(lines)