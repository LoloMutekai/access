"""
A.C.C.E.S.S. — Trajectory Tracker (Phase 3)

Maintains a lightweight model of the user's goal orientation over a session.

Design:
    - Stateful, lives on AgentCore
    - Updated after each reflection
    - Pure computation — no external calls
    - Serializable via to_dict()

TrajectoryState captures:
    - dominant_goal_signal : most frequent goal signal in the recent window
    - recent_goal_signals  : ordered list of recent signals (newest last)
    - drift_score          : [0.0, 1.0] — how much the goal has shifted recently
    - dominant_trajectory  : most frequent trajectory signal in the recent window

Drift score algorithm:
    Count distinct goal signals in the last N turns.
    drift = (distinct_count - 1) / (window_size - 1)
    0.0 = completely consistent, 1.0 = maximally inconsistent

This captures "is the user bouncing between goals?" without needing another LLM call.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

logger = logging.getLogger(__name__)

# Valid goal signal values (including None → stored as "none")
VALID_GOAL_SIGNALS = frozenset({
    "push_forward", "execute", "stabilize", "recover", "explore", None
})

# Valid trajectory signals
VALID_TRAJECTORY_SIGNALS = frozenset({
    "progressing", "declining", "stable", "escalating", None
})


@dataclass
class TrajectoryState:
    """
    Snapshot of the user's goal trajectory at a point in time.

    dominant_goal_signal:
        The most frequent goal signal across the recent window.
        None if the window is empty or signals are uniformly distributed.

    recent_goal_signals:
        Ordered list (oldest → newest) of the last N goal signals.

    drift_score:
        [0.0, 1.0] — 0.0 = consistent goal, 1.0 = maximally drifting.

    dominant_trajectory:
        Most frequent trajectory signal in recent window (progressing/declining/etc.).

    turn_count:
        Total turns processed since tracker was created or reset.

    updated_at:
        UTC timestamp of last update.
    """
    dominant_goal_signal: Optional[str] = None
    recent_goal_signals: list[Optional[str]] = field(default_factory=list)
    drift_score: float = 0.0
    dominant_trajectory: Optional[str] = None
    turn_count: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "dominant_goal_signal": self.dominant_goal_signal,
            "recent_goal_signals":  [s or "none" for s in self.recent_goal_signals],
            "drift_score":          round(self.drift_score, 3),
            "dominant_trajectory":  self.dominant_trajectory,
            "turn_count":           self.turn_count,
            "updated_at":           self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"TrajectoryState("
            f"goal={self.dominant_goal_signal!r}, "
            f"drift={self.drift_score:.2f}, "
            f"traj={self.dominant_trajectory!r}, "
            f"turns={self.turn_count})"
        )


class TrajectoryTracker:
    """
    Maintains and updates TrajectoryState from reflection results.

    Usage (called by AgentCore after each reflection):
        tracker = TrajectoryTracker(window_size=10)
        tracker.update(goal_signal="push_forward", trajectory_signal="progressing")
        state = tracker.state
        print(state.drift_score)
    """

    def __init__(self, window_size: int = 10):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self._window_size = window_size
        self._goal_signals: list[Optional[str]] = []
        self._trajectory_signals: list[Optional[str]] = []
        self._turn_count: int = 0

    def update(
        self,
        goal_signal: Optional[str],
        trajectory_signal: Optional[str],
    ) -> TrajectoryState:
        """
        Update tracker with signals from the latest reflection.

        Args:
            goal_signal:       From ReflectionResult.goal_signal.
            trajectory_signal: From ReflectionResult.trajectory_signal.

        Returns:
            New TrajectoryState snapshot.
        """
        self._turn_count += 1

        # Append and trim to window
        self._goal_signals.append(goal_signal)
        if len(self._goal_signals) > self._window_size:
            self._goal_signals = self._goal_signals[-self._window_size:]

        self._trajectory_signals.append(trajectory_signal)
        if len(self._trajectory_signals) > self._window_size:
            self._trajectory_signals = self._trajectory_signals[-self._window_size:]

        # Compute dominant signals (exclude None for dominance)
        dominant_goal = self._dominant_signal(self._goal_signals)
        dominant_traj = self._dominant_signal(self._trajectory_signals)

        # Compute drift
        drift = self._compute_drift(self._goal_signals)

        state = TrajectoryState(
            dominant_goal_signal=dominant_goal,
            recent_goal_signals=list(self._goal_signals),
            drift_score=drift,
            dominant_trajectory=dominant_traj,
            turn_count=self._turn_count,
        )

        logger.debug(
            f"TrajectoryTracker: turn={self._turn_count}, "
            f"goal={dominant_goal}, drift={drift:.2f}, traj={dominant_traj}"
        )
        return state

    def reset(self) -> None:
        """Reset all trajectory state (e.g., on session change)."""
        self._goal_signals = []
        self._trajectory_signals = []
        self._turn_count = 0
        logger.info("TrajectoryTracker reset")

    @property
    def state(self) -> TrajectoryState:
        """Return current trajectory state without updating it."""
        if not self._goal_signals:
            return TrajectoryState(turn_count=self._turn_count)
        return TrajectoryState(
            dominant_goal_signal=self._dominant_signal(self._goal_signals),
            recent_goal_signals=list(self._goal_signals),
            drift_score=self._compute_drift(self._goal_signals),
            dominant_trajectory=self._dominant_signal(self._trajectory_signals),
            turn_count=self._turn_count,
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_signal(signals: list[Optional[str]]) -> Optional[str]:
        """
        Return the most frequent non-None signal.
        Returns None if all signals are None or list is empty.
        """
        non_none = [s for s in signals if s is not None]
        if not non_none:
            return None
        return Counter(non_none).most_common(1)[0][0]

    @staticmethod
    def _compute_drift(signals: list[Optional[str]]) -> float:
        """
        Drift = proportion of distinct non-None signals in the window.

        Formula: (distinct_count - 1) / max(1, len(signals) - 1)
        0.0 = all same signal → no drift
        1.0 = all different → maximum drift

        None signals are excluded from the distinct count (they are "gaps").
        """
        non_none = [s for s in signals if s is not None]
        if len(non_none) < 2:
            return 0.0

        distinct = len(set(non_none))
        max_possible = len(non_none) - 1
        return min(1.0, (distinct - 1) / max_possible)