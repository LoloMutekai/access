"""
A.C.C.E.S.S. — Internal Goal Queue (Phase 4)

Self-generated goals derived from reflection, trajectory, and user signals.

Design:
    - InternalGoal is FROZEN
    - GoalQueue is the only mutable component — managed list of goals
    - Goals have priority decay (goals that aren't acted on fade)
    - Priority: higher = more urgent
    - Origin tracks where the goal came from (reflection / autonomy / user)
    - Deduplication by description similarity (exact match)
    - Maximum capacity enforced

Integration:
    After each finalized turn, AgentCore can optionally inject a goal
    based on reflection.goal_signal + trajectory drift patterns.

Goal lifecycle:
    created → active (priority may shift) → completed | expired (priority < floor)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalQueueConfig:
    max_goals: int = 20              # hard cap on simultaneous goals
    priority_decay_rate: float = 0.02   # per decay cycle
    priority_floor: float = 0.05     # below this → auto-expire
    default_priority: float = 0.5
    max_priority: float = 1.0
    min_priority: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GOAL
# ─────────────────────────────────────────────────────────────────────────────

VALID_ORIGINS = frozenset({"reflection", "autonomy", "user", "trajectory"})
VALID_STATUSES = frozenset({"active", "completed", "expired"})

@dataclass(frozen=True)
class InternalGoal:
    """Immutable goal record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    description: str = ""
    priority: float = 0.5
    origin: str = "reflection"
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        object.__setattr__(self, "priority", max(0.0, min(1.0, self.priority)))
        if self.origin not in VALID_ORIGINS:
            object.__setattr__(self, "origin", "reflection")
        if self.status not in VALID_STATUSES:
            object.__setattr__(self, "status", "active")

    def with_priority(self, new_priority: float) -> "InternalGoal":
        """Return a copy with updated priority."""
        return InternalGoal(
            id=self.id, description=self.description,
            priority=max(0.0, min(1.0, new_priority)),
            origin=self.origin, status=self.status,
            created_at=self.created_at,
        )

    def with_status(self, new_status: str) -> "InternalGoal":
        return InternalGoal(
            id=self.id, description=self.description,
            priority=self.priority, origin=self.origin,
            status=new_status, created_at=self.created_at,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "priority": round(self.priority, 4),
            "origin": self.origin,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InternalGoal":
        from .relationship_state import _parse_dt
        return cls(
            id=d.get("id", str(uuid.uuid4())[:12]),
            description=d.get("description", ""),
            priority=d.get("priority", 0.5),
            origin=d.get("origin", "reflection"),
            status=d.get("status", "active"),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    def __repr__(self) -> str:
        return (
            f"Goal({self.id}, p={self.priority:.2f}, "
            f"origin={self.origin!r}, "
            f"desc='{self.description[:40]}')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE
# ─────────────────────────────────────────────────────────────────────────────

class GoalQueue:
    """
    Managed collection of InternalGoals.
    Thread-safe for single-threaded agent (no locks needed in Phase 4).
    """

    def __init__(self, config: Optional[GoalQueueConfig] = None):
        self._cfg = config or GoalQueueConfig()
        self._goals: list[InternalGoal] = []

    def add_goal(
        self,
        description: str,
        priority: Optional[float] = None,
        origin: str = "reflection",
    ) -> Optional[InternalGoal]:
        """
        Add a goal. Returns the goal if added, None if duplicate or at capacity.
        """
        # Dedup by exact description match among active goals
        for g in self._goals:
            if g.status == "active" and g.description == description:
                logger.debug(f"Goal already exists: {description[:40]}")
                return None

        # Capacity check — expire lowest priority if full
        active = [g for g in self._goals if g.status == "active"]
        if len(active) >= self._cfg.max_goals:
            self._expire_lowest()

        goal = InternalGoal(
            description=description,
            priority=priority if priority is not None else self._cfg.default_priority,
            origin=origin,
        )
        self._goals.append(goal)
        logger.debug(f"Goal added: {goal}")
        return goal

    def pop_highest_priority(self) -> Optional[InternalGoal]:
        """Remove and return the highest-priority active goal."""
        active = [(i, g) for i, g in enumerate(self._goals) if g.status == "active"]
        if not active:
            return None
        idx, best = max(active, key=lambda x: x[1].priority)
        self._goals[idx] = best.with_status("completed")
        return best

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed by ID. Returns True if found."""
        for i, g in enumerate(self._goals):
            if g.id == goal_id and g.status == "active":
                self._goals[i] = g.with_status("completed")
                return True
        return False

    def decay_priorities(self) -> int:
        """
        Apply priority decay to all active goals.
        Returns count of goals that expired (dropped below floor).
        """
        expired_count = 0
        for i, g in enumerate(self._goals):
            if g.status != "active":
                continue
            new_p = g.priority - self._cfg.priority_decay_rate
            if new_p < self._cfg.priority_floor:
                self._goals[i] = g.with_status("expired")
                expired_count += 1
            else:
                self._goals[i] = g.with_priority(new_p)
        return expired_count

    def boost_priority(self, goal_id: str, amount: float) -> bool:
        """Increase a goal's priority. Returns True if found."""
        for i, g in enumerate(self._goals):
            if g.id == goal_id and g.status == "active":
                self._goals[i] = g.with_priority(g.priority + amount)
                return True
        return False

    def list_active(self) -> list[InternalGoal]:
        """Return active goals sorted by priority descending."""
        return sorted(
            [g for g in self._goals if g.status == "active"],
            key=lambda g: -g.priority,
        )

    def list_all(self) -> list[InternalGoal]:
        return list(self._goals)

    @property
    def active_count(self) -> int:
        return sum(1 for g in self._goals if g.status == "active")

    @property
    def total_count(self) -> int:
        return len(self._goals)

    def to_dict(self) -> dict:
        return {
            "goals": [g.to_dict() for g in self._goals],
            "active_count": self.active_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, d: dict, config: Optional[GoalQueueConfig] = None) -> "GoalQueue":
        queue = cls(config=config)
        for gd in d.get("goals", []):
            queue._goals.append(InternalGoal.from_dict(gd))
        return queue

    def _expire_lowest(self) -> None:
        """Expire the lowest-priority active goal to make room."""
        active = [(i, g) for i, g in enumerate(self._goals) if g.status == "active"]
        if not active:
            return
        idx, worst = min(active, key=lambda x: x[1].priority)
        self._goals[idx] = worst.with_status("expired")
        logger.debug(f"Expired goal to make room: {worst.id}")