"""
A.C.C.E.S.S. — Memory Decay Engine

Implements time-based importance decay.

Design rationale:
- Decay is EXPONENTIAL, not linear. This mirrors how human memory works:
  recent forgetting is fast, then it slows down.
- Protected memories (importance >= protection_threshold) decay slower.
- Access count acts as a reinforcement signal: frequently recalled memories
  resist decay more.
- Decay runs as a background task, not on every read.

Formula:
    new_importance = current * decay_factor^(hours_elapsed / half_life_hours)
    
    Where decay_factor is modulated by:
    - protection threshold (high importance = slower decay)
    - access_count reinforcement
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecayConfig:
    # How many hours until an unaccessed memory loses half its importance
    half_life_hours: float = 72.0           # 3 days baseline

    # Memories above this threshold decay at half the normal rate
    protection_threshold: float = 0.8

    # Each access adds this bonus to the half-life (in hours)
    access_reinforcement_hours: float = 12.0

    # Memories below this floor are candidates for purge
    decay_floor: float = 0.05

    # Never decay below this — prevents complete erasure without explicit delete
    absolute_minimum: float = 0.05

    # Memory types that are fully exempt from decay
    exempt_types: tuple = ("semantic",)     # facts don't decay, events do


@dataclass
class DecayResult:
    """Result of a decay run."""
    processed: int          # total memories evaluated
    updated: int            # memories whose score actually changed
    below_floor: int        # memories now at or below decay_floor (purge candidates)
    skipped_exempt: int     # memories skipped due to exempt_types
    duration_ms: float      # processing time


class DecayEngine:
    """
    Computes and applies importance decay to memory records.

    This class is PURE — it takes records and returns decay values.
    It does NOT write to the database directly: MemoryManager orchestrates that.
    This makes it fully unit-testable without any DB setup.
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()

    def compute_decay(
        self,
        current_importance: float,
        memory_type: str,
        created_at: datetime,
        last_accessed_at: Optional[datetime],
        access_count: int,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Compute the new importance score after decay.

        Args:
            current_importance: current score [0.0, 1.0]
            memory_type: used to check exemptions
            created_at: when memory was created
            last_accessed_at: last retrieval time (None → use created_at)
            access_count: total number of retrievals
            reference_time: compute decay relative to this time (default: now, for testability)

        Returns:
            New importance score, clamped to [absolute_minimum, 1.0]
        """
        # Exempt types skip decay entirely
        if memory_type in self.config.exempt_types:
            return current_importance

        now = reference_time or datetime.utcnow()
        last_event = last_accessed_at or created_at

        hours_elapsed = max(0.0, (now - last_event).total_seconds() / 3600)

        if hours_elapsed == 0:
            return current_importance

        # Compute effective half-life with reinforcements
        reinforcement = access_count * self.config.access_reinforcement_hours
        effective_half_life = self.config.half_life_hours + reinforcement

        # Protected memories decay at half speed
        if current_importance >= self.config.protection_threshold:
            effective_half_life *= 2.0

        # Exponential decay: score * 0.5^(t / half_life)
        decay_factor = math.pow(0.5, hours_elapsed / effective_half_life)
        new_score = current_importance * decay_factor

        return max(self.config.absolute_minimum, min(1.0, new_score))

    def batch_compute(
        self,
        records: list[dict],  # list of dicts with required fields
        reference_time: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Compute decay for a batch of records.

        Args:
            records: list of dicts with keys:
                     id, importance_score, memory_type, created_at,
                     last_accessed_at (optional), access_count
            reference_time: for testability

        Returns:
            List of dicts: {id, old_score, new_score, changed, below_floor}
        """
        now = reference_time or datetime.utcnow()
        results = []

        for r in records:
            old_score = r["importance_score"]
            new_score = self.compute_decay(
                current_importance=old_score,
                memory_type=r["memory_type"],
                created_at=r["created_at"],
                last_accessed_at=r.get("last_accessed_at"),
                access_count=r.get("access_count", 0),
                reference_time=now,
            )
            results.append({
                "id": r["id"],
                "old_score": old_score,
                "new_score": new_score,
                "changed": abs(new_score - old_score) > 1e-6,
                "below_floor": new_score <= self.config.decay_floor,
            })

        return results