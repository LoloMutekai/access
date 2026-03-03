"""
A.C.C.E.S.S. — Memory Maintenance

Two responsibilities, one module (intentionally — they share the same concern: DB health):
1. ConsistencyChecker — verify SQLite ↔ FAISS are in sync
2. PurgePolicy       — decide what to delete when max_episodic_entries is exceeded
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore
    from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────

@dataclass
class ConsistencyReport:
    """Result of a DB ↔ FAISS consistency check."""
    db_count: int
    faiss_count: int
    in_sync: bool

    # IDs present in DB but missing from FAISS (embeddings lost)
    orphaned_in_db: list[str] = field(default_factory=list)

    # Int IDs present in FAISS but not in the UUID map (should never happen)
    orphaned_in_faiss: list[str] = field(default_factory=list)

    checked_at: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        status = "✅ IN SYNC" if self.in_sync else "❌ INCONSISTENT"
        lines = [
            f"[Consistency Check — {self.checked_at.strftime('%Y-%m-%d %H:%M:%S')}]",
            f"Status          : {status}",
            f"DB records      : {self.db_count}",
            f"FAISS vectors   : {self.faiss_count}",
        ]
        if self.orphaned_in_db:
            lines.append(f"Orphaned in DB  : {len(self.orphaned_in_db)} IDs (embeddings missing)")
            for oid in self.orphaned_in_db[:5]:
                lines.append(f"  - {oid[:8]}...")
            if len(self.orphaned_in_db) > 5:
                lines.append(f"  ... and {len(self.orphaned_in_db) - 5} more")
        if self.orphaned_in_faiss:
            lines.append(f"Orphaned in FAISS: {len(self.orphaned_in_faiss)} UUIDs (no DB entry)")
        return "\n".join(lines)


class ConsistencyChecker:
    """
    Verifies that SQLite metadata and FAISS index are in sync.

    Detects two failure modes:
    A) Memory in DB but not in FAISS → retrieval will silently miss it
    B) Vector in FAISS map but not in DB → ghost vector, wastes search capacity

    Does NOT auto-repair (MemoryManager decides repair strategy).
    """

    def __init__(self, store: "MemoryStore", index: "VectorIndex"):
        self._store = store
        self._index = index

    def check(self) -> ConsistencyReport:
        db_count = self._store.count()
        faiss_count = self._index.total_vectors

        # Get all IDs from DB
        db_ids = set(self._store.get_all_ids())

        # Get all UUIDs known to FAISS id map
        faiss_uuids = set(self._index.get_all_uuids())

        orphaned_in_db = list(db_ids - faiss_uuids)
        orphaned_in_faiss = list(faiss_uuids - db_ids)

        in_sync = (
            db_count == faiss_count
            and len(orphaned_in_db) == 0
            and len(orphaned_in_faiss) == 0
        )

        report = ConsistencyReport(
            db_count=db_count,
            faiss_count=faiss_count,
            in_sync=in_sync,
            orphaned_in_db=orphaned_in_db,
            orphaned_in_faiss=orphaned_in_faiss,
        )

        if not in_sync:
            logger.warning(f"Consistency check FAILED:\n{report.summary()}")
        else:
            logger.info(f"Consistency check PASSED — {db_count} records in sync")

        return report


# ─────────────────────────────────────────────────────────────
# PURGE POLICY
# ─────────────────────────────────────────────────────────────

class PurgeStrategy(Enum):
    LOWEST_IMPORTANCE = "lowest_importance"     # delete least important first
    OLDEST_UNACCESSED  = "oldest_unaccessed"    # delete oldest never-accessed first
    COMBINED           = "combined"             # score = importance * recency * access


@dataclass
class PurgeCandidate:
    id: str
    importance_score: float
    created_at: datetime
    last_accessed_at: datetime | None
    access_count: int
    purge_score: float      # lower = more likely to be purged


@dataclass
class PurgePlan:
    """Describes what would be deleted — not yet executed."""
    total_current: int
    max_allowed: int
    to_delete: list[PurgeCandidate]
    triggered: bool         # False if count < max (no purge needed)

    def summary(self) -> str:
        if not self.triggered:
            return f"[Purge] Not needed — {self.total_current}/{self.max_allowed} entries"
        return (
            f"[Purge Plan] {self.total_current} entries → "
            f"deleting {len(self.to_delete)} → "
            f"target ≤ {self.max_allowed}"
        )


class PurgePolicy:
    """
    Determines which memories to delete when max_episodic_entries is exceeded.

    Design:
    - Only purges 'episodic' memories (semantic and emotional are protected)
    - Never purges importance >= 0.8 (protected tier)
    - Purges down to 90% of max to create headroom (avoids thrashing)
    - Returns a PurgePlan — MemoryManager decides whether to execute it
    """

    PROTECTED_IMPORTANCE = 0.8
    HEADROOM_FACTOR = 0.9       # purge to 90% of max, not 100%
    PROTECTED_TYPES = ("semantic", "emotional")

    def __init__(self, max_entries: int, strategy: PurgeStrategy = PurgeStrategy.COMBINED):
        self.max_entries = max_entries
        self.strategy = strategy

    def build_plan(self, store: "MemoryStore") -> PurgePlan:
        """
        Build a purge plan without executing it.
        Only evaluates 'episodic' memories below the protection threshold.
        """
        total = store.count()
        if total <= self.max_entries:
            return PurgePlan(
                total_current=total,
                max_allowed=self.max_entries,
                to_delete=[],
                triggered=False,
            )

        target = int(self.max_entries * self.HEADROOM_FACTOR)
        need_to_delete = total - target

        # Fetch purgeable candidates from DB
        raw = store.get_purgeable_candidates(
            exclude_types=self.PROTECTED_TYPES,
            max_importance=self.PROTECTED_IMPORTANCE,
        )

        # Score candidates
        candidates = [self._score(r) for r in raw]
        candidates.sort(key=lambda c: c.purge_score)  # lowest score = delete first

        to_delete = candidates[:need_to_delete]

        plan = PurgePlan(
            total_current=total,
            max_allowed=self.max_entries,
            to_delete=to_delete,
            triggered=True,
        )
        logger.warning(plan.summary())
        return plan

    def _score(self, row: dict) -> PurgeCandidate:
        """
        Compute a purge score. Lower = more expendable.
        
        Combined formula:
            purge_score = importance * recency_factor * (1 + log(1 + access_count))
        
        So a memory survives if it's important, recent, OR frequently accessed.
        """
        importance = row["importance_score"]
        access_count = row.get("access_count", 0)
        now = datetime.utcnow()
        last_event = row.get("last_accessed_at") or row["created_at"]

        hours_since = max(1.0, (now - last_event).total_seconds() / 3600)
        recency_factor = 1.0 / (1.0 + hours_since / 24.0)  # decays over days

        import math
        access_bonus = 1.0 + math.log1p(access_count)

        if self.strategy == PurgeStrategy.LOWEST_IMPORTANCE:
            purge_score = importance
        elif self.strategy == PurgeStrategy.OLDEST_UNACCESSED:
            purge_score = recency_factor
        else:  # COMBINED
            purge_score = importance * recency_factor * access_bonus

        return PurgeCandidate(
            id=row["id"],
            importance_score=importance,
            created_at=row["created_at"],
            last_accessed_at=row.get("last_accessed_at"),
            access_count=access_count,
            purge_score=purge_score,
        )