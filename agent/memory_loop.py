"""
A.C.C.E.S.S. — Adaptive Memory Loop (Phase 3)

Provides higher-level memory maintenance operations on top of MemoryManager.

Design:
    - Manual trigger only — no background threads
    - Fully optional — memory_loop=None → no maintenance
    - Deterministic and testable
    - Returns MaintenanceReport for introspection and telemetry

Operations:
    decay_old_memories()
        Applies importance decay to all episodic memories using DecayEngine.
        Returns a list of updated (id, old_score, new_score) records.

    consolidate_low_importance()
        Identifies memories below consolidation_threshold and prepares them
        for summary compression. Phase 3: marks them with "consolidated" tag.
        Full compression (summarization) is a Phase 4 LLM feature.

    detect_recurrent_topics()
        Groups memories by tag co-occurrence to identify recurring themes.
        Returns a dict: tag → count.

    adjust_importance_based_on_repetition()
        Boosts importance of memories that share tags with recent high-importance
        memories. Repeated themes signal relevance.

All methods return structured reports — never raise on partial failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecayReport:
    memories_evaluated: int = 0
    memories_updated: int = 0
    memories_below_floor: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memories_evaluated":  self.memories_evaluated,
            "memories_updated":    self.memories_updated,
            "memories_below_floor": self.memories_below_floor,
            "duration_ms":         round(self.duration_ms, 2),
        }


@dataclass
class ConsolidationReport:
    candidates_found: int = 0
    tagged_for_review: int = 0
    threshold_used: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidates_found":  self.candidates_found,
            "tagged_for_review": self.tagged_for_review,
            "threshold_used":    round(self.threshold_used, 3),
            "duration_ms":       round(self.duration_ms, 2),
        }


@dataclass
class TopicReport:
    tag_counts: dict = field(default_factory=dict)
    recurrent_topics: list[str] = field(default_factory=list)  # tags appearing > min_count
    min_count: int = 2
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tag_counts":       self.tag_counts,
            "recurrent_topics": self.recurrent_topics,
            "min_count":        self.min_count,
            "duration_ms":      round(self.duration_ms, 2),
        }


@dataclass
class RepetitionAdjustmentReport:
    memories_boosted: int = 0
    boost_amount: float = 0.0
    recurrent_tags: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memories_boosted": self.memories_boosted,
            "boost_amount":     round(self.boost_amount, 3),
            "recurrent_tags":   self.recurrent_tags,
            "duration_ms":      round(self.duration_ms, 2),
        }


@dataclass
class MaintenanceReport:
    """
    Composite report from a full run_memory_maintenance() call.
    Contains sub-reports for each operation that was requested.
    """
    ran_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    decay: Optional[DecayReport] = None
    consolidation: Optional[ConsolidationReport] = None
    topics: Optional[TopicReport] = None
    repetition: Optional[RepetitionAdjustmentReport] = None
    errors: list[str] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ran_at":             self.ran_at.isoformat(),
            "total_duration_ms":  round(self.total_duration_ms, 2),
            "decay":              self.decay.to_dict() if self.decay else None,
            "consolidation":      self.consolidation.to_dict() if self.consolidation else None,
            "topics":             self.topics.to_dict() if self.topics else None,
            "repetition":         self.repetition.to_dict() if self.repetition else None,
            "errors":             self.errors,
        }

    def __repr__(self) -> str:
        parts = [f"MaintenanceReport(total={self.total_duration_ms:.0f}ms"]
        if self.decay:
            parts.append(f"decay={self.decay.memories_updated}/{self.decay.memories_evaluated}")
        if self.consolidation:
            parts.append(f"consolidation={self.consolidation.tagged_for_review}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        return ", ".join(parts) + ")"


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY LOOP
# ─────────────────────────────────────────────────────────────────────────────

class MemoryLoop:
    """
    Adaptive memory maintenance loop.

    Wraps a MemoryManager instance and provides higher-level operations.
    All operations are synchronous and manually triggered — no background threads.

    Usage:
        loop = MemoryLoop(
            memory_manager=manager,
            consolidation_threshold=0.3,
            recurrent_topic_min_count=3,
            repetition_boost=0.05,
        )

        # Called via agent.run_memory_maintenance()
        report = loop.run(
            run_decay=True,
            run_consolidation=True,
            run_topics=True,
            run_repetition=True,
        )
        print(report.decay.memories_updated)
    """

    def __init__(
        self,
        memory_manager,          # MemoryManager (duck-typed — no circular import)
        consolidation_threshold: float = 0.3,
        recurrent_topic_min_count: int = 3,
        repetition_boost: float = 0.05,
        repetition_boost_cap: float = 0.85,
    ):
        self._memory = memory_manager
        self._consolidation_threshold = consolidation_threshold
        self._recurrent_topic_min_count = recurrent_topic_min_count
        self._repetition_boost = repetition_boost
        self._repetition_boost_cap = repetition_boost_cap

    def run(
        self,
        run_decay: bool = True,
        run_consolidation: bool = True,
        run_topics: bool = True,
        run_repetition: bool = True,
    ) -> MaintenanceReport:
        """
        Execute requested maintenance operations and return a composite report.

        Any operation that fails is recorded in report.errors — never raises.
        """
        import time
        t_total = time.perf_counter()
        report = MaintenanceReport()

        if run_decay:
            try:
                report.decay = self.decay_old_memories()
            except Exception as exc:
                logger.error(f"[MemoryLoop] decay failed: {exc}", exc_info=True)
                report.errors.append(f"decay: {exc}")

        if run_topics:
            try:
                report.topics = self.detect_recurrent_topics()
            except Exception as exc:
                logger.error(f"[MemoryLoop] topic detection failed: {exc}", exc_info=True)
                report.errors.append(f"topics: {exc}")

        if run_consolidation:
            try:
                report.consolidation = self.consolidate_low_importance()
            except Exception as exc:
                logger.error(f"[MemoryLoop] consolidation failed: {exc}", exc_info=True)
                report.errors.append(f"consolidation: {exc}")

        if run_repetition and report.topics is not None:
            try:
                recurrent = report.topics.recurrent_topics
                report.repetition = self.adjust_importance_based_on_repetition(recurrent)
            except Exception as exc:
                logger.error(f"[MemoryLoop] repetition adjustment failed: {exc}", exc_info=True)
                report.errors.append(f"repetition: {exc}")

        report.total_duration_ms = (time.perf_counter() - t_total) * 1000
        logger.info(f"MemoryLoop maintenance done — {report}")
        return report

    def decay_old_memories(self) -> DecayReport:
        """
        Apply importance decay to all memories using MemoryManager's decay engine.

        Reads all memory records, computes new importance scores,
        and bulk-updates the store.
        """
        import time
        t0 = time.perf_counter()
        report = DecayReport()

        # Use MemoryManager's run_decay if available (preferred)
        if hasattr(self._memory, "run_decay"):
            try:
                result = self._memory.run_decay()
                # result is DecayResult from DecayEngine
                report.memories_evaluated = getattr(result, "processed", 0)
                report.memories_updated = getattr(result, "updated", 0)
                report.memories_below_floor = getattr(result, "below_floor", 0)
            except Exception as exc:
                logger.warning(f"run_decay failed, falling back: {exc}")
                report.memories_evaluated = 0
        else:
            logger.warning("MemoryManager has no run_decay() method — skipping decay")

        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report

    def consolidate_low_importance(self) -> ConsolidationReport:
        """
        Identify and tag memories below consolidation_threshold for review.

        Phase 3 implementation: marks candidates with "consolidated" tag.
        Phase 4: will call LLM to summarize and replace content.
        """
        import time
        t0 = time.perf_counter()
        report = ConsolidationReport(threshold_used=self._consolidation_threshold)

        if not hasattr(self._memory, "_store"):
            logger.warning("MemoryManager has no _store — cannot consolidate")
            report.duration_ms = (time.perf_counter() - t0) * 1000
            return report

        store = self._memory._store
        try:
            candidates = store.get_purgeable_candidates(
                exclude_types=("semantic",),
                max_importance=self._consolidation_threshold,
            )
            report.candidates_found = len(candidates)

            # Tag each candidate for review
            for c in candidates:
                try:
                    store.add_tags(c["id"], ["consolidated"])
                    report.tagged_for_review += 1
                except Exception as exc:
                    logger.warning(f"Failed to tag {c['id'][:8]}: {exc}")

        except Exception as exc:
            logger.error(f"consolidate_low_importance error: {exc}", exc_info=True)

        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report

    def detect_recurrent_topics(self) -> TopicReport:
        """
        Analyze memory tags to find recurring themes.

        Returns a TopicReport with tag_counts and recurrent_topics
        (tags appearing at least recurrent_topic_min_count times).
        """
        import time
        t0 = time.perf_counter()
        report = TopicReport(min_count=self._recurrent_topic_min_count)

        if not hasattr(self._memory, "_store"):
            logger.warning("MemoryManager has no _store — cannot detect topics")
            report.duration_ms = (time.perf_counter() - t0) * 1000
            return report

        store = self._memory._store
        try:
            # Get all memory records
            all_ids = store.get_all_ids()
            records = store.get_by_ids(all_ids) if all_ids else []

            # Count tags
            from collections import Counter
            tag_counter: Counter = Counter()
            for record in records:
                tags = getattr(record, "tags", []) or []
                for tag in tags:
                    if tag and tag not in ("consolidated",):  # skip meta-tags
                        tag_counter[tag] += 1

            report.tag_counts = dict(tag_counter.most_common())
            report.recurrent_topics = [
                tag for tag, count in tag_counter.items()
                if count >= self._recurrent_topic_min_count
            ]

        except Exception as exc:
            logger.error(f"detect_recurrent_topics error: {exc}", exc_info=True)

        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report

    def adjust_importance_based_on_repetition(
        self,
        recurrent_tags: list[str],
    ) -> RepetitionAdjustmentReport:
        """
        Boost importance of memories that share tags with recurrent topics.

        Repeated themes signal long-term relevance → they should survive decay longer.

        Args:
            recurrent_tags: List of tags identified as recurrent by detect_recurrent_topics().
        """
        import time
        t0 = time.perf_counter()
        report = RepetitionAdjustmentReport(
            boost_amount=self._repetition_boost,
            recurrent_tags=list(recurrent_tags),
        )

        if not recurrent_tags or not hasattr(self._memory, "_store"):
            report.duration_ms = (time.perf_counter() - t0) * 1000
            return report

        store = self._memory._store
        try:
            all_ids = store.get_all_ids()
            records = store.get_by_ids(all_ids) if all_ids else []
            recurrent_set = set(recurrent_tags)

            updates: list[tuple[float, str]] = []
            for record in records:
                tags = set(getattr(record, "tags", []) or [])
                if tags & recurrent_set:
                    new_score = min(
                        self._repetition_boost_cap,
                        record.importance_score + self._repetition_boost,
                    )
                    if new_score > record.importance_score:
                        updates.append((new_score, record.id))
                        report.memories_boosted += 1

            if updates:
                store.bulk_update_importance(updates)
                logger.info(
                    f"RepetitionAdjustment: boosted {len(updates)} memories "
                    f"by +{self._repetition_boost}"
                )

        except Exception as exc:
            logger.error(f"adjust_importance error: {exc}", exc_info=True)

        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report