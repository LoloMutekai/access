"""
A.C.C.E.S.S. — Memory Manager v2

Changes from v1:
- retrieve_relevant_memories() supports optional emotional_context for emotion-aware ranking
- set_emotion_alignment() allows EmotionEngine to inject alignment function (no circular dep)
- add_tags_to_memory() — public API for tag mutation
- get_recent_memories() — recent memories by creation time (used by EmotionEngine protection)
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

from sentence_transformers import SentenceTransformer

from .config import MemoryConfig
from .decay import DecayEngine, DecayConfig, DecayResult
from .maintenance import ConsistencyChecker, ConsistencyReport, PurgePolicy, PurgePlan, PurgeStrategy
from .models import MemoryRecord, RetrievedMemory
from .store import MemoryStore
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Public interface for the A.C.C.E.S.S. memory system.

    Orchestrates:
    - MemoryStore  → SQLite metadata
    - VectorIndex  → FAISS embeddings
    - SentenceTransformer → embedding generation

    External code should ONLY interact with this class.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._store = MemoryStore(self.config)
        self._index = VectorIndex(self.config)
        self._embedder = SentenceTransformer(self.config.embedding_model)
        self._decay = DecayEngine(DecayConfig())
        self._checker = ConsistencyChecker(self._store, self._index)
        self._purge = PurgePolicy(
            max_entries=self.config.max_episodic_entries,
            strategy=PurgeStrategy.COMBINED,
        )
        # Emotional alignment function — injected by EmotionEngine
        # Signature: (MemoryRecord, EmotionalState) → float [0.0, 1.0]
        # None until EmotionEngine registers it via set_emotion_alignment()
        self._alignment_fn: Optional[Callable] = None

        logger.info(
            f"MemoryManager ready — "
            f"{self._store.count()} records in DB, "
            f"{self._index.total_vectors} vectors in FAISS"
        )

    def set_emotion_alignment(self, alignment_fn: Callable) -> None:
        """
        Register an emotional alignment function.

        Called by EmotionEngine.__init__() to inject EmotionAlignment.compute.
        This avoids circular imports: MemoryManager knows nothing about emotion types.

        Args:
            alignment_fn: callable(record, emotional_context) → float [0.0, 1.0]
        """
        self._alignment_fn = alignment_fn
        logger.info("Emotion alignment function registered.")

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    # ─────────────────────────────────────────────────────────────
    # ADD MEMORY
    # ─────────────────────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        summary: Optional[str] = None,
        memory_type: str = "episodic",
        tags: Optional[list[str]] = None,
        importance_score: Optional[float] = None,
        source: str = "interaction",
        session_id: Optional[str] = None,
    ) -> MemoryRecord:
        """
        Creates and stores a new memory entry.

        - Embeds the `content` field (full text → better semantic search).
        - `summary` is used for RAG injection (shorter = cheaper LLM context).
          If not provided, defaults to content[:300].

        Returns the created MemoryRecord.
        """
        record = MemoryRecord(
            content=content,
            summary=summary or content[:300],
            memory_type=memory_type,
            tags=tags or [],
            importance_score=importance_score or self.config.default_importance,
            source=source,
            session_id=session_id,
        )

        embedding = self._embed(content)
        record.embedding = embedding

        self._store.insert(record)
        self._index.add(record.id, embedding)

        logger.info(
            f"Memory added [{record.memory_type}] "
            f"id={record.id[:8]}... "
            f"importance={record.importance_score:.2f}"
        )
        return record

    # ─────────────────────────────────────────────────────────────
    # RETRIEVE RELEVANT MEMORIES
    # ─────────────────────────────────────────────────────────────

    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        emotional_context: Optional[Any] = None,
    ) -> list[RetrievedMemory]:
        """
        Returns the most relevant memories for a given query.

        Ranking formula:
            v1: relevance = similarity × importance
            v2: relevance = similarity × importance × emotional_alignment  (if emotional_context given)

        Emotional alignment is computed by the registered alignment_fn (injected by EmotionEngine).
        If no alignment_fn is registered, falls back to v1 formula (alignment = 1.0).

        Args:
            query: natural language query (will be embedded)
            top_k: max results (default: config.default_top_k)
            memory_type: optional filter ('episodic', 'semantic', etc.)
            min_importance: filter out memories below this score
            emotional_context: optional EmotionalState for emotion-aware ranking
        """
        k = top_k or self.config.default_top_k

        query_embedding = self._embed(query)
        candidates = self._index.search(query_embedding, top_k=k * 3)

        if not candidates:
            return []

        candidate_ids = [uid for uid, _ in candidates]
        records = self._store.get_by_ids(candidate_ids)
        sim_map = {uid: sim for uid, sim in candidates}

        results: list[RetrievedMemory] = []
        for record in records:
            if record.importance_score < min_importance:
                continue
            if memory_type and record.memory_type != memory_type:
                continue

            similarity = sim_map.get(record.id, 0.0)

            # Emotional alignment modulation
            alignment = 1.0
            if emotional_context is not None and self._alignment_fn is not None:
                alignment = self._alignment_fn(record, emotional_context)

            relevance = similarity * record.importance_score * alignment

            results.append(RetrievedMemory(
                record=record,
                similarity=similarity,
                relevance=relevance,
            ))

        results.sort(key=lambda r: r.relevance, reverse=True)
        results = results[:k]

        for r in results:
            self._store.mark_accessed(r.record.id)

        logger.debug(f"Retrieved {len(results)} memories for query: '{query[:60]}...'")
        return results

    # ─────────────────────────────────────────────────────────────
    # UPDATE IMPORTANCE
    # ─────────────────────────────────────────────────────────────

    def update_importance(
        self,
        memory_id: str,
        new_score: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> float:
        """
        Update importance score. Two modes:
        - Absolute: update_importance(id, new_score=0.9)
        - Relative: update_importance(id, delta=+0.1)
        Returns the final clamped score.
        """
        if new_score is None and delta is None:
            raise ValueError("Provide either new_score or delta.")

        if delta is not None:
            record = self._store.get_by_id(memory_id)
            if record is None:
                raise ValueError(f"Memory not found: {memory_id}")
            new_score = record.importance_score + delta

        final_score = max(self.config.importance_min, min(self.config.importance_max, new_score))
        self._store.update_importance(memory_id, final_score)

        logger.debug(f"Importance updated for {memory_id[:8]}... → {final_score:.3f}")
        return final_score

    # ─────────────────────────────────────────────────────────────
    # TAGS
    # ─────────────────────────────────────────────────────────────

    def add_tags_to_memory(self, memory_id: str, tags: list[str]) -> None:
        """
        Add tags to an existing memory (non-destructive — appends, no duplicates).
        Used by EmotionEngine for protection tagging (emotion_stress, emotion_boost).
        """
        self._store.add_tags(memory_id, tags)

    # ─────────────────────────────────────────────────────────────
    # RECENT MEMORIES
    # ─────────────────────────────────────────────────────────────

    def get_recent_memories(
        self,
        n: int = 10,
        min_importance: float = 0.0,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """
        Return the N most recently created memories.
        Used by EmotionEngine to find candidates for protection tagging.
        """
        return self._store.get_recent_memories(
            n=n,
            min_importance=min_importance,
            memory_type=memory_type,
        )

    # ─────────────────────────────────────────────────────────────
    # UTILS
    # ─────────────────────────────────────────────────────────────

    def delete_memory(self, memory_id: str) -> None:
        self._store.delete(memory_id)
        self._index.remove(memory_id)
        logger.info(f"Memory deleted: {memory_id[:8]}...")

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._store.get_by_id(memory_id)

    def stats(self) -> dict:
        return {
            "total_memories": self._store.count(),
            "total_vectors": self._index.total_vectors,
            "embedding_model": self.config.embedding_model,
            "db_path": str(self.config.db_path),
            "index_path": str(self.config.faiss_index_path),
        }

    def format_for_rag(self, memories: list[RetrievedMemory]) -> str:
        """Format retrieved memories into a compact string for LLM injection."""
        if not memories:
            return ""
        parts = []
        for i, m in enumerate(memories, 1):
            date_str = m.record.created_at.strftime("%Y-%m-%d")
            parts.append(
                f"[MEMORY {i} | {m.record.memory_type} | "
                f"importance={m.record.importance_score:.1f} | {date_str}]\n"
                f"{m.record.summary}"
            )
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────
    # DECAY
    # ─────────────────────────────────────────────────────────────

    def run_decay(self, reference_time: Optional[datetime] = None) -> DecayResult:
        """Apply time-based importance decay to all eligible memories."""
        t_start = time.monotonic()
        records = self._store.get_all_for_decay()

        if not records:
            return DecayResult(0, 0, 0, 0, 0.0)

        now = reference_time or datetime.utcnow()
        decay_results = self._decay.batch_compute(records, reference_time=now)

        updates = [
            (r["new_score"], r["id"])
            for r in decay_results
            if r["changed"]
        ]
        if updates:
            self._store.bulk_update_importance(updates)

        skipped_exempt = sum(
            1 for rec in records
            if rec["memory_type"] in self._decay.config.exempt_types
        )
        below_floor = sum(1 for r in decay_results if r["below_floor"])
        elapsed_ms = (time.monotonic() - t_start) * 1000

        result = DecayResult(
            processed=len(records),
            updated=len(updates),
            below_floor=below_floor,
            skipped_exempt=skipped_exempt,
            duration_ms=elapsed_ms,
        )
        logger.info(
            f"Decay run — processed={result.processed}, updated={result.updated}, "
            f"below_floor={result.below_floor}, took={result.duration_ms:.1f}ms"
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # CONSISTENCY CHECK
    # ─────────────────────────────────────────────────────────────

    def check_consistency(self) -> ConsistencyReport:
        return self._checker.check()

    def repair_consistency(self, report: Optional[ConsistencyReport] = None) -> int:
        if report is None:
            report = self.check_consistency()
        if report.in_sync:
            return 0
        repaired = 0
        for orphan_id in report.orphaned_in_db:
            self._store.delete(orphan_id)
            repaired += 1
        for ghost_uuid in report.orphaned_in_faiss:
            self._index.remove(ghost_uuid)
            repaired += 1
        logger.info(f"Consistency repair — {repaired} entries fixed")
        return repaired

    # ─────────────────────────────────────────────────────────────
    # PURGE
    # ─────────────────────────────────────────────────────────────

    def purge_if_needed(self, dry_run: bool = False) -> PurgePlan:
        """Purge low-value memories if max_episodic_entries exceeded."""
        plan = self._purge.build_plan(self._store)
        if not plan.triggered or dry_run:
            logger.info(plan.summary())
            return plan
        for candidate in plan.to_delete:
            self._store.delete(candidate.id)
            self._index.remove(candidate.id)
        logger.warning(
            f"Purge executed — deleted {len(plan.to_delete)} memories. "
            f"New count: {self._store.count()}"
        )
        return plan