"""
A.C.C.E.S.S. — Memory Layer: Unit Tests

Run: python -m pytest tests/ -v
Or:  python -m pytest tests/ -v --tb=short

Coverage:
  [CORE]
  - test_add_memory_basic
  - test_add_memory_defaults
  - test_add_memory_persists_to_db
  - test_add_memory_persists_to_faiss
  
  [RETRIEVE]
  - test_retrieve_returns_results
  - test_retrieve_relevance_ordering
  - test_retrieve_min_importance_filter
  - test_retrieve_type_filter
  - test_retrieve_empty_index
  - test_retrieve_marks_accessed
  
  [UPDATE IMPORTANCE]
  - test_update_importance_absolute
  - test_update_importance_delta_positive
  - test_update_importance_delta_negative
  - test_update_importance_clamp_max
  - test_update_importance_clamp_min
  - test_update_importance_no_args_raises

  [DECAY]
  - test_decay_reduces_score
  - test_decay_semantic_exempt
  - test_decay_protected_slower
  - test_decay_access_reinforcement
  - test_decay_floor_respected
  - test_run_decay_updates_db
  - test_run_decay_skips_zero_elapsed

  [CONSISTENCY]
  - test_consistency_in_sync
  - test_consistency_detects_db_orphan
  - test_repair_consistency_fixes_db_orphan

  [PURGE]
  - test_purge_not_triggered_below_limit
  - test_purge_triggered_above_limit
  - test_purge_dry_run_no_deletion
  - test_purge_protects_high_importance
  - test_purge_protects_semantic

  [PERSISTENCE RELOAD]
  - test_persistence_reload_db
  - test_persistence_reload_faiss_vectors
  - test_persistence_reload_retrieval_works
"""

import math
import shutil
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# ── We test against real MemoryManager with tmp dirs ──────────────────────
# No mocking of FAISS or SQLite — these are integration-style unit tests.
# The only mock is the embedder, replaced by a deterministic fake.

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.config import MemoryConfig
from memory.decay import DecayEngine, DecayConfig
from memory.models import MemoryRecord
from memory.store import MemoryStore
from memory.vector_index import VectorIndex


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def config(tmp_dir):
    cfg = MemoryConfig(data_dir=tmp_dir / "memory")
    # FakeEmbedder produces random unit vectors → cosine similarities near 0.
    # Lower threshold to 0.0 so FAISS results aren't silently dropped in tests.
    # Production config keeps 0.25 (real embeddings have genuine similarity).
    cfg.min_similarity_score = 0.0
    return cfg


@pytest.fixture
def manager(config, monkeypatch):
    """MemoryManager with a fake embedder (no model download required)."""
    from memory.memory_manager import MemoryManager
    import numpy as np

    class FakeEmbedder:
        """Deterministic fake: encodes text as a reproducible unit vector."""
        def encode(self, text, normalize_embeddings=True):
            # Hash the text to a seed → reproducible vector
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                vec = vec / norm if norm > 0 else vec
            return vec

    m = MemoryManager.__new__(MemoryManager)
    m.config = config
    from memory.store import MemoryStore
    from memory.vector_index import VectorIndex
    from memory.decay import DecayEngine, DecayConfig
    from memory.maintenance import ConsistencyChecker, PurgePolicy, PurgeStrategy

    m._store = MemoryStore(config)
    m._index = VectorIndex(config)
    m._embedder = FakeEmbedder()
    m._decay = DecayEngine(DecayConfig(half_life_hours=24.0, decay_floor=0.1))
    m._checker = ConsistencyChecker(m._store, m._index)
    m._purge = PurgePolicy(max_entries=config.max_episodic_entries, strategy=PurgeStrategy.COMBINED)

    return m


def make_mem(manager, content="Test content", **kwargs):
    """Helper to add a memory with minimal boilerplate."""
    return manager.add_memory(content=content, **kwargs)


# ─────────────────────────────────────────────────────────────
# CORE — ADD MEMORY
# ─────────────────────────────────────────────────────────────

class TestAddMemory:

    def test_add_memory_basic(self, manager):
        m = make_mem(manager, content="User completed the task.", importance_score=0.8)
        assert m.id is not None
        assert m.content == "User completed the task."
        assert m.importance_score == 0.8

    def test_add_memory_defaults(self, manager):
        m = make_mem(manager, content="Something happened.")
        assert m.memory_type == "episodic"
        assert m.source == "interaction"
        assert m.tags == []
        assert m.importance_score == manager.config.default_importance

    def test_add_memory_persists_to_db(self, manager):
        m = make_mem(manager, content="Persisted content.")
        retrieved = manager.get_memory(m.id)
        assert retrieved is not None
        assert retrieved.content == "Persisted content."

    def test_add_memory_persists_to_faiss(self, manager):
        assert manager._index.total_vectors == 0
        make_mem(manager)
        assert manager._index.total_vectors == 1

    def test_add_multiple_memories(self, manager):
        for i in range(5):
            make_mem(manager, content=f"Memory number {i}")
        assert manager._store.count() == 5
        assert manager._index.total_vectors == 5

    def test_add_memory_summary_default(self, manager):
        long_content = "x" * 500
        m = make_mem(manager, content=long_content)
        assert m.summary == long_content[:300]

    def test_add_memory_custom_summary(self, manager):
        m = make_mem(manager, content="Long content here.", summary="Short summary.")
        assert m.summary == "Short summary."


# ─────────────────────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────────────────────

class TestRetrieve:

    def test_retrieve_returns_results(self, manager):
        # Use exact content as query → similarity=1.0 with FakeEmbedder (same hash seed)
        make_mem(manager, content="User was very productive today.", importance_score=0.8)
        make_mem(manager, content="User ate lunch.", importance_score=0.3)
        results = manager.retrieve_relevant_memories("User was very productive today.", top_k=2)
        assert len(results) > 0

    def test_retrieve_empty_index(self, manager):
        results = manager.retrieve_relevant_memories("anything")
        assert results == []

    def test_retrieve_min_importance_filter(self, manager):
        make_mem(manager, content="Low importance event.", importance_score=0.1)
        make_mem(manager, content="High importance insight.", importance_score=0.9)
        results = manager.retrieve_relevant_memories(
            "event insight", top_k=5, min_importance=0.5
        )
        # All returned memories should be above min_importance
        for r in results:
            assert r.record.importance_score >= 0.5

    def test_retrieve_type_filter(self, manager):
        make_mem(manager, content="Semantic fact about Python.", memory_type="semantic")
        make_mem(manager, content="Episodic event about Python.", memory_type="episodic")
        results = manager.retrieve_relevant_memories(
            "Python", top_k=5, memory_type="semantic"
        )
        for r in results:
            assert r.record.memory_type == "semantic"

    def test_retrieve_marks_accessed(self, manager):
        m = make_mem(manager, content="Some memory.")
        assert m.access_count == 0
        # Query must match content exactly → same hash → similarity=1.0
        manager.retrieve_relevant_memories("Some memory.")
        reloaded = manager.get_memory(m.id)
        assert reloaded.access_count == 1

    def test_retrieve_relevance_has_score(self, manager):
        make_mem(manager, content="Important user preference.", importance_score=0.9)
        results = manager.retrieve_relevant_memories("user preference")
        if results:
            assert results[0].similarity > 0
            assert results[0].relevance > 0
            assert results[0].relevance == pytest.approx(
                results[0].similarity * results[0].record.importance_score, abs=1e-5
            )

    def test_retrieve_format_for_rag(self, manager):
        make_mem(manager, content="User prefers dark mode.", summary="User likes dark mode.")
        results = manager.retrieve_relevant_memories("user preferences")
        rag = manager.format_for_rag(results)
        if results:
            assert "[MEMORY 1" in rag
            assert "episodic" in rag


# ─────────────────────────────────────────────────────────────
# UPDATE IMPORTANCE
# ─────────────────────────────────────────────────────────────

class TestUpdateImportance:

    def test_update_importance_absolute(self, manager):
        m = make_mem(manager, importance_score=0.5)
        final = manager.update_importance(m.id, new_score=0.9)
        assert final == pytest.approx(0.9)
        reloaded = manager.get_memory(m.id)
        assert reloaded.importance_score == pytest.approx(0.9)

    def test_update_importance_delta_positive(self, manager):
        m = make_mem(manager, importance_score=0.5)
        final = manager.update_importance(m.id, delta=+0.2)
        assert final == pytest.approx(0.7)

    def test_update_importance_delta_negative(self, manager):
        m = make_mem(manager, importance_score=0.5)
        final = manager.update_importance(m.id, delta=-0.2)
        assert final == pytest.approx(0.3)

    def test_update_importance_clamp_max(self, manager):
        m = make_mem(manager, importance_score=0.9)
        final = manager.update_importance(m.id, delta=+0.5)
        assert final == pytest.approx(1.0)

    def test_update_importance_clamp_min(self, manager):
        m = make_mem(manager, importance_score=0.1)
        final = manager.update_importance(m.id, delta=-0.5)
        assert final == pytest.approx(0.0)

    def test_update_importance_no_args_raises(self, manager):
        m = make_mem(manager)
        with pytest.raises(ValueError, match="Provide either"):
            manager.update_importance(m.id)


# ─────────────────────────────────────────────────────────────
# DECAY ENGINE (pure unit tests — no DB)
# ─────────────────────────────────────────────────────────────

class TestDecayEngine:

    @pytest.fixture
    def engine(self):
        return DecayEngine(DecayConfig(
            half_life_hours=24.0,
            protection_threshold=0.8,
            access_reinforcement_hours=12.0,
            decay_floor=0.05,
            absolute_minimum=0.05,
            exempt_types=("semantic",),
        ))

    def test_decay_reduces_score(self, engine):
        now = datetime.utcnow()
        past = now - timedelta(hours=48)
        new_score = engine.compute_decay(
            current_importance=0.8,
            memory_type="episodic",
            created_at=past,
            last_accessed_at=None,
            access_count=0,
            reference_time=now,
        )
        assert new_score < 0.8

    def test_decay_semantic_exempt(self, engine):
        now = datetime.utcnow()
        past = now - timedelta(hours=240)  # 10 days
        score = engine.compute_decay(
            current_importance=0.7,
            memory_type="semantic",
            created_at=past,
            last_accessed_at=None,
            access_count=0,
            reference_time=now,
        )
        assert score == pytest.approx(0.7)  # unchanged

    def test_decay_protected_slower(self, engine):
        """A high-importance memory should decay slower than a low-importance one."""
        now = datetime.utcnow()
        past = now - timedelta(hours=48)

        score_high = engine.compute_decay(0.9, "episodic", past, None, 0, now)
        score_low = engine.compute_decay(0.4, "episodic", past, None, 0, now)

        # High importance gets 2x half-life, so loses less % of value
        ratio_high = score_high / 0.9
        ratio_low = score_low / 0.4
        assert ratio_high > ratio_low

    def test_decay_access_reinforcement(self, engine):
        """Frequently accessed memory should decay slower."""
        now = datetime.utcnow()
        past = now - timedelta(hours=48)

        score_no_access = engine.compute_decay(0.5, "episodic", past, None, 0, now)
        score_with_access = engine.compute_decay(0.5, "episodic", past, None, 10, now)

        assert score_with_access > score_no_access

    def test_decay_floor_respected(self, engine):
        """Score should never go below absolute_minimum."""
        now = datetime.utcnow()
        ancient = now - timedelta(days=365)
        score = engine.compute_decay(0.1, "episodic", ancient, None, 0, now)
        assert score >= engine.config.absolute_minimum

    def test_decay_zero_elapsed(self, engine):
        """Decay with 0 time elapsed should return unchanged score."""
        now = datetime.utcnow()
        score = engine.compute_decay(0.6, "episodic", now, now, 0, now)
        assert score == pytest.approx(0.6)


class TestRunDecay:

    def test_run_decay_updates_db(self, manager):
        """After decay, old unaccessed memories should have lower scores."""
        m = manager.add_memory(
            content="Old forgotten event.",
            importance_score=0.6,
            memory_type="episodic",
        )
        # Manually set created_at to past in DB
        with manager._store._get_conn() as conn:
            old_time = (datetime.utcnow() - timedelta(hours=96)).isoformat()
            conn.execute(
                "UPDATE memories SET created_at = ?, last_accessed_at = NULL WHERE id = ?",
                (old_time, m.id)
            )

        result = manager.run_decay()
        assert result.processed >= 1
        assert result.updated >= 1

        reloaded = manager.get_memory(m.id)
        assert reloaded.importance_score < 0.6

    def test_run_decay_skips_semantic(self, manager):
        """Semantic memories should not be decayed."""
        m = manager.add_memory(
            content="Python is a programming language.",
            importance_score=0.7,
            memory_type="semantic",
        )
        with manager._store._get_conn() as conn:
            old_time = (datetime.utcnow() - timedelta(hours=200)).isoformat()
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                (old_time, m.id)
            )

        result = manager.run_decay()
        assert result.skipped_exempt >= 1
        reloaded = manager.get_memory(m.id)
        assert reloaded.importance_score == pytest.approx(0.7)

    def test_run_decay_empty_db(self, manager):
        result = manager.run_decay()
        assert result.processed == 0
        assert result.updated == 0


# ─────────────────────────────────────────────────────────────
# CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────

class TestConsistency:

    def test_consistency_in_sync(self, manager):
        make_mem(manager, content="Memory A")
        make_mem(manager, content="Memory B")
        report = manager.check_consistency()
        assert report.in_sync
        assert report.db_count == 2
        assert report.faiss_count == 2
        assert report.orphaned_in_db == []
        assert report.orphaned_in_faiss == []

    def test_consistency_empty_is_sync(self, manager):
        report = manager.check_consistency()
        assert report.in_sync
        assert report.db_count == 0

    def test_consistency_detects_db_orphan(self, manager):
        """Insert into DB without corresponding FAISS vector → orphan detected."""
        from memory.models import MemoryRecord
        orphan = MemoryRecord(content="Ghost in DB", importance_score=0.5)
        manager._store.insert(orphan)
        # Do NOT add to FAISS

        report = manager.check_consistency()
        assert not report.in_sync
        assert orphan.id in report.orphaned_in_db

    def test_repair_consistency_fixes_db_orphan(self, manager):
        from memory.models import MemoryRecord
        orphan = MemoryRecord(content="Ghost in DB", importance_score=0.5)
        manager._store.insert(orphan)

        report = manager.check_consistency()
        assert not report.in_sync

        repaired = manager.repair_consistency(report)
        assert repaired == 1

        # DB should no longer have the orphan
        assert manager.get_memory(orphan.id) is None

        # Now should be in sync
        new_report = manager.check_consistency()
        assert new_report.in_sync


# ─────────────────────────────────────────────────────────────
# PURGE POLICY
# ─────────────────────────────────────────────────────────────

class TestPurge:

    @pytest.fixture
    def small_config(self, tmp_dir):
        cfg = MemoryConfig(data_dir=tmp_dir / "purge_memory")
        cfg.max_episodic_entries = 5
        return cfg

    @pytest.fixture
    def small_manager(self, small_config, monkeypatch):
        from memory.memory_manager import MemoryManager
        from memory.maintenance import PurgePolicy, PurgeStrategy
        import numpy as np

        class FakeEmbedder:
            def encode(self, text, normalize_embeddings=True):
                seed = hash(text) % (2**31)
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    vec = vec / norm if norm > 0 else vec
                return vec

        from memory.store import MemoryStore
        from memory.vector_index import VectorIndex
        from memory.decay import DecayEngine, DecayConfig
        from memory.maintenance import ConsistencyChecker

        m = MemoryManager.__new__(MemoryManager)
        m.config = small_config
        m._store = MemoryStore(small_config)
        m._index = VectorIndex(small_config)
        m._embedder = FakeEmbedder()
        m._decay = DecayEngine(DecayConfig())
        m._checker = ConsistencyChecker(m._store, m._index)
        m._purge = PurgePolicy(max_entries=5, strategy=PurgeStrategy.COMBINED)
        return m

    def test_purge_not_triggered_below_limit(self, small_manager):
        for i in range(3):
            make_mem(small_manager, content=f"Memory {i}", importance_score=0.3)
        plan = small_manager.purge_if_needed(dry_run=True)
        assert not plan.triggered
        assert plan.to_delete == []

    def test_purge_triggered_above_limit(self, small_manager):
        for i in range(8):
            make_mem(small_manager, content=f"Memory {i}", importance_score=0.3)
        plan = small_manager.purge_if_needed(dry_run=True)
        assert plan.triggered
        assert len(plan.to_delete) > 0

    def test_purge_dry_run_no_deletion(self, small_manager):
        for i in range(8):
            make_mem(small_manager, content=f"Memory {i}", importance_score=0.3)
        count_before = small_manager._store.count()
        plan = small_manager.purge_if_needed(dry_run=True)
        count_after = small_manager._store.count()
        assert count_before == count_after  # dry run → no deletion

    def test_purge_actually_deletes(self, small_manager):
        for i in range(8):
            make_mem(small_manager, content=f"Memory {i}", importance_score=0.3)
        count_before = small_manager._store.count()
        plan = small_manager.purge_if_needed(dry_run=False)
        count_after = small_manager._store.count()
        assert count_after < count_before

    def test_purge_protects_high_importance(self, small_manager):
        """High importance memories should survive purge."""
        protected = make_mem(small_manager, content="Critical memory.", importance_score=0.95)
        for i in range(8):
            make_mem(small_manager, content=f"Filler memory {i}", importance_score=0.2)

        small_manager.purge_if_needed(dry_run=False)
        assert small_manager.get_memory(protected.id) is not None

    def test_purge_protects_semantic(self, small_manager):
        """Semantic memories should never be purged."""
        fact = make_mem(
            small_manager,
            content="Python was created by Guido.",
            memory_type="semantic",
            importance_score=0.3,
        )
        for i in range(8):
            make_mem(small_manager, content=f"Filler {i}", importance_score=0.2)

        small_manager.purge_if_needed(dry_run=False)
        assert small_manager.get_memory(fact.id) is not None


# ─────────────────────────────────────────────────────────────
# PERSISTENCE RELOAD
# ─────────────────────────────────────────────────────────────

class TestPersistenceReload:
    """
    Simulate a full process restart: create manager, add data, destroy instance,
    create new manager from same directory → verify data survives.
    """

    def _make_manager(self, config):
        """Helper to instantiate a manager with fake embedder."""
        from memory.memory_manager import MemoryManager
        from memory.maintenance import ConsistencyChecker, PurgePolicy, PurgeStrategy
        from memory.store import MemoryStore
        from memory.vector_index import VectorIndex
        from memory.decay import DecayEngine, DecayConfig
        import numpy as np

        class FakeEmbedder:
            def encode(self, text, normalize_embeddings=True):
                seed = hash(text) % (2**31)
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    vec = vec / norm if norm > 0 else vec
                return vec

        m = MemoryManager.__new__(MemoryManager)
        m.config = config
        m._store = MemoryStore(config)
        m._index = VectorIndex(config)
        m._embedder = FakeEmbedder()
        m._decay = DecayEngine(DecayConfig())
        m._checker = ConsistencyChecker(m._store, m._index)
        m._purge = PurgePolicy(max_entries=config.max_episodic_entries)
        return m

    def test_persistence_reload_db(self, config):
        """Records inserted in session 1 are readable in session 2."""
        m1 = self._make_manager(config)
        mem = m1.add_memory("Persistent memory.", importance_score=0.75)
        memory_id = mem.id
        del m1  # simulate process restart

        m2 = self._make_manager(config)
        reloaded = m2.get_memory(memory_id)
        assert reloaded is not None
        assert reloaded.content == "Persistent memory."
        assert reloaded.importance_score == pytest.approx(0.75)

    def test_persistence_reload_faiss_vectors(self, config):
        """FAISS index survives restart — vector count matches."""
        m1 = self._make_manager(config)
        m1.add_memory("Memory Alpha.")
        m1.add_memory("Memory Beta.")
        count_before = m1._index.total_vectors
        del m1

        m2 = self._make_manager(config)
        assert m2._index.total_vectors == count_before

    def test_persistence_reload_retrieval_works(self, config):
        """Retrieval still works after restart (vectors in FAISS, meta in DB)."""
        m1 = self._make_manager(config)
        m1.add_memory("User prefers working at night.", importance_score=0.8)
        del m1

        m2 = self._make_manager(config)
        # Query must match stored content exactly → same hash → similarity=1.0
        results = m2.retrieve_relevant_memories("User prefers working at night.")
        assert len(results) > 0
        assert "night" in results[0].record.content.lower()

    def test_persistence_reload_consistency(self, config):
        """After reload, DB and FAISS should still be in sync."""
        m1 = self._make_manager(config)
        for i in range(3):
            m1.add_memory(f"Memory {i}")
        del m1

        m2 = self._make_manager(config)
        report = m2.check_consistency()
        assert report.in_sync