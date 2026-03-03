"""
A.C.C.E.S.S. — Memory Loop Test Suite

TestMemoryLoopReports       — report dataclass contracts
TestMemoryLoopDecay         — decay fallback and run
TestMemoryLoopConsolidate   — consolidation tagging
TestMemoryLoopTopics        — topic detection
TestMemoryLoopRepetition    — importance boost
TestMemoryLoopErrors        — failures produce reports, not exceptions
TestAgentCoreMemoryLoop     — run_memory_maintenance() integration
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional
from agent.memory_loop import MemoryLoop, MaintenanceReport, DecayReport, ConsolidationReport, TopicReport
from agent.agent_core import AgentCore
from agent.agent_config import AgentConfig
from agent.llm_client import FakeLLMClient


# ─── Fake Memory Manager with store ──────────────────────────────────────────

@dataclass
class FakeRecord:
    id: str; content: str = "content"; summary: str = "sum"
    importance_score: float = 0.5; tags: list = field(default_factory=list)
    memory_type: str = "episodic"; source: str = "interaction"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0


class FakeStore:
    def __init__(self, records=None):
        self._records = {r.id: r for r in (records or [])}
        self.updated_importances = []
        self.added_tags = {}

    def get_all_ids(self): return list(self._records.keys())
    def get_by_ids(self, ids): return [self._records[i] for i in ids if i in self._records]
    def count(self): return len(self._records)

    def get_purgeable_candidates(self, exclude_types, max_importance):
        return [
            {"id": r.id, "importance_score": r.importance_score,
             "created_at": r.created_at, "last_accessed_at": r.last_accessed_at,
             "access_count": r.access_count, "memory_type": r.memory_type}
            for r in self._records.values()
            if r.memory_type not in exclude_types and r.importance_score < max_importance
        ]

    def bulk_update_importance(self, updates):
        for score, rid in updates:
            if rid in self._records:
                self._records[rid].importance_score = score
                self.updated_importances.append((score, rid))

    def add_tags(self, rid, new_tags):
        if rid not in self.added_tags: self.added_tags[rid] = []
        self.added_tags[rid].extend(new_tags)
        if rid in self._records:
            self._records[rid].tags = list(set(self._records[rid].tags + new_tags))


class FakeDecayResult:
    processed = 5; updated = 3; below_floor = 1


class FakeMemoryManager:
    def __init__(self, records=None):
        self._store = FakeStore(records or [])
        self.decay_called = False

    def run_decay(self):
        self.decay_called = True
        return FakeDecayResult()

    def add_memory(self, **kw): pass
    def retrieve_relevant_memories(self, **kw): return []
    def format_for_rag(self, m): return ""


# ─── Agent fakes ─────────────────────────────────────────────────────────────

@dataclass
class FakeState:
    primary_emotion: str = "neutral"; intensity: float = 0.5; label: str = "neutral"
    is_positive: bool = False; is_negative: bool = False; pad: object = None
    is_high_arousal: bool = False
    def __post_init__(self):
        @dataclass
        class P: valence: float = 0.0; arousal: float = 0.5; dominance: float = 0.5
        if self.pad is None: self.pad = P()

@dataclass
class FakeModulation:
    tone: str = "neutral"; pacing: str = "normal"; verbosity: str = "normal"
    structure_bias: str = "conversational"; emotional_validation: bool = False
    motivational_bias: float = 0.0; cognitive_load_limit: float = 1.0; active_strategies: tuple = ()

@dataclass
class FakeBuiltPrompt:
    sections: tuple = ()
    def to_api_messages(self): return [{"role":"system","content":"S"},{"role":"user","content":"u"}]

class FakeEngine:
    def __init__(self): self.protection_calls = []
    def process_interaction(self, t, session_id=None): return FakeState()
    def emotional_trend(self): return {}
    def dominant_pattern(self, last_n=10): return None
    def apply_emotional_protection(self, s): self.protection_calls.append(s)
    def stats(self): return {}

class FakeMod2:
    def build_modulation(self, state, trend, dominant_pattern=None): return FakeModulation()

class FakeBuilder2:
    def build(self, user_input, modulation, memory_context=None): return FakeBuiltPrompt()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryLoopReports:
    def test_decay_report_to_dict(self):
        r = DecayReport(memories_evaluated=10, memories_updated=5)
        d = r.to_dict()
        assert {"memories_evaluated", "memories_updated", "duration_ms"}.issubset(set(d.keys()))

    def test_consolidation_report_to_dict(self):
        from agent.memory_loop import ConsolidationReport
        r = ConsolidationReport(candidates_found=3, tagged_for_review=2)
        d = r.to_dict()
        assert "candidates_found" in d and "tagged_for_review" in d

    def test_maintenance_report_to_dict_has_ran_at(self):
        r = MaintenanceReport()
        d = r.to_dict()
        assert "ran_at" in d and "total_duration_ms" in d

    def test_maintenance_report_ran_at_is_utc(self):
        r = MaintenanceReport()
        assert r.ran_at.tzinfo is not None

    def test_maintenance_report_repr_contains_total(self):
        r = MaintenanceReport(total_duration_ms=42.5)
        assert "42" in repr(r)

    def test_maintenance_report_errors_default_empty(self):
        r = MaintenanceReport()
        assert r.errors == []


class TestMemoryLoopDecay:
    def test_decay_calls_memory_run_decay(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        report = loop.decay_old_memories()
        assert mem.decay_called is True

    def test_decay_report_has_counts(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        report = loop.decay_old_memories()
        assert isinstance(report, DecayReport)
        assert report.memories_evaluated >= 0

    def test_decay_no_run_decay_method_does_not_crash(self):
        class NoDecayMem:
            _store = FakeStore()
        loop = MemoryLoop(NoDecayMem())
        report = loop.decay_old_memories()  # should not raise
        assert isinstance(report, DecayReport)

    def test_decay_duration_is_positive(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        report = loop.decay_old_memories()
        assert report.duration_ms >= 0.0


class TestMemoryLoopConsolidate:
    def test_consolidation_tags_low_importance_memories(self):
        records = [
            FakeRecord(id="a1", importance_score=0.1),
            FakeRecord(id="a2", importance_score=0.9),  # should not be tagged
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, consolidation_threshold=0.35)
        report = loop.consolidate_low_importance()
        assert report.tagged_for_review >= 1
        # Only a1 should be tagged (0.1 < 0.35)
        assert "a1" in mem._store.added_tags

    def test_consolidation_protects_high_importance(self):
        records = [FakeRecord(id="b1", importance_score=0.9)]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, consolidation_threshold=0.35)
        loop.consolidate_low_importance()
        assert "b1" not in mem._store.added_tags

    def test_consolidation_threshold_in_report(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem, consolidation_threshold=0.25)
        report = loop.consolidate_low_importance()
        assert report.threshold_used == 0.25

    def test_consolidation_duration_positive(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        report = loop.consolidate_low_importance()
        assert report.duration_ms >= 0.0

    def test_consolidation_no_store_does_not_crash(self):
        class NoStore: pass
        loop = MemoryLoop(NoStore())
        report = loop.consolidate_low_importance()  # should not raise
        assert isinstance(report, ConsolidationReport)


class TestMemoryLoopTopics:
    def test_detects_recurrent_tag(self):
        records = [
            FakeRecord(id="t1", tags=["frustration", "code"]),
            FakeRecord(id="t2", tags=["frustration", "work"]),
            FakeRecord(id="t3", tags=["frustration", "deadline"]),
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, recurrent_topic_min_count=3)
        report = loop.detect_recurrent_topics()
        assert "frustration" in report.recurrent_topics

    def test_non_recurrent_tag_not_in_recurrent(self):
        records = [
            FakeRecord(id="t1", tags=["frustration", "unique"]),
            FakeRecord(id="t2", tags=["frustration"]),
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, recurrent_topic_min_count=3)
        report = loop.detect_recurrent_topics()
        assert "unique" not in report.recurrent_topics

    def test_tag_counts_populated(self):
        records = [
            FakeRecord(id="t1", tags=["flow"]),
            FakeRecord(id="t2", tags=["flow"]),
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem)
        report = loop.detect_recurrent_topics()
        assert report.tag_counts.get("flow", 0) >= 2

    def test_empty_memory_returns_empty_topics(self):
        mem = FakeMemoryManager([])
        loop = MemoryLoop(mem)
        report = loop.detect_recurrent_topics()
        assert report.recurrent_topics == []
        assert report.tag_counts == {}


class TestMemoryLoopRepetition:
    def test_boosts_memories_with_recurrent_tags(self):
        records = [
            FakeRecord(id="r1", tags=["frustration"], importance_score=0.4),
            FakeRecord(id="r2", tags=["confidence"], importance_score=0.4),
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, repetition_boost=0.1, repetition_boost_cap=0.9)
        loop.adjust_importance_based_on_repetition(["frustration"])
        updated_ids = [u[1] for u in mem._store.updated_importances]
        assert "r1" in updated_ids
        assert "r2" not in updated_ids

    def test_boost_capped(self):
        records = [FakeRecord(id="r1", tags=["frustration"], importance_score=0.84)]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, repetition_boost=0.1, repetition_boost_cap=0.85)
        loop.adjust_importance_based_on_repetition(["frustration"])
        final_score = mem._store._records["r1"].importance_score
        assert final_score <= 0.85

    def test_empty_recurrent_tags_no_updates(self):
        records = [FakeRecord(id="r1", tags=["frustration"], importance_score=0.4)]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem)
        loop.adjust_importance_based_on_repetition([])
        assert len(mem._store.updated_importances) == 0

    def test_memories_boosted_count_correct(self):
        records = [
            FakeRecord(id="r1", tags=["flow"], importance_score=0.3),
            FakeRecord(id="r2", tags=["flow"], importance_score=0.3),
            FakeRecord(id="r3", tags=["doubt"], importance_score=0.3),
        ]
        mem = FakeMemoryManager(records)
        loop = MemoryLoop(mem, repetition_boost=0.1)
        report = loop.adjust_importance_based_on_repetition(["flow"])
        assert report.memories_boosted == 2


class TestMemoryLoopErrors:
    def test_run_all_returns_report_on_error(self):
        # Override the method on the loop instance to raise (simulates catastrophic failure)
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        def failing_decay(): raise RuntimeError("DB catastrophic failure")
        loop.decay_old_memories = failing_decay
        report = loop.run(run_decay=True, run_consolidation=False,
                          run_topics=False, run_repetition=False)
        assert isinstance(report, MaintenanceReport)
        assert len(report.errors) > 0

    def test_partial_failure_does_not_prevent_other_ops(self):
        # run_decay works, consolidation fails
        mem = FakeMemoryManager()
        class PartialLoop(MemoryLoop):
            def consolidate_low_importance(self):
                raise RuntimeError("forced")
        loop = PartialLoop(mem)
        report = loop.run(run_decay=True, run_consolidation=True,
                          run_topics=False, run_repetition=False)
        assert report.decay is not None      # decay succeeded
        assert "consolidation" in " ".join(report.errors)

    def test_total_duration_positive(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        report = loop.run()
        assert report.total_duration_ms >= 0.0


class TestAgentCoreMemoryLoop:
    def _make_agent_with_loop(self, loop=None, config_overrides=None):
        cfg = dict(enable_rag=False, apply_emotional_protection=False,
                   write_user_turn_to_memory=False)
        if config_overrides: cfg.update(config_overrides)
        config = AgentConfig(**cfg)
        return AgentCore(
            emotion_engine=FakeEngine(), conversation_modulator=FakeMod2(),
            prompt_builder=FakeBuilder2(), llm_client=FakeLLMClient(response="R."),
            config=config, memory_loop=loop,
        )

    def test_run_maintenance_returns_none_when_no_loop(self):
        agent = self._make_agent_with_loop(loop=None)
        result = agent.run_memory_maintenance()
        assert result is None

    def test_run_maintenance_returns_report_when_loop_present(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        agent = self._make_agent_with_loop(loop=loop)
        report = agent.run_memory_maintenance()
        assert isinstance(report, MaintenanceReport)

    def test_run_maintenance_skips_decay_when_disabled(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        config = AgentConfig(enable_rag=False, apply_emotional_protection=False,
                             write_user_turn_to_memory=False, memory_decay_enabled=False)
        agent = AgentCore(
            emotion_engine=FakeEngine(), conversation_modulator=FakeMod2(),
            prompt_builder=FakeBuilder2(), llm_client=FakeLLMClient(response="R."),
            config=config, memory_loop=loop,
        )
        agent.run_memory_maintenance(run_decay=True)
        assert mem.decay_called is False

    def test_run_maintenance_in_stats(self):
        mem = FakeMemoryManager()
        loop = MemoryLoop(mem)
        agent = self._make_agent_with_loop(loop=loop)
        stats = agent.stats()
        assert stats["memory_loop"] is True


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestMemoryLoopReports, TestMemoryLoopDecay, TestMemoryLoopConsolidate,
        TestMemoryLoopTopics, TestMemoryLoopRepetition, TestMemoryLoopErrors,
        TestAgentCoreMemoryLoop,
    ]
    passed, failed = [], []
    for cls in test_classes:
        instance = cls()
        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            label = f"{cls.__name__}.{name}"
            try:
                getattr(instance, name)(); passed.append(label); print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e)); print(f"  ❌ {label}: {e}"); traceback.print_exc()
    total = len(passed) + len(failed)
    print(f"\n{'='*60}\nResults: {len(passed)}/{total} passed")
    if failed:
        for l, e in failed: print(f"  {l}: {e}")
        sys.exit(1)
    else: print("All checks passed ✅")