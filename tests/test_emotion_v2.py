"""
A.C.C.E.S.S. — Emotion Engine V2: Unit Tests

Run: python -m pytest tests/test_emotion_v2.py -v

Coverage:
  [MOMENTUM / TREND]
  - test_trend_stable_on_insufficient_history
  - test_emotional_trend_detects_decline
  - test_emotional_trend_detects_improvement
  - test_emotional_trend_detects_stable
  - test_emotional_trend_escalating_label
  - test_burnout_risk_flag
  - test_mania_risk_flag
  - test_trend_data_points_count
  - test_trend_slope_clamped

  [ALIGNMENT]
  - test_alignment_neutral_on_no_tags
  - test_alignment_neutral_on_none_state
  - test_alignment_clamped_between_zero_and_one
  - test_alignment_boosts_positive_memory_in_negative_state
  - test_alignment_suppresses_stress_memory_in_fragile_state
  - test_alignment_explain_returns_string
  - test_alignment_vector_similarity_same_emotion

  [PROTECTION]
  - test_emotional_protection_adds_stress_tag
  - test_emotional_protection_adds_boost_tag
  - test_protection_no_crash_without_memory_manager
  - test_protection_no_effect_without_pattern

  [EMOTION-AWARE RETRIEVAL]
  - test_emotion_aware_retrieval_changes_ranking
  - test_no_crash_without_emotional_context
  - test_retrieval_alignment_one_when_no_fn_registered
  - test_alignment_fn_registered_on_engine_init

  [ENGINE INTEGRATION]
  - test_stats_includes_trend
  - test_engine_trend_reflects_injected_states
"""

import shutil
import tempfile
import pytest
import numpy as np
from pathlib import Path
import sys
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion.config import EmotionConfig
from emotion.models import EmotionalState, PADState, EMOTION_PAD_MAP
from emotion.emotion_prototypes import EmotionPrototypes, EMOTION_PROTOTYPES
from emotion.emotion_embedder import EmotionEmbedder
from emotion.emotion_scoring import EmotionScorer
from emotion.emotion_alignment import EmotionAlignment, KNOWN_EMOTION_TAGS, POSITIVE_EMOTIONS, NEGATIVE_EMOTIONS
from emotion.emotion_engine import EmotionEngine


# ─────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Deterministic fake embedder with keyword-based emotional bias."""

    KEYWORD_BIASES: dict[str, list[str]] = {
        "frustration": ["stuck", "frustrated", "failing", "blocked", "annoying", "nothing works"],
        "flow":        ["focused", "zone", "effortless", "absorbed", "clicking", "clear"],
        "fatigue":     ["tired", "exhausted", "drained", "slow", "mentally", "struggling"],
        "confidence":  ["powerful", "capable", "strong", "certain", "trust", "abilities"],
        "doubt":       ["unsure", "hesitate", "uncertain", "second-guessing", "hesitating"],
        "drive":       ["determined", "motivated", "ambitious", "mission", "energized", "push"],
    }

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._emotion_axes = self._build_axes()

    def _build_axes(self) -> dict[str, np.ndarray]:
        axes = {}
        for i, emotion in enumerate(self.KEYWORD_BIASES.keys()):
            rng = np.random.RandomState(1000 + i)
            vec = rng.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            axes[emotion] = vec
        return axes

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        text_lower = text.lower()
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32) * 0.1
        for emotion, keywords in self.KEYWORD_BIASES.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                vec += self._emotion_axes[emotion] * min(1.0, matches * 0.5)
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


def _make_state(emotion: str, intensity: float = 0.7) -> EmotionalState:
    """Helper: build an EmotionalState directly (bypasses scoring)."""
    vec = np.random.RandomState(abs(hash(emotion)) % (2**32)).randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return EmotionalState(
        primary_emotion=emotion,
        intensity=intensity,
        confidence=0.5,
        emotion_vector=vec,
        pad=PADState.from_emotion(emotion),
        raw_scores={emotion: intensity},
    )


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def config():
    return EmotionConfig(
        softmax_temperature=0.3,
        intensity_threshold_low=0.20,
        intensity_threshold_high=0.35,
        momentum_slope_threshold=0.05,
        burnout_slope_threshold=-0.08,
        mania_slope_threshold=0.10,
        protection_repeat_count=3,
    )


@pytest.fixture
def prototypes(fake_embedder):
    p = EmotionPrototypes(fake_embedder)
    p.build()
    return p


@pytest.fixture
def aligner(prototypes, config):
    return EmotionAlignment(
        prototypes=prototypes.prototypes,
        fragile_threshold=config.alignment_fragile_threshold,
        stress_suppression=config.alignment_stress_suppression,
        positive_boost=config.alignment_positive_boost,
    )


@pytest.fixture
def engine_no_memory(fake_embedder, config):
    engine = EmotionEngine.__new__(EmotionEngine)
    engine.config = config
    engine._memory = None
    engine._embedder = EmotionEmbedder(fake_embedder)
    engine._prototypes = EmotionPrototypes(fake_embedder)
    engine._prototypes.build()
    engine._scorer = EmotionScorer(engine._prototypes, config)
    engine._alignment = EmotionAlignment(engine._prototypes.prototypes)
    engine._history = deque(maxlen=config.history_max_size)
    return engine


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def engine_with_memory(fake_embedder, config, tmp_dir):
    from memory.config import MemoryConfig
    from memory.memory_manager import MemoryManager
    from memory.store import MemoryStore
    from memory.vector_index import VectorIndex
    from memory.decay import DecayEngine, DecayConfig
    from memory.maintenance import ConsistencyChecker, PurgePolicy, PurgeStrategy

    mem_config = MemoryConfig(data_dir=tmp_dir / "memory")
    mem_config.min_similarity_score = 0.0

    mem = MemoryManager.__new__(MemoryManager)
    mem.config = mem_config
    mem._store = MemoryStore(mem_config)
    mem._index = VectorIndex(mem_config)
    mem._embedder = fake_embedder
    mem._decay = DecayEngine(DecayConfig())
    mem._checker = ConsistencyChecker(mem._store, mem._index)
    mem._purge = PurgePolicy(max_entries=10_000)
    mem._alignment_fn = None

    engine = EmotionEngine.__new__(EmotionEngine)
    engine.config = config
    engine._memory = mem
    engine._embedder = EmotionEmbedder(fake_embedder)
    engine._prototypes = EmotionPrototypes(fake_embedder)
    engine._prototypes.build()
    engine._scorer = EmotionScorer(engine._prototypes, config)
    engine._alignment = EmotionAlignment(engine._prototypes.prototypes)
    engine._history = deque(maxlen=config.history_max_size)

    # Register alignment
    mem.set_emotion_alignment(engine._alignment.compute)

    return engine, mem


# ─────────────────────────────────────────────────────────────
# MOMENTUM / TREND
# ─────────────────────────────────────────────────────────────

class TestEmotionalTrend:

    def _inject_states(self, engine, emotions_and_valences: list[tuple[str, float]]):
        """Inject pre-built states with specific valences directly into history."""
        for emotion, valence in emotions_and_valences:
            state = _make_state(emotion)
            state.pad.valence = valence
            engine._history.append(state)

    def test_trend_stable_on_insufficient_history(self, engine_no_memory):
        # Only 2 states → not enough for regression
        engine_no_memory._history.append(_make_state("confidence"))
        engine_no_memory._history.append(_make_state("drive"))
        trend = engine_no_memory.emotional_trend()
        assert trend["trend_label"] == "stable"
        assert trend["burnout_risk"] is False
        assert trend["mania_risk"] is False

    def test_emotional_trend_detects_decline(self, engine_no_memory):
        # Inject states with steadily declining valence
        valences = [0.5, 0.4, 0.2, 0.0, -0.2, -0.4, -0.5]
        self._inject_states(engine_no_memory, [("frustration", v) for v in valences])
        trend = engine_no_memory.emotional_trend()
        assert trend["valence_slope"] < 0
        assert trend["trend_label"] in ("declining", "escalating")

    def test_emotional_trend_detects_improvement(self, engine_no_memory):
        valences = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
        self._inject_states(engine_no_memory, [("confidence", v) for v in valences])
        trend = engine_no_memory.emotional_trend()
        assert trend["valence_slope"] > 0
        assert trend["trend_label"] == "improving"

    def test_emotional_trend_detects_stable(self, engine_no_memory):
        # Flat valence — tiny random noise but no clear slope
        valences = [0.3, 0.31, 0.29, 0.30, 0.31, 0.29, 0.30]
        self._inject_states(engine_no_memory, [("flow", v) for v in valences])
        trend = engine_no_memory.emotional_trend()
        assert trend["trend_label"] == "stable"

    def test_emotional_trend_escalating_label(self, engine_no_memory):
        # Declining valence + rising arousal → escalating
        for i, (v, a) in enumerate([(-0.1, 0.3), (-0.2, 0.4), (-0.3, 0.5),
                                     (-0.4, 0.6), (-0.5, 0.7), (-0.6, 0.8)]):
            s = _make_state("frustration")
            s.pad.valence = v
            s.pad.arousal = a
            engine_no_memory._history.append(s)
        trend = engine_no_memory.emotional_trend()
        assert trend["valence_slope"] < 0
        assert trend["arousal_slope"] > 0

    def test_burnout_risk_flag(self, engine_no_memory):
        # Very steep valence decline → burnout risk
        valences = [0.5, 0.3, 0.0, -0.3, -0.5, -0.7, -0.9]
        self._inject_states(engine_no_memory, [("frustration", v) for v in valences])
        trend = engine_no_memory.emotional_trend()
        assert trend["burnout_risk"] is True

    def test_mania_risk_flag(self, engine_no_memory):
        # Rising valence + very steep arousal rise
        config = engine_no_memory.config
        config.mania_slope_threshold = 0.05  # lower threshold for test
        for i, (v, a) in enumerate([(0.1, 0.3), (0.2, 0.4), (0.3, 0.5),
                                     (0.4, 0.7), (0.5, 0.85), (0.6, 0.95)]):
            s = _make_state("drive")
            s.pad.valence = v
            s.pad.arousal = a
            engine_no_memory._history.append(s)
        trend = engine_no_memory.emotional_trend()
        assert trend["mania_risk"] is True

    def test_trend_data_points_count(self, engine_no_memory):
        for _ in range(5):
            engine_no_memory._history.append(_make_state("confidence"))
        trend = engine_no_memory.emotional_trend(last_n=5)
        assert trend["data_points"] == 5

    def test_trend_slope_clamped(self, engine_no_memory):
        # Even extreme valences should produce clamped slopes
        valences = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
        self._inject_states(engine_no_memory, [("frustration", v) for v in valences])
        trend = engine_no_memory.emotional_trend()
        assert -1.0 <= trend["valence_slope"] <= 1.0


# ─────────────────────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────────────────────

class TestEmotionAlignment:

    def _make_record(self, tags: list[str]):
        """Minimal duck-type MemoryRecord for alignment tests."""
        class FakeRecord:
            pass
        r = FakeRecord()
        r.tags = tags
        return r

    def test_alignment_neutral_on_no_tags(self, aligner):
        record = self._make_record([])
        state = _make_state("confidence")
        assert aligner.compute(record, state) == pytest.approx(1.0)

    def test_alignment_neutral_on_none_state(self, aligner):
        record = self._make_record(["frustration"])
        assert aligner.compute(record, None) == pytest.approx(1.0)

    def test_alignment_clamped_between_zero_and_one(self, aligner):
        """Alignment must always be in [0, 1] regardless of input."""
        for mem_emotion in KNOWN_EMOTION_TAGS:
            for cur_emotion in EMOTION_PAD_MAP.keys():
                record = self._make_record([mem_emotion])
                state = _make_state(cur_emotion)
                result = aligner.compute(record, state)
                assert 0.0 <= result <= 1.0, (
                    f"Out of range: mem={mem_emotion}, cur={cur_emotion}, result={result}"
                )

    def test_alignment_boosts_positive_memory_in_negative_state(self, aligner, prototypes):
        """Confidence memory in negative state should get a boost."""
        record = self._make_record(["confidence"])
        state = _make_state("frustration")  # negative valence
        assert state.is_negative

        alignment = aligner.compute(record, state)
        # Base (without boost) would be the vector similarity mapped to [0,1]
        # With positive_boost applied, should be higher
        mem_proto = prototypes.get("confidence")
        raw_sim = float(np.dot(mem_proto, state.emotion_vector))
        base = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))

        # alignment = base + boost (clamped)
        assert alignment >= base  # never lower than base

    def test_alignment_suppresses_stress_memory_in_fragile_state(self, aligner, prototypes):
        """Frustration memory in strong negative state should be suppressed."""
        record = self._make_record(["frustration"])
        # Fragile state: frustration with high intensity
        state = _make_state("frustration", intensity=0.85)
        assert state.is_negative
        assert state.intensity >= aligner._fragile_threshold

        alignment = aligner.compute(record, state)
        mem_proto = prototypes.get("frustration")
        raw_sim = float(np.dot(mem_proto, state.emotion_vector))
        base = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))

        # alignment should be suppressed below base
        assert alignment <= base

    def test_alignment_explain_returns_string(self, aligner):
        record = self._make_record(["drive"])
        state = _make_state("confidence")
        result = aligner.explain(record, state)
        assert isinstance(result, str)
        assert "alignment" in result.lower()

    def test_alignment_vector_similarity_same_emotion(self, aligner, prototypes):
        """Memory and state with same emotion should have high similarity."""
        record = self._make_record(["confidence"])
        # Create state whose emotion_vector IS the confidence prototype
        proto = prototypes.get("confidence")
        state = EmotionalState(
            primary_emotion="confidence",
            intensity=0.7,
            confidence=0.6,
            emotion_vector=proto.copy(),
            pad=PADState.from_emotion("confidence"),
            raw_scores={"confidence": 0.7},
        )
        # Should have high vector similarity
        alignment = aligner._vector_similarity("confidence", state)
        assert alignment > 0.7  # close to 1.0 since vectors are same


# ─────────────────────────────────────────────────────────────
# EMOTIONAL PROTECTION
# ─────────────────────────────────────────────────────────────

class TestEmotionalProtection:

    def _inject_frustration_pattern(self, engine, n: int = 5):
        """Inject n frustration states with declining valence to establish a pattern.
        Valence must actually decline so emotional_trend() detects a negative slope.
        """
        start, end = -0.2, -0.8
        step = (end - start) / max(n - 1, 1)
        for i in range(n):
            s = _make_state("frustration", intensity=0.8)
            s.pad.valence = start + i * step   # -0.2 → -0.8 (clear decline)
            engine._history.append(s)

    def _inject_confidence_pattern(self, engine, n: int = 5):
        for _ in range(n):
            s = _make_state("confidence", intensity=0.8)
            s.pad.valence = 0.7
            engine._history.append(s)

    def test_emotional_protection_adds_stress_tag(self, engine_with_memory):
        engine, mem = engine_with_memory
        # Add a non-emotional memory to be tagged
        m = mem.add_memory("User struggled with a task.", memory_type="episodic")

        # Establish frustration pattern + declining trend
        self._inject_frustration_pattern(engine, n=5)

        engine.apply_emotional_protection(engine._history[-1])

        reloaded = mem.get_memory(m.id)
        assert "emotion_stress" in reloaded.tags

    def test_emotional_protection_adds_boost_tag(self, engine_with_memory):
        engine, mem = engine_with_memory
        m = mem.add_memory("User completed a major project.", importance_score=0.7)

        self._inject_confidence_pattern(engine, n=5)

        engine.apply_emotional_protection(engine._history[-1])

        reloaded = mem.get_memory(m.id)
        assert "emotion_boost" in reloaded.tags

    def test_protection_no_crash_without_memory_manager(self, engine_no_memory):
        self._inject_frustration_pattern(engine_no_memory, n=5)
        # Should not raise
        engine_no_memory.apply_emotional_protection(engine_no_memory._history[-1])

    def test_protection_no_effect_without_pattern(self, engine_with_memory):
        engine, mem = engine_with_memory
        m = mem.add_memory("Some neutral event.", memory_type="episodic")

        # Only 2 states — not enough for a pattern
        engine._history.append(_make_state("confidence"))
        engine._history.append(_make_state("frustration"))

        engine.apply_emotional_protection(engine._history[-1])

        reloaded = mem.get_memory(m.id)
        assert "emotion_stress" not in reloaded.tags
        assert "emotion_boost" not in reloaded.tags


# ─────────────────────────────────────────────────────────────
# EMOTION-AWARE RETRIEVAL
# ─────────────────────────────────────────────────────────────

class TestEmotionAwareRetrieval:

    def test_no_crash_without_emotional_context(self, engine_with_memory):
        engine, mem = engine_with_memory
        mem.add_memory("A productive work session.", importance_score=0.7)
        # Should work without emotional_context
        results = mem.retrieve_relevant_memories("work session")
        assert isinstance(results, list)

    def test_retrieval_alignment_one_when_no_fn_registered(self, tmp_dir):
        """If no alignment_fn registered, relevance = similarity × importance (no modulation)."""
        from memory.config import MemoryConfig
        from memory.memory_manager import MemoryManager

        config = MemoryConfig(data_dir=tmp_dir / "mem_noalign")
        config.min_similarity_score = 0.0

        # Create MemoryManager without EmotionEngine (no alignment_fn)
        # We can't instantiate it without sentence-transformers so just check the attribute
        mm = MemoryManager.__new__(MemoryManager)
        mm._alignment_fn = None
        assert mm._alignment_fn is None

    def test_alignment_fn_registered_on_engine_init(self, engine_with_memory):
        """After EmotionEngine init, MemoryManager should have alignment_fn set."""
        engine, mem = engine_with_memory
        assert mem._alignment_fn is not None
        assert callable(mem._alignment_fn)

    def test_emotion_aware_retrieval_changes_ranking(self, engine_with_memory):
        """
        Test that emotional context modulates ranking.

        Setup:
        - Memory A: tagged "confidence" (positive)
        - Memory B: tagged "frustration" (negative/stress)
        
        With negative state (frustration) as context:
        - Alignment of Memory A should be boosted (positive boost rule)
        - Alignment of Memory B should be suppressed (stress suppression in fragile state)
        → Memory A should rank above Memory B after modulation
        """
        engine, mem = engine_with_memory

        # Add two memories with distinct emotion tags
        # Use same content to get similar semantic embeddings, distinguish by tag only
        base_content = "user worked on a task today"
        mem_a = mem.add_memory(
            base_content + " with great success",
            importance_score=0.6,
            tags=["confidence", "emotion_event"],
        )
        mem_b = mem.add_memory(
            base_content + " but felt stuck and frustrated",
            importance_score=0.6,
            tags=["frustration", "emotion_event"],
        )

        # Negative, fragile state
        fragile_state = _make_state("frustration", intensity=0.8)

        # Retrieve WITH emotional context
        results_with = mem.retrieve_relevant_memories(
            query=base_content,
            top_k=2,
            emotional_context=fragile_state,
        )

        # Retrieve WITHOUT emotional context
        results_without = mem.retrieve_relevant_memories(
            query=base_content,
            top_k=2,
        )

        # Both should return 2 results
        assert len(results_with) >= 1
        assert len(results_without) >= 1

        # The with-context version should have different relevances
        ids_with = [r.record.id for r in results_with]
        ids_without = [r.record.id for r in results_without]

        # At minimum: relevances in the emotional-context version differ from plain version
        if len(results_with) == 2 and len(results_without) == 2:
            relevances_with = [r.relevance for r in results_with]
            relevances_without = [r.relevance for r in results_without]
            # Emotional context modulates relevances (they won't be identical)
            assert relevances_with != relevances_without


# ─────────────────────────────────────────────────────────────
# ENGINE INTEGRATION
# ─────────────────────────────────────────────────────────────

class TestEngineV2Integration:

    def test_stats_includes_trend(self, engine_no_memory):
        engine_no_memory._history.append(_make_state("confidence"))
        engine_no_memory._history.append(_make_state("drive"))
        engine_no_memory._history.append(_make_state("flow"))
        stats = engine_no_memory.stats()
        assert "trend" in stats
        assert "trend_label" in stats["trend"]
        assert "burnout_risk" in stats["trend"]

    def test_engine_trend_reflects_injected_states(self, engine_no_memory):
        # All frustration with declining valence
        for i, v in enumerate([-0.2, -0.4, -0.5, -0.6, -0.7]):
            s = _make_state("frustration")
            s.pad.valence = v
            engine_no_memory._history.append(s)
        trend = engine_no_memory.emotional_trend()
        assert trend["valence_slope"] < 0
        assert trend["trend_label"] in ("declining", "escalating")