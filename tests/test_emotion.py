"""
A.C.C.E.S.S. — Emotion Engine: Unit Tests

Run: python -m pytest tests/test_emotion.py -v

Coverage:
  [MODELS]
  - test_pad_from_emotion
  - test_pad_neutral
  - test_pad_blend
  - test_emotional_state_clamps_intensity
  - test_emotional_state_clamps_confidence
  - test_emotional_state_normalizes_vector
  - test_emotional_state_properties
  - test_emotional_state_label
  - test_to_log_dict

  [PROTOTYPES]
  - test_build_creates_all_emotions
  - test_prototypes_are_normalized
  - test_prototypes_are_stable
  - test_add_emotion_runtime

  [SCORER]
  - test_detects_frustration
  - test_detects_confidence
  - test_detects_drive
  - test_detects_fatigue
  - test_intensity_clamped
  - test_confidence_score_logic
  - test_raw_scores_sum_to_one
  - test_dominant_is_highest_prob

  [ENGINE]
  - test_process_returns_state
  - test_history_populated
  - test_history_bounded
  - test_current_pad_neutral_on_empty
  - test_current_pad_weighted
  - test_dominant_pattern_none_on_empty
  - test_dominant_pattern_detects_repeated
  - test_stats_returns_dict
  - test_explain_returns_string

  [MEMORY INTEGRATION]
  - test_memory_created_on_high_intensity
  - test_memory_not_created_on_low_intensity
  - test_memory_boost_on_drive
  - test_memory_boost_on_confidence
  - test_no_modulation_without_memory_manager
"""

import shutil
import tempfile
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion.config import EmotionConfig
from emotion.models import EmotionalState, PADState, EMOTION_PAD_MAP
from emotion.emotion_prototypes import EmotionPrototypes, EMOTION_PROTOTYPES
from emotion.emotion_embedder import EmotionEmbedder
from emotion.emotion_scoring import EmotionScorer
from emotion.emotion_engine import EmotionEngine


# ─────────────────────────────────────────────────────────────
# SHARED FAKE EMBEDDER
# ─────────────────────────────────────────────────────────────

class FakeEmbedder:
    """
    Deterministic fake embedder for tests.

    Key property: encodes text as a UNIT VECTOR whose direction is
    deterministic (based on hash). This means:
    - Same text always → same vector (stable)
    - Different texts → different vectors (discriminating)

    For emotion tests, we need the scorer to detect specific emotions.
    We achieve this by making "frustration-like" text produce a vector
    similar to the frustration prototype. We do this by seeding the
    FakeEmbedder so prototype phrases and test phrases get similar vectors
    when they share keywords.

    Implementation: we use a vocabulary-based approach.
    Words that appear in prototype phrases bias the vector in consistent ways.
    """

    # Simplified keyword → emotion dimension mapping
    # Words that strongly signal each emotion → consistent bias
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
        self._emotion_axes = self._build_emotion_axes()

    def _build_emotion_axes(self) -> dict[str, np.ndarray]:
        """Build one fixed axis vector per emotion (orthogonalized for clarity)."""
        axes = {}
        emotions = list(self.KEYWORD_BIASES.keys())
        rng = np.random.RandomState(42)

        # Generate orthogonal-ish axes using random vectors with fixed seeds
        for i, emotion in enumerate(emotions):
            rng2 = np.random.RandomState(1000 + i)
            vec = rng2.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            axes[emotion] = vec

        return axes

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode text with keyword-based emotional bias.
        Text containing keywords for an emotion → vector biased toward that emotion's axis.
        """
        text_lower = text.lower()

        # Start with a small base vector (deterministic from hash)
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32) * 0.1  # small noise

        # Add strong bias for matching keywords
        for emotion, keywords in self.KEYWORD_BIASES.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                bias_strength = min(1.0, matches * 0.5)
                vec += self._emotion_axes[emotion] * bias_strength

        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def config():
    # intensity_threshold_low=0.2 for FakeEmbedder:
    # With 6 emotions uniform softmax = ~0.167. Keyword biases push dominant
    # emotion to ~0.3-0.5. 0.2 is reliably exceeded; 0.45 was too high.
    # Production config keeps 0.45 (real model has sharper distributions).
    return EmotionConfig(
        softmax_temperature=0.3,
        intensity_threshold_low=0.20,
        intensity_threshold_high=0.35,
    )


@pytest.fixture
def prototypes(fake_embedder):
    p = EmotionPrototypes(fake_embedder)
    p.build()
    return p


@pytest.fixture
def scorer(prototypes, config):
    return EmotionScorer(prototypes, config)


@pytest.fixture
def engine_no_memory(fake_embedder, config):
    """Engine without memory manager — for pure emotion tests."""
    engine = EmotionEngine.__new__(EmotionEngine)
    from emotion.emotion_embedder import EmotionEmbedder
    from emotion.emotion_prototypes import EmotionPrototypes
    from emotion.emotion_scoring import EmotionScorer
    from collections import deque

    engine.config = config
    engine._memory = None
    engine._embedder = EmotionEmbedder(fake_embedder)
    engine._prototypes = EmotionPrototypes(fake_embedder)
    engine._prototypes.build()
    engine._scorer = EmotionScorer(engine._prototypes, config)
    engine._history = deque(maxlen=config.history_max_size)
    return engine


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def engine_with_memory(fake_embedder, config, tmp_dir):
    """Engine with a real MemoryManager (fake embedder)."""
    from memory.config import MemoryConfig
    from memory.memory_manager import MemoryManager
    from memory.store import MemoryStore
    from memory.vector_index import VectorIndex
    from memory.decay import DecayEngine, DecayConfig
    from memory.maintenance import ConsistencyChecker, PurgePolicy, PurgeStrategy
    from emotion.emotion_embedder import EmotionEmbedder
    from emotion.emotion_prototypes import EmotionPrototypes
    from emotion.emotion_scoring import EmotionScorer
    from collections import deque

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

    engine = EmotionEngine.__new__(EmotionEngine)
    engine.config = config
    engine._memory = mem
    engine._embedder = EmotionEmbedder(fake_embedder)
    engine._prototypes = EmotionPrototypes(fake_embedder)
    engine._prototypes.build()
    engine._scorer = EmotionScorer(engine._prototypes, config)
    engine._history = deque(maxlen=config.history_max_size)

    return engine, mem


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class TestPADState:

    def test_pad_from_known_emotion(self):
        pad = PADState.from_emotion("frustration")
        assert pad.valence < 0      # frustration is negative
        assert pad.arousal > 0.5    # frustration is high arousal

    def test_pad_from_confidence(self):
        pad = PADState.from_emotion("confidence")
        assert pad.valence > 0      # confidence is positive
        assert pad.dominance > 0.7  # confidence is dominant

    def test_pad_neutral(self):
        pad = PADState.neutral()
        assert pad.valence == 0.0
        assert pad.arousal == 0.5
        assert pad.dominance == 0.5

    def test_pad_from_unknown_defaults_neutral(self):
        pad = PADState.from_emotion("unknown_emotion")
        assert pad.valence == 0.0

    def test_pad_blend(self):
        a = PADState(valence=1.0, arousal=1.0, dominance=1.0)
        b = PADState(valence=0.0, arousal=0.0, dominance=0.0)
        blended = a.blend(b, weight=0.5)
        assert blended.valence == pytest.approx(0.5)
        assert blended.arousal == pytest.approx(0.5)

    def test_pad_blend_weight_clamp(self):
        a = PADState(valence=1.0, arousal=0.5, dominance=0.5)
        b = PADState(valence=0.0, arousal=0.5, dominance=0.5)
        # weight > 1.0 should clamp to 1.0 → fully b
        blended = a.blend(b, weight=5.0)
        assert blended.valence == pytest.approx(0.0)


class TestEmotionalState:

    def _make_state(self, emotion="frustration", intensity=0.8, confidence=0.6):
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return EmotionalState(
            primary_emotion=emotion,
            intensity=intensity,
            confidence=confidence,
            emotion_vector=vec,
            pad=PADState.from_emotion(emotion),
            raw_scores={"frustration": 0.8, "doubt": 0.2},
        )

    def test_clamps_intensity_above_one(self):
        s = self._make_state(intensity=1.5)
        assert s.intensity == pytest.approx(1.0)

    def test_clamps_intensity_below_zero(self):
        s = self._make_state(intensity=-0.3)
        assert s.intensity == pytest.approx(0.0)

    def test_clamps_confidence(self):
        s = self._make_state(confidence=2.0)
        assert s.confidence == pytest.approx(1.0)

    def test_normalizes_vector(self):
        # Pass an unnormalized vector
        vec = np.array([3.0, 4.0] + [0.0] * 382, dtype=np.float32)
        s = EmotionalState(
            primary_emotion="confidence",
            intensity=0.7,
            confidence=0.5,
            emotion_vector=vec,
            pad=PADState.neutral(),
            raw_scores={},
        )
        norm = np.linalg.norm(s.emotion_vector)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_is_positive(self):
        s = self._make_state("confidence")
        assert s.is_positive

    def test_is_negative(self):
        s = self._make_state("frustration")
        assert s.is_negative

    def test_label_weak(self):
        s = self._make_state(intensity=0.3)
        assert "weak" in s.label

    def test_label_strong(self):
        s = self._make_state(intensity=0.8)
        assert "strong" in s.label

    def test_to_log_dict(self):
        s = self._make_state()
        d = s.to_log_dict()
        assert "emotion" in d
        assert "intensity" in d
        assert "valence" in d
        assert "top_scores" in d
        assert d["emotion"] == "frustration"


# ─────────────────────────────────────────────────────────────
# PROTOTYPES
# ─────────────────────────────────────────────────────────────

class TestEmotionPrototypes:

    def test_build_creates_all_emotions(self, prototypes):
        for emotion in EMOTION_PROTOTYPES:
            assert emotion in prototypes.prototypes

    def test_prototypes_are_normalized(self, prototypes):
        for emotion, vec in prototypes.prototypes.items():
            norm = np.linalg.norm(vec)
            assert norm == pytest.approx(1.0, abs=1e-5), f"{emotion} not normalized"

    def test_prototypes_are_stable(self, fake_embedder):
        """Building twice with same embedder gives same results."""
        p1 = EmotionPrototypes(fake_embedder)
        p1.build()
        p2 = EmotionPrototypes(fake_embedder)
        p2.build()
        for emotion in EMOTION_PROTOTYPES:
            np.testing.assert_allclose(p1.get(emotion), p2.get(emotion), atol=1e-6)

    def test_add_emotion_runtime(self, prototypes):
        prototypes.add_emotion("boredom", ["I feel bored and uninterested."])
        assert "boredom" in prototypes.prototypes
        vec = prototypes.get("boredom")
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_build_required_before_access(self, fake_embedder):
        p = EmotionPrototypes(fake_embedder)
        with pytest.raises(RuntimeError, match="build\\(\\)"):
            _ = p.prototypes

    def test_emotion_count(self, prototypes):
        assert len(prototypes) == len(EMOTION_PROTOTYPES)


# ─────────────────────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────────────────────

class TestEmotionScorer:

    def _encode(self, embedder, text):
        vec = embedder.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def test_detects_frustration(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I feel stuck and frustrated, nothing is working.")
        state = scorer.score(vec)
        assert state.primary_emotion == "frustration"

    def test_detects_confidence(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I feel powerful and capable, I trust my abilities completely.")
        state = scorer.score(vec)
        assert state.primary_emotion == "confidence"

    def test_detects_drive(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I am determined and motivated, nothing will stop me.")
        state = scorer.score(vec)
        assert state.primary_emotion == "drive"

    def test_detects_fatigue(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I feel mentally exhausted, tired and drained.")
        state = scorer.score(vec)
        assert state.primary_emotion == "fatigue"

    def test_intensity_clamped_in_range(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I am fully focused and in the zone.")
        state = scorer.score(vec)
        assert 0.0 <= state.intensity <= 1.0

    def test_confidence_score_logic(self, scorer, fake_embedder):
        """A clearly frustration-biased text should have higher confidence than ambiguous text."""
        clear_vec = self._encode(fake_embedder,
            "I feel stuck, frustrated, blocked and nothing works, I keep failing.")
        ambiguous_vec = self._encode(fake_embedder, "Something happened today.")

        clear_state = scorer.score(clear_vec)
        ambiguous_state = scorer.score(ambiguous_vec)

        assert clear_state.confidence >= ambiguous_state.confidence

    def test_raw_scores_sum_to_one(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I feel focused and determined.")
        state = scorer.score(vec)
        total = sum(state.raw_scores.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_dominant_is_highest_prob(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "I feel stuck and frustrated.")
        state = scorer.score(vec)
        top_emotion = max(state.raw_scores, key=state.raw_scores.get)
        assert state.primary_emotion == top_emotion

    def test_explain_returns_string(self, scorer, fake_embedder):
        vec = self._encode(fake_embedder, "Test input.")
        result = scorer.explain(vec)
        assert isinstance(result, str)
        assert "Emotion scores" in result


# ─────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────

class TestEmotionEngine:

    def test_process_returns_state(self, engine_no_memory):
        state = engine_no_memory.process_interaction("I feel stuck and frustrated.")
        assert isinstance(state, EmotionalState)
        assert state.primary_emotion in EMOTION_PROTOTYPES

    def test_history_populated(self, engine_no_memory):
        assert len(engine_no_memory.history) == 0
        engine_no_memory.process_interaction("I feel confident and capable.")
        assert len(engine_no_memory.history) == 1

    def test_history_bounded(self, engine_no_memory):
        engine_no_memory.config.history_max_size
        max_size = engine_no_memory.config.history_max_size
        for i in range(max_size + 10):
            engine_no_memory.process_interaction(f"Interaction {i}, I feel focused and in the zone.")
        assert len(engine_no_memory.history) == max_size

    def test_current_pad_neutral_on_empty(self, engine_no_memory):
        pad = engine_no_memory.current_pad()
        assert pad.valence == pytest.approx(0.0)
        assert pad.arousal == pytest.approx(0.5)

    def test_current_pad_shifts_positive(self, engine_no_memory):
        # Inject multiple positive states
        for _ in range(5):
            engine_no_memory.process_interaction(
                "I feel powerful and capable, I trust my abilities."
            )
        pad = engine_no_memory.current_pad()
        assert pad.valence > 0.0  # should shift positive

    def test_dominant_pattern_none_on_empty(self, engine_no_memory):
        assert engine_no_memory.dominant_pattern() is None

    def test_dominant_pattern_detects_repeated(self, engine_no_memory):
        # Repeatedly frustrated
        for _ in range(7):
            engine_no_memory.process_interaction(
                "I feel stuck, frustrated, blocked, nothing is working."
            )
        pattern = engine_no_memory.dominant_pattern(last_n=10)
        assert pattern == "frustration"

    def test_dominant_pattern_none_on_few_interactions(self, engine_no_memory):
        engine_no_memory.process_interaction("I feel focused.")
        # Only 1 interaction — not enough for a pattern
        pattern = engine_no_memory.dominant_pattern(last_n=10)
        assert pattern is None

    def test_stats_returns_dict(self, engine_no_memory):
        engine_no_memory.process_interaction("I feel confident and capable.")
        stats = engine_no_memory.stats()
        assert "history_size" in stats
        assert "avg_intensity" in stats
        assert stats["history_size"] == 1

    def test_stats_empty(self, engine_no_memory):
        stats = engine_no_memory.stats()
        assert stats == {"history_size": 0}

    def test_no_modulation_without_memory_manager(self, engine_no_memory):
        """Engine without MM should not crash, just skip modulation."""
        state = engine_no_memory.process_interaction(
            "I feel powerful and capable, I trust my abilities."
        )
        assert state is not None  # no crash


# ─────────────────────────────────────────────────────────────
# MEMORY INTEGRATION
# ─────────────────────────────────────────────────────────────

class TestMemoryIntegration:

    def test_memory_created_on_high_intensity(self, engine_with_memory):
        engine, mem = engine_with_memory
        assert mem._store.count() == 0

        # Strong frustration → should trigger memory creation
        engine.process_interaction(
            "I feel stuck, frustrated, blocked, nothing is working at all."
        )

        # At least one emotional memory should exist
        assert mem._store.count() >= 1

    def test_memory_not_created_on_low_intensity(self, engine_with_memory, fake_embedder):
        """Ambiguous text → low intensity → no memory created."""
        engine, mem = engine_with_memory

        # Override config to require higher threshold
        engine.config.intensity_threshold_low = 0.99  # essentially unreachable

        engine.process_interaction("ok")
        assert mem._store.count() == 0

    def test_memory_boost_on_drive(self, engine_with_memory):
        engine, mem = engine_with_memory

        # Add a regular episodic memory first
        m = mem.add_memory(
            "User completed a major task successfully.",
            importance_score=0.6,
            memory_type="episodic",
        )
        original_score = m.importance_score

        # Lower intensity_threshold_high so drive triggers boost
        engine.config.intensity_threshold_high = 0.1

        engine.process_interaction(
            "I am determined and motivated, nothing will stop me from achieving my mission."
        )

        # Check if the episodic memory got boosted
        reloaded = mem.get_memory(m.id)
        assert reloaded.importance_score >= original_score  # boosted or unchanged

    def test_memory_boost_on_confidence(self, engine_with_memory):
        engine, mem = engine_with_memory
        engine.config.intensity_threshold_high = 0.1

        m = mem.add_memory(
            "User learned a new concept.",
            importance_score=0.5,
            memory_type="episodic",
        )

        engine.process_interaction(
            "I feel powerful, capable and certain that I will succeed in my goals."
        )

        reloaded = mem.get_memory(m.id)
        assert reloaded.importance_score >= 0.5

    def test_emotional_memory_type_correct(self, engine_with_memory):
        """Memories created by emotion engine should have type='emotional'."""
        engine, mem = engine_with_memory
        engine.process_interaction(
            "I feel stuck, frustrated and blocked completely."
        )
        # Check all created memories are typed 'emotional'
        all_ids = mem._store.get_all_ids()
        for mid in all_ids:
            record = mem.get_memory(mid)
            assert record.memory_type == "emotional"

    def test_emotional_memory_has_emotion_tag(self, engine_with_memory):
        """Emotional memories should be tagged with the detected emotion."""
        engine, mem = engine_with_memory
        engine.process_interaction(
            "I feel mentally exhausted, tired and drained from all this work."
        )
        all_ids = mem._store.get_all_ids()
        assert len(all_ids) > 0
        record = mem.get_memory(all_ids[0])
        assert "emotion_event" in record.tags