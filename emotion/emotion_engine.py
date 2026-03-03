"""
A.C.C.E.S.S. — Emotion Engine v2

Central orchestrator for emotion detection, memory modulation, and adaptive behavior.

Architecture:
    EmotionEngine
        ├── EmotionEmbedder    (text → vector)
        ├── EmotionPrototypes  (emotion → centroid vector)
        ├── EmotionScorer      (vector → EmotionalState)
        ├── EmotionAlignment   (memory × state → alignment score)
        └── MemoryManager      (optional — modulation target)

New in v2:
    - emotional_trend()          : linear regression on PAD history → trend + risk flags
    - apply_emotional_protection(): tag memories based on sustained emotional patterns
    - EmotionAlignment injected into MemoryManager for emotion-aware retrieval
"""

import logging
from collections import deque
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

from .config import EmotionConfig
from .emotion_alignment import EmotionAlignment
from .emotion_embedder import EmotionEmbedder, EmbedderProtocol
from .emotion_prototypes import EmotionPrototypes, EMOTION_PROTOTYPES
from .emotion_scoring import EmotionScorer
from .models import EmotionalState, PADState

if TYPE_CHECKING:
    from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class EmotionEngine:
    """
    Main interface for the A.C.C.E.S.S. Emotion subsystem.

    Usage:
        engine = EmotionEngine(memory_manager=mem)
        state = engine.process_interaction("I feel completely stuck on this task.")
        trend = engine.emotional_trend()
        print(trend["trend_label"], trend["burnout_risk"])
    """

    def __init__(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        embedder: Optional[EmbedderProtocol] = None,
        config: Optional[EmotionConfig] = None,
    ):
        self.config = config or EmotionConfig()
        self._memory = memory_manager

        # Embedder: reuse MemoryManager's if possible — avoid loading model twice
        if embedder is None and memory_manager is not None:
            raw_embedder = memory_manager._embedder
        elif embedder is not None:
            raw_embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer
            raw_embedder = SentenceTransformer(self.config.embedding_model)

        self._embedder = EmotionEmbedder(raw_embedder)

        self._prototypes = EmotionPrototypes(raw_embedder)
        self._prototypes.build()

        self._scorer = EmotionScorer(self._prototypes, self.config)

        # Alignment module — injected into MemoryManager for emotion-aware retrieval
        self._alignment = EmotionAlignment(
            prototypes=self._prototypes.prototypes,
            fragile_threshold=self.config.alignment_fragile_threshold,
            stress_suppression=self.config.alignment_stress_suppression,
            positive_boost=self.config.alignment_positive_boost,
        )

        # Register alignment function with MemoryManager (zero circular dependency)
        if self._memory is not None:
            self._memory.set_emotion_alignment(self._alignment.compute)

        # Rolling history — bounded deque, O(1) append/pop
        self._history: deque[EmotionalState] = deque(maxlen=self.config.history_max_size)

        logger.info(
            f"EmotionEngine v2 ready — "
            f"{len(self._prototypes)} emotions, "
            f"memory={'ON' if memory_manager else 'OFF'}"
        )

    # ─────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────

    def process_interaction(self, text: str, session_id: Optional[str] = None) -> EmotionalState:
        """
        Full pipeline: text → EmotionalState + optional memory modulation.
        """
        vec = self._embedder.encode(text)
        state = self._scorer.score(vec, source_text=text)
        self._history.append(state)

        if self._memory is not None and state.intensity >= self.config.intensity_threshold_low:
            self.apply_memory_modulation(state, text=text, session_id=session_id)

        logger.info(
            f"Processed: {state.label} | "
            f"intensity={state.intensity:.3f} | "
            f"confidence={state.confidence:.3f} | "
            f"{state.pad}"
        )
        return state

    # ─────────────────────────────────────────────────────────────
    # MEMORY MODULATION
    # ─────────────────────────────────────────────────────────────

    def apply_memory_modulation(
        self,
        state: EmotionalState,
        text: str = "",
        session_id: Optional[str] = None,
    ) -> None:
        """Apply emotion-driven effects on memory (create record + reinforce/penalize)."""
        importance = min(
            1.0,
            self.config.emotional_memory_base_importance
            + state.intensity * self.config.emotional_memory_intensity_factor
        )

        emotion_summary = (
            f"[{state.primary_emotion.upper()}] "
            f"intensity={state.intensity:.2f} | "
            f"valence={state.pad.valence:+.2f} | "
            f"{text[:120]}"
        )

        self._memory.add_memory(
            content=text or f"Emotional event: {state.primary_emotion}",
            summary=emotion_summary,
            memory_type="emotional",
            tags=[state.primary_emotion, "emotion_event"],
            importance_score=importance,
            source="emotion_engine",
            session_id=session_id,
        )

        if state.intensity < self.config.intensity_threshold_high:
            return

        is_positive = state.primary_emotion in self.config.positive_emotions
        is_negative = state.primary_emotion in self.config.negative_emotions

        if not (is_positive or is_negative):
            return

        recent = self._memory.retrieve_relevant_memories(
            query=text or state.primary_emotion,
            top_k=self.config.modulation_lookback,
            min_importance=0.3,
        )

        delta = (
            self.config.positive_emotion_boost if is_positive
            else self.config.negative_emotion_penalty
        )
        action = "boost" if is_positive else "fragility_mark"

        for retrieved in recent:
            record = retrieved.record
            if record.memory_type == "emotional":
                continue
            new_score = self._memory.update_importance(record.id, delta=delta)
            logger.debug(
                f"Memory modulation [{action}]: "
                f"{record.id[:8]}... → importance={new_score:.3f} ({delta:+.3f})"
            )

    # ─────────────────────────────────────────────────────────────
    # EMOTIONAL MOMENTUM MODEL (NEW)
    # ─────────────────────────────────────────────────────────────

    def emotional_trend(self, last_n: Optional[int] = None) -> dict:
        """
        Compute emotional trajectory via linear regression on PAD history.

        Returns slopes for Valence, Arousal, Dominance + a human-readable label.
        Also flags burnout_risk and mania_risk based on configurable thresholds.

        Returns:
            {
                "valence_slope":   float,  # negative = worsening, positive = improving
                "arousal_slope":   float,
                "dominance_slope": float,
                "trend_label":     str,    # "stable"|"improving"|"declining"|"escalating"
                "burnout_risk":    bool,
                "mania_risk":      bool,
                "data_points":     int,    # how many states were used
            }

        Requires >= 3 data points. Returns "stable" with no risk flags if insufficient.
        """
        n = last_n or self.config.momentum_window
        history = list(self._history)[-n:]

        default = {
            "valence_slope": 0.0,
            "arousal_slope": 0.0,
            "dominance_slope": 0.0,
            "trend_label": "stable",
            "burnout_risk": False,
            "mania_risk": False,
            "data_points": len(history),
        }

        if len(history) < 3:
            return default

        x = np.arange(len(history), dtype=np.float64)
        valences = np.array([s.pad.valence for s in history], dtype=np.float64)
        arousals = np.array([s.pad.arousal for s in history], dtype=np.float64)
        dominances = np.array([s.pad.dominance for s in history], dtype=np.float64)

        v_slope = self._linear_slope(x, valences)
        a_slope = self._linear_slope(x, arousals)
        d_slope = self._linear_slope(x, dominances)

        # Clamp to [-1, 1] — slope > 1 per step is unrealistic
        v_slope = max(-1.0, min(1.0, v_slope))
        a_slope = max(-1.0, min(1.0, a_slope))
        d_slope = max(-1.0, min(1.0, d_slope))

        threshold = self.config.momentum_slope_threshold

        # Trend label: based on valence direction primarily
        if abs(v_slope) < threshold:
            trend_label = "stable"
        elif v_slope > threshold:
            trend_label = "improving"
        elif v_slope < -threshold:
            # Escalating = declining valence + rising arousal (agitation)
            trend_label = "escalating" if a_slope > threshold else "declining"
        else:
            trend_label = "stable"

        # Risk flags
        burnout_risk = v_slope < self.config.burnout_slope_threshold
        # Mania: valence rising AND arousal spiking (unsustainable high energy)
        mania_risk = (
            a_slope > self.config.mania_slope_threshold
            and v_slope > threshold
        )

        return {
            "valence_slope": round(v_slope, 4),
            "arousal_slope": round(a_slope, 4),
            "dominance_slope": round(d_slope, 4),
            "trend_label": trend_label,
            "burnout_risk": burnout_risk,
            "mania_risk": mania_risk,
            "data_points": len(history),
        }

    @staticmethod
    def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Simple OLS slope: Σ(xi-x̄)(yi-ȳ) / Σ(xi-x̄)²"""
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = float(((x - x_mean) * (y - y_mean)).sum())
        denominator = float(((x - x_mean) ** 2).sum())
        return numerator / denominator if abs(denominator) > 1e-10 else 0.0

    # ─────────────────────────────────────────────────────────────
    # EMOTIONAL PROTECTION (NEW)
    # ─────────────────────────────────────────────────────────────

    def apply_emotional_protection(self, state: EmotionalState) -> None:
        """
        Tag recent memories based on sustained emotional patterns.

        Triggered when a dominant emotional pattern is detected:
        - Sustained NEGATIVE + declining valence → tag recent memories as 'emotion_stress'
          (signals these memories are associated with a difficult period)
        - Sustained POSITIVE → tag recent memories as 'emotion_boost' + slight importance boost
          (protects momentum memories from decay)

        Uses ONLY the MemoryManager public API. No direct DB access.
        """
        if self._memory is None:
            return

        # Need at least protection_repeat_count + a few more states for pattern detection
        if len(self._history) < self.config.protection_repeat_count:
            return

        pattern = self.dominant_pattern(last_n=self.config.protection_repeat_count + 2)
        if pattern is None:
            return

        trend = self.emotional_trend()

        if pattern in self.config.negative_emotions and trend["valence_slope"] < 0:
            # Declining negative pattern → stress protection
            self._apply_stress_protection()

        elif pattern in self.config.positive_emotions:
            # Sustained positive pattern → boost protection
            self._apply_boost_protection()

    def _apply_stress_protection(self) -> None:
        """Tag recent non-emotional memories as stress-linked."""
        recent = self._memory.get_recent_memories(
            n=self.config.protection_lookback_memories
        )
        tagged = 0
        for record in recent:
            if record.memory_type == "emotional":
                continue
            self._memory.add_tags_to_memory(record.id, ["emotion_stress"])
            tagged += 1
        if tagged:
            logger.info(f"Protection [stress]: tagged {tagged} memories as emotion_stress")

    def _apply_boost_protection(self) -> None:
        """Tag recent important memories as emotion_boost + reinforce importance."""
        recent = self._memory.get_recent_memories(
            n=self.config.protection_lookback_memories,
            min_importance=0.4,
        )
        boosted = 0
        for record in recent:
            if record.memory_type == "emotional":
                continue
            self._memory.add_tags_to_memory(record.id, ["emotion_boost"])
            # Only boost if not already near max (avoid score inflation)
            if record.importance_score < 0.8:
                self._memory.update_importance(record.id, delta=0.05)
            boosted += 1
        if boosted:
            logger.info(f"Protection [boost]: reinforced {boosted} memories with emotion_boost")

    # ─────────────────────────────────────────────────────────────
    # HISTORY & ANALYTICS
    # ─────────────────────────────────────────────────────────────

    @property
    def history(self) -> list[EmotionalState]:
        return list(self._history)

    def current_pad(self) -> PADState:
        """Exponentially weighted blend of recent PAD states."""
        if not self._history:
            return PADState.neutral()

        states = list(self._history)
        weights = np.array([1.1 ** i for i in range(len(states))], dtype=float)
        weights /= weights.sum()

        blended_v = sum(w * s.pad.valence for w, s in zip(weights, states))
        blended_a = sum(w * s.pad.arousal for w, s in zip(weights, states))
        blended_d = sum(w * s.pad.dominance for w, s in zip(weights, states))

        return PADState(
            valence=max(-1.0, min(1.0, blended_v)),
            arousal=max(0.0, min(1.0, blended_a)),
            dominance=max(0.0, min(1.0, blended_d)),
        )

    def dominant_pattern(self, last_n: int = 10) -> Optional[str]:
        """Return dominant emotion if it appears in >50% of recent strong interactions."""
        if not self._history:
            return None

        recent = list(self._history)[-last_n:]
        if len(recent) < 3:
            return None

        counts: dict[str, int] = {}
        for s in recent:
            if s.intensity >= self.config.intensity_threshold_low:
                counts[s.primary_emotion] = counts.get(s.primary_emotion, 0) + 1

        if not counts:
            return None

        dominant_emotion, count = max(counts.items(), key=lambda x: x[1])
        if count / len(recent) > 0.5:
            return dominant_emotion
        return None

    def stats(self) -> dict:
        history = list(self._history)
        if not history:
            return {"history_size": 0}

        intensities = [s.intensity for s in history]
        emotions = [s.primary_emotion for s in history]
        emotion_counts: dict[str, int] = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        trend = self.emotional_trend()

        return {
            "history_size": len(history),
            "avg_intensity": round(sum(intensities) / len(intensities), 3),
            "max_intensity": round(max(intensities), 3),
            "emotion_distribution": emotion_counts,
            "dominant_pattern": self.dominant_pattern(),
            "current_pad": str(self.current_pad()),
            "trend": trend,
        }

    def explain(self, text: str) -> str:
        """Debug helper: score and explain without side effects."""
        vec = self._embedder.encode(text)
        breakdown = self._scorer.explain(vec)
        state = self._scorer.score(vec, source_text=text)
        return f"{breakdown}\n\nFinal: {state}"