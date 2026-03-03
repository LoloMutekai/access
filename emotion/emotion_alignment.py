"""
A.C.C.E.S.S. — Emotional Alignment

Computes how aligned a stored memory is with the user's current emotional state.
This modulates retrieval relevance so memory recall is context-sensitive.

New retrieval formula:
    relevance = similarity × importance × emotional_alignment

Alignment logic:
    1. Memory has no emotion tags → neutral (1.0, no modulation)
    2. Memory has emotion tags → compare its prototype vector to current state vector
    3. If current state is negative → boost positive memories, suppress stress memories
    4. Fragile state (strong negative) → extra suppression of stress-tagged memories

Architecture note:
    This module is PURE — no DB access, no side effects.
    MemoryManager receives an alignment_fn callable (injected by EmotionEngine)
    so there is ZERO circular dependency between emotion ↔ memory packages.

    EmotionEngine registers:  memory_manager.set_emotion_alignment(aligner.compute)
    MemoryManager calls:      alignment_fn(record, emotional_context)
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# All known emotion labels — used to detect emotion tags on memory records
KNOWN_EMOTION_TAGS: frozenset = frozenset({
    "frustration", "flow", "fatigue", "confidence", "doubt", "drive"
})

# Valence categories for rule-based modulation
POSITIVE_EMOTIONS: frozenset = frozenset({"confidence", "drive", "flow"})
NEGATIVE_EMOTIONS: frozenset = frozenset({"frustration", "doubt", "fatigue"})


class EmotionAlignment:
    """
    Computes emotional alignment between a memory record and current emotional state.

    Injected into MemoryManager via set_emotion_alignment().
    Called during retrieval as: alignment_fn(record, state) → float [0.0, 1.0]
    """

    def __init__(
        self,
        prototypes: Dict[str, np.ndarray],
        fragile_threshold: float = 0.6,
        stress_suppression: float = 0.25,
        positive_boost: float = 0.20,
    ):
        """
        Args:
            prototypes: dict of emotion_name → unit vector (from EmotionPrototypes)
            fragile_threshold: intensity above which state is "fragile" (extra suppression)
            stress_suppression: penalty subtracted from alignment for stress memories in fragile state
            positive_boost: bonus added for positive memories when current state is negative
        """
        self._prototypes = prototypes
        self._fragile_threshold = fragile_threshold
        self._stress_suppression = stress_suppression
        self._positive_boost = positive_boost

    def compute(self, memory_record, current_state) -> float:
        """
        Compute alignment score [0.0, 1.0] for a memory given current emotional state.

        Returns 1.0 (neutral) if:
        - memory has no emotion tags
        - current_state is None
        - emotion tag not in known prototypes

        Args:
            memory_record: MemoryRecord (duck-typed — no direct import)
            current_state: EmotionalState (duck-typed — no direct import)

        Returns:
            float in [0.0, 1.0] — 1.0 = perfectly aligned, 0.0 = strongly misaligned
        """
        if current_state is None:
            return 1.0

        tags = getattr(memory_record, 'tags', []) or []
        emotion_tags = [t for t in tags if t in KNOWN_EMOTION_TAGS]

        if not emotion_tags:
            return 1.0  # no emotional color on this memory → neutral

        memory_emotion = emotion_tags[0]  # primary emotion tag

        # ── Base alignment: prototype vector cosine similarity ─────────────
        alignment = self._vector_similarity(memory_emotion, current_state)

        # ── Rule 1: negative current state → favor positive memories ──────
        is_negative = getattr(current_state, 'is_negative', False)
        if is_negative and memory_emotion in POSITIVE_EMOTIONS:
            alignment = min(1.0, alignment + self._positive_boost)

        # ── Rule 2: fragile state → suppress stress memories ──────────────
        intensity = getattr(current_state, 'intensity', 0.0)
        if (is_negative
                and intensity >= self._fragile_threshold
                and memory_emotion in NEGATIVE_EMOTIONS):
            alignment = max(0.0, alignment - self._stress_suppression)

        return round(alignment, 4)

    def _vector_similarity(self, memory_emotion: str, current_state) -> float:
        """
        Cosine similarity between memory's emotion prototype and current emotion vector.
        Maps from [-1, 1] to [0, 1] (negative similarity → 0, not negative alignment).
        """
        proto = self._prototypes.get(memory_emotion)
        cur_vec = getattr(current_state, 'emotion_vector', None)

        if proto is None or cur_vec is None:
            return 0.5  # unknown → neutral-ish

        raw = float(np.dot(proto, cur_vec))
        # Map cosine [-1,+1] → [0,1]
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def explain(self, memory_record, current_state) -> str:
        """Human-readable alignment breakdown for debugging."""
        tags = getattr(memory_record, 'tags', []) or []
        emotion_tags = [t for t in tags if t in KNOWN_EMOTION_TAGS]
        score = self.compute(memory_record, current_state)

        if not emotion_tags:
            return f"Alignment: 1.0 (no emotion tags on memory)"

        memory_emotion = emotion_tags[0]
        vec_sim = self._vector_similarity(memory_emotion, current_state)
        is_neg = getattr(current_state, 'is_negative', False)
        intensity = getattr(current_state, 'intensity', 0.0)

        lines = [
            f"Memory emotion  : {memory_emotion}",
            f"Current emotion : {getattr(current_state, 'primary_emotion', '?')}",
            f"Vector sim      : {vec_sim:.3f}",
            f"Is negative     : {is_neg}",
            f"Intensity       : {intensity:.3f}",
            f"Final alignment : {score:.3f}",
        ]
        return "\n".join(lines)