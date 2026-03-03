"""
A.C.C.E.S.S. — Emotion Scorer

Converts text embeddings into quantified emotional states.

Scoring pipeline:
1. Cosine similarity between input vector and each prototype centroid
2. Clip negatives (similarity below 0 = orthogonal/unrelated → ignore)
3. Temperature-scaled softmax → probability distribution over emotions
4. Extract: primary emotion, intensity, confidence (top1 - top2 gap)
5. Map to PAD coordinates
6. Produce EmotionalState

Why temperature-scaled softmax over simple argmax?
    - Gives a proper probability distribution (not just a winner)
    - Temperature controls sharpness: low T → decisive, high T → uncertain
    - The top1-top2 gap (confidence) becomes meaningful and calibrated
    - Enables future multi-label extension naturally

Why clip negatives?
    - Negative cosine similarity = the text is semantically OPPOSITE to the prototype
    - Including them would pollute the distribution
    - Emotion scoring should only reward relevance, not penalize opposition
"""

import logging
import math
from typing import Dict

import numpy as np

from .config import EmotionConfig
from .emotion_prototypes import EmotionPrototypes
from .models import EmotionalState, PADState

logger = logging.getLogger(__name__)


class EmotionScorer:
    """
    Pure scorer: takes an embedding, returns an EmotionalState.
    No side effects. Fully testable in isolation.
    """

    def __init__(self, prototypes: EmotionPrototypes, config: EmotionConfig):
        self._prototypes = prototypes
        self._config = config

    def score(self, text_vector: np.ndarray, source_text: str = "") -> EmotionalState:
        """
        Score an embedding against all emotion prototypes.

        Args:
            text_vector: normalized float32 embedding of input text
            source_text: original text (stored for logging/debugging)

        Returns:
            EmotionalState with all fields populated.
        """
        raw_similarities = self._compute_similarities(text_vector)
        softmax_probs = self._softmax(raw_similarities)

        primary_emotion, intensity = self._dominant(softmax_probs)
        confidence = self._confidence_gap(softmax_probs)
        pad = PADState.from_emotion(primary_emotion)

        state = EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            emotion_vector=text_vector.copy(),
            pad=pad,
            raw_scores=softmax_probs,
            source_text=source_text[:200],  # truncate for storage
        )

        logger.debug(f"EmotionScorer: {state}")
        return state

    def _compute_similarities(self, vec: np.ndarray) -> Dict[str, float]:
        """
        Cosine similarity between input and each prototype.
        Since both are unit vectors: dot product = cosine similarity.
        Negatives clipped to 0.
        """
        sims = {}
        for emotion, prototype in self._prototypes.prototypes.items():
            sim = float(np.dot(vec, prototype))
            sims[emotion] = max(self._config.min_raw_score, sim)
        return sims

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Temperature-scaled softmax over emotion scores.

        temperature < 1 → sharper peaks (confident decisions)
        temperature = 1 → standard softmax
        temperature > 1 → flatter distribution (more uncertain)
        """
        T = self._config.softmax_temperature
        emotions = list(scores.keys())
        values = np.array([scores[e] for e in emotions], dtype=np.float64)

        # Scale by temperature
        scaled = values / T

        # Numerically stable softmax
        scaled -= scaled.max()
        exp_vals = np.exp(scaled)
        probs = exp_vals / exp_vals.sum()

        return {e: float(p) for e, p in zip(emotions, probs)}

    def _dominant(self, probs: Dict[str, float]) -> tuple[str, float]:
        """
        Return (dominant_emotion, intensity).
        Intensity = probability of the dominant emotion (already [0,1]).
        """
        dominant = max(probs, key=probs.get)
        return dominant, probs[dominant]

    def _confidence_gap(self, probs: Dict[str, float]) -> float:
        """
        Confidence = gap between top-1 and top-2 probability.
        Large gap → clear winner → high confidence.
        Small gap → ambiguous → low confidence.
        """
        sorted_probs = sorted(probs.values(), reverse=True)
        if len(sorted_probs) < 2:
            return 1.0
        return max(0.0, sorted_probs[0] - sorted_probs[1])

    def explain(self, text_vector: np.ndarray) -> str:
        """
        Human-readable breakdown of scoring for debugging.
        """
        raw = self._compute_similarities(text_vector)
        probs = self._softmax(raw)
        ranked = sorted(probs.items(), key=lambda x: -x[1])

        lines = ["Emotion scores:"]
        for emotion, prob in ranked:
            bar = "█" * int(prob * 30)
            lines.append(f"  {emotion:<14} {prob:.3f}  {bar}")

        top, second = ranked[0], ranked[1]
        gap = top[1] - second[1]
        lines.append(f"\nDominant : {top[0]} (p={top[1]:.3f})")
        lines.append(f"Runner-up: {second[0]} (p={second[1]:.3f})")
        lines.append(f"Confidence gap: {gap:.3f}")
        return "\n".join(lines)