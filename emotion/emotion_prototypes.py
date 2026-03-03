"""
A.C.C.E.S.S. — Emotion Prototypes

Key design decision: multi-phrase prototypes.

Instead of one sentence per emotion, we use 3 anchor phrases and average their
embeddings. This dramatically improves robustness — a single phrase can be
too narrow or accidentally close to another emotion's embedding.

The averaged centroid is a much more stable representative of the emotional
concept. This is the same technique used in few-shot classification with
sentence embeddings.

Adding new emotions: just add an entry to EMOTION_PROTOTYPES.
The system will pick it up automatically at next instantiation.
"""

import logging
from typing import Dict, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# ── Prototype definitions ──────────────────────────────────────────────────
# Each emotion has 3 anchor phrases. More phrases = more stable centroid.
# These are carefully chosen to be semantically distinct from each other.

EMOTION_PROTOTYPES: Dict[str, list[str]] = {
    "frustration": [
        "I feel stuck and nothing works no matter what I try.",
        "This is so annoying, I keep failing and I don't understand why.",
        "I am blocked and frustrated, I can't make any progress.",
    ],
    "flow": [
        "I am fully focused and everything is clicking into place.",
        "I am in the zone, time flies and the work feels effortless.",
        "Everything is clear and I am completely absorbed in what I am doing.",
    ],
    "fatigue": [
        "I feel mentally exhausted and completely drained.",
        "My brain is slow and I cannot think clearly anymore.",
        "I am tired and struggling to concentrate on anything.",
    ],
    "confidence": [
        "I feel powerful, capable, and ready to take on any challenge.",
        "I know exactly what I am doing and I trust my abilities completely.",
        "I feel strong and certain that I will succeed.",
    ],
    "doubt": [
        "I am unsure about this and I keep second-guessing every decision.",
        "I don't know if I am on the right track, everything feels uncertain.",
        "I hesitate and I am not confident in my choices.",
    ],
    "drive": [
        "I am determined and highly motivated to push through and achieve my goals.",
        "I feel ambitious and energized, nothing will stop me from making progress.",
        "I am focused on my mission and ready to work as hard as it takes.",
    ],
}


class EmbedderProtocol(Protocol):
    """Protocol for any object that can embed text — real or fake."""
    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        ...


class EmotionPrototypes:
    """
    Builds and stores centroid embeddings for each emotion.

    Lifecycle:
    1. Instantiate with an embedder
    2. Call build() — encodes all phrases, averages per emotion, normalizes
    3. Access via .prototypes dict or .get(emotion_name)

    Thread-safe after build() completes (read-only after that).
    """

    def __init__(self, embedder: EmbedderProtocol, phrases: Dict[str, list[str]] = None):
        self._embedder = embedder
        self._phrases = phrases or EMOTION_PROTOTYPES
        self._prototypes: Dict[str, np.ndarray] = {}
        self._built = False

    def build(self) -> None:
        """
        Encode all prototype phrases and compute per-emotion centroids.
        Call once at startup — takes ~0.5s with real model, instant with fake.
        """
        logger.info(f"Building emotion prototypes for {len(self._phrases)} emotions...")

        for emotion, phrases in self._phrases.items():
            vectors = []
            for phrase in phrases:
                vec = self._embedder.encode(phrase, normalize_embeddings=True)
                vectors.append(vec)

            # Average embeddings → centroid
            centroid = np.mean(vectors, axis=0).astype(np.float32)

            # Re-normalize the centroid (averaging denormalizes)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            self._prototypes[emotion] = centroid
            logger.debug(f"  Built prototype: {emotion} ({len(phrases)} phrases)")

        self._built = True
        logger.info(f"Emotion prototypes ready — {len(self._prototypes)} emotions")

    @property
    def prototypes(self) -> Dict[str, np.ndarray]:
        if not self._built:
            raise RuntimeError("Call build() before accessing prototypes.")
        return self._prototypes

    @property
    def emotion_names(self) -> list[str]:
        return list(self._phrases.keys())

    def get(self, emotion_name: str) -> np.ndarray:
        if emotion_name not in self._prototypes:
            raise KeyError(f"Unknown emotion: '{emotion_name}'. Known: {self.emotion_names}")
        return self._prototypes[emotion_name]

    def add_emotion(self, name: str, phrases: list[str]) -> None:
        """
        Register a new emotion at runtime.
        Useful for future emotion expansion without restarting.
        """
        if len(phrases) < 1:
            raise ValueError("Need at least 1 phrase per emotion.")
        self._phrases[name] = phrases
        if self._built:
            # Build just this new emotion
            vectors = [self._embedder.encode(p, normalize_embeddings=True) for p in phrases]
            centroid = np.mean(vectors, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            self._prototypes[name] = centroid / norm if norm > 0 else centroid
            logger.info(f"Added new emotion prototype: {name}")

    def __len__(self) -> int:
        return len(self._phrases)