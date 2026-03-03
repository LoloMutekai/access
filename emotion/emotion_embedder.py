"""
A.C.C.E.S.S. — Emotion Embedder

Thin wrapper around a sentence encoder.
Zero business logic — just encode and normalize.

Design: dependency injection.
In production → inject the real SentenceTransformer.
In tests → inject FakeEmbedder.
This avoids loading the model twice (MemoryManager already holds one instance).
"""

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        ...


class EmotionEmbedder:
    """
    Encapsulates text → embedding conversion for the emotion module.

    Accepts any object implementing EmbedderProtocol — real or fake.
    This is the ONLY place in the emotion module that touches the raw embedder.
    """

    def __init__(self, embedder: EmbedderProtocol):
        self._embedder = embedder

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to a normalized float32 vector.
        Always returns a unit vector (L2 norm = 1.0).
        """
        vec = self._embedder.encode(text, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)

        # Extra safety normalization
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @classmethod
    def from_sentence_transformers(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmotionEmbedder":
        """
        Factory method for production use.
        Creates a real SentenceTransformer instance.
        
        Prefer injecting the existing MemoryManager embedder when possible
        to avoid loading the model twice.
        """
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"EmotionEmbedder loaded model: {model_name}")
        return cls(model)