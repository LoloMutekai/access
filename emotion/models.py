"""
A.C.C.E.S.S. — Emotion Engine Models

Design note on PAD dimensions:
    Valence   : positive ↔ negative affect [-1, +1]
    Arousal   : calm ↔ excited [0, 1]
    Dominance : submissive ↔ dominant [0, 1]

These are pre-populated from a static mapping per emotion for now.
Phase 2 will compute them dynamically from the full embedding space.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


# Static PAD coordinates per emotion (literature-based approximations)
# Format: (valence [-1,1], arousal [0,1], dominance [0,1])
EMOTION_PAD_MAP: Dict[str, tuple[float, float, float]] = {
    "frustration": (-0.65, 0.70, 0.30),
    "flow":        (+0.70, 0.55, 0.65),
    "fatigue":     (-0.40, 0.10, 0.20),
    "confidence":  (+0.75, 0.60, 0.85),
    "doubt":       (-0.45, 0.35, 0.20),
    "drive":       (+0.60, 0.80, 0.75),
}


@dataclass
class PADState:
    """
    Dimensional emotional coordinates (Pleasure-Arousal-Dominance).
    Enables cross-emotion blending and future continuous emotion modeling.
    """
    valence: float    # [-1.0, +1.0] — negative to positive affect
    arousal: float    # [0.0, 1.0]  — calm to excited
    dominance: float  # [0.0, 1.0]  — submissive to dominant

    @classmethod
    def from_emotion(cls, emotion_name: str) -> "PADState":
        v, a, d = EMOTION_PAD_MAP.get(emotion_name, (0.0, 0.5, 0.5))
        return cls(valence=v, arousal=a, dominance=d)

    @classmethod
    def neutral(cls) -> "PADState":
        return cls(valence=0.0, arousal=0.5, dominance=0.5)

    def blend(self, other: "PADState", weight: float = 0.5) -> "PADState":
        """Blend two PAD states. weight=1.0 → fully other."""
        w = max(0.0, min(1.0, weight))
        return PADState(
            valence=self.valence * (1 - w) + other.valence * w,
            arousal=self.arousal * (1 - w) + other.arousal * w,
            dominance=self.dominance * (1 - w) + other.dominance * w,
        )

    def __repr__(self) -> str:
        sign = "+" if self.valence >= 0 else ""
        return (
            f"PAD(V={sign}{self.valence:.2f}, "
            f"A={self.arousal:.2f}, "
            f"D={self.dominance:.2f})"
        )


@dataclass
class EmotionalState:
    """
    The complete emotional output for a single interaction.

    primary_emotion  : detected dominant emotion label
    intensity        : strength of the dominant emotion [0.0, 1.0]
    confidence       : certainty of detection (top1 - top2 softmax gap) [0.0, 1.0]
    emotion_vector   : normalized embedding of the input text
    pad              : PAD dimensional coordinates for this state
    raw_scores       : softmax probabilities per emotion
    detected_at      : timestamp of detection
    source_text      : original text that was analyzed (truncated for storage)
    """

    primary_emotion: str
    intensity: float
    confidence: float
    emotion_vector: np.ndarray
    pad: PADState
    raw_scores: Dict[str, float]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    source_text: str = ""

    def __post_init__(self):
        # Safety clamps
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.confidence = max(0.0, min(1.0, self.confidence))
        # Normalize vector if not already
        norm = np.linalg.norm(self.emotion_vector)
        if norm > 0:
            self.emotion_vector = self.emotion_vector / norm

    @property
    def is_positive(self) -> bool:
        return self.pad.valence > 0.1

    @property
    def is_negative(self) -> bool:
        return self.pad.valence < -0.1

    @property
    def is_high_arousal(self) -> bool:
        return self.pad.arousal > 0.6

    @property
    def label(self) -> str:
        """Human-readable label with intensity."""
        intensity_label = (
            "weak" if self.intensity < 0.4 else
            "moderate" if self.intensity < 0.65 else
            "strong"
        )
        return f"{intensity_label} {self.primary_emotion}"

    def to_log_dict(self) -> dict:
        """Compact dict for structured logging."""
        return {
            "emotion": self.primary_emotion,
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "valence": round(self.pad.valence, 3),
            "arousal": round(self.pad.arousal, 3),
            "dominance": round(self.pad.dominance, 3),
            "top_scores": {k: round(v, 3) for k, v in
                          sorted(self.raw_scores.items(), key=lambda x: -x[1])[:3]},
        }

    def __repr__(self) -> str:
        return (
            f"EmotionalState({self.label}, "
            f"confidence={self.confidence:.2f}, "
            f"{self.pad})"
        )