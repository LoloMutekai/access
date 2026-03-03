"""
A.C.C.E.S.S. — Emotion Engine Configuration

Single source of truth for all emotion-related thresholds and parameters.
No magic numbers anywhere else in the emotion module.
"""

from dataclasses import dataclass, field


@dataclass
class EmotionConfig:
    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Scoring ────────────────────────────────────────────────────────────
    softmax_temperature: float = 0.3
    min_raw_score: float = 0.0
    min_confidence_threshold: float = 0.15

    # ── Intensity thresholds ───────────────────────────────────────────────
    intensity_threshold_low: float = 0.45
    intensity_threshold_high: float = 0.70

    # ── Memory modulation ──────────────────────────────────────────────────
    emotional_memory_base_importance: float = 0.55
    emotional_memory_intensity_factor: float = 0.35
    positive_emotion_boost: float = 0.05
    negative_emotion_penalty: float = -0.03
    modulation_lookback: int = 3

    # ── History ────────────────────────────────────────────────────────────
    history_max_size: int = 50

    # ── Emotion taxonomy ───────────────────────────────────────────────────
    positive_emotions: tuple = ("confidence", "drive", "flow")
    negative_emotions: tuple = ("frustration", "doubt", "fatigue")

    # ── Momentum model (NEW) ───────────────────────────────────────────────
    # Number of recent states to include in trend computation
    momentum_window: int = 10
    # Minimum |slope| to qualify as a meaningful trend (below = "stable")
    momentum_slope_threshold: float = 0.05
    # Valence slope below this → burnout_risk = True
    burnout_slope_threshold: float = -0.08
    # Arousal slope above this (+ valence rising) → mania_risk = True
    mania_slope_threshold: float = 0.10

    # ── Emotional protection (NEW) ─────────────────────────────────────────
    # Min recent interactions with same negative/positive emotion to trigger protection
    protection_repeat_count: int = 3
    # How many recent memories to scan for protection tagging
    protection_lookback_memories: int = 5

    # ── Alignment / emotion-aware retrieval (NEW) ─────────────────────────
    # Intensity above which negative state is considered "fragile"
    alignment_fragile_threshold: float = 0.6
    # Alignment penalty for stress memories in fragile state
    alignment_stress_suppression: float = 0.25
    # Alignment boost for positive memories in a negative state
    alignment_positive_boost: float = 0.20