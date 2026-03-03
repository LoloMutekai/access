"""
A.C.C.E.S.S. — Conversation Configuration

Single source of truth for all conversation modulation thresholds.
No hard-coded values anywhere in the conversation module.

Design principles:
- Every threshold is named and documented
- Defaults are carefully calibrated but fully overridable
- Grouping by concern (PAD / trend / risk / pattern / output defaults)
"""

from dataclasses import dataclass


@dataclass
class ConversationConfig:
    # ── Output defaults ────────────────────────────────────────────────────
    # Starting values before any strategy fires
    default_tone: str = "neutral"
    default_pacing: str = "normal"
    default_verbosity: str = "normal"
    default_structure_bias: str = "conversational"
    default_emotional_validation: bool = False
    default_motivational_bias: float = 0.0
    default_cognitive_load_limit: float = 1.0

    # ── PAD thresholds ─────────────────────────────────────────────────────
    # Valence: [-1, +1]
    pad_negative_valence_threshold: float = -0.1   # below this = negative state
    pad_positive_valence_threshold: float = 0.1    # above this = positive state

    # Arousal: [0, 1]
    pad_high_arousal_threshold: float = 0.6        # above this = high arousal

    # Dominance: [0, 1]
    pad_low_dominance_threshold: float = 0.35      # below this = low dominance
    pad_high_dominance_threshold: float = 0.65     # above this = high dominance

    # ── Trend thresholds ───────────────────────────────────────────────────
    # These match against valence_slope from emotional_trend()
    trend_declining_slope: float = -0.03   # below this = meaningful decline
    trend_improving_slope: float = 0.03    # above this = meaningful improvement

    # ── Risk flag outputs ─────────────────────────────────────────────────
    # Burnout
    burnout_tone: str = "calm"
    burnout_pacing: str = "slow"
    burnout_verbosity: str = "concise"
    burnout_structure_bias: str = "conversational"
    burnout_emotional_validation: bool = True
    burnout_cognitive_load: float = 0.35
    burnout_motivational_bias: float = 0.1         # small push, not aggressive

    # Mania
    mania_tone: str = "grounding"
    mania_pacing: str = "normal"
    mania_verbosity: str = "concise"
    mania_structure_bias: str = "structured"
    mania_emotional_validation: bool = False
    mania_motivational_bias: float = -0.25         # slight brake on momentum
    mania_cognitive_load: float = 0.7              # keep it manageable

    # ── Trend outputs ──────────────────────────────────────────────────────
    # Declining trend
    declining_tone: str = "reassuring"
    declining_verbosity: str = "normal"
    declining_emotional_validation: bool = True
    declining_motivational_bias: float = 0.15      # gentle encouragement

    # Improving / positive momentum
    improving_tone: str = "energizing"
    improving_pacing: str = "fast"
    improving_motivational_bias: float = 0.4
    improving_verbosity: str = "normal"

    # ── Dominant pattern outputs ───────────────────────────────────────────
    # Sustained frustration
    frustration_pattern: str = "frustration"       # emotion label to match
    frustration_cognitive_load: float = 0.5
    frustration_structure_bias: str = "structured"
    frustration_motivational_bias: float = 0.1

    # Sustained confidence
    confidence_pattern: str = "confidence"         # emotion label to match
    confidence_motivational_bias: float = 0.55     # push harder
    confidence_cognitive_load: float = 1.0         # full complexity OK
    confidence_emotional_validation: bool = False   # less hand-holding

    # ── PAD-derived tone outputs ───────────────────────────────────────────
    # Low valence + high arousal → grounding (agitated negative state)
    pad_grounding_tone: str = "grounding"
    pad_grounding_pacing: str = "slow"

    # High valence + high dominance → challenging (empowered positive state)
    pad_challenging_tone: str = "challenging"
    pad_challenging_motivational_bias: float = 0.5

    # Low dominance → supportive (user feels out of control)
    pad_supportive_tone: str = "supportive"
    pad_supportive_emotional_validation: bool = True
    pad_supportive_motivational_bias: float = 0.2