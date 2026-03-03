"""
A.C.C.E.S.S. — Conversation Modulation Engine

Translates EmotionEngine output into style instructions for the LLM prompt layer.

Architecture: Priority-Ordered Strategy Chain
─────────────────────────────────────────────
Each rule is a ToneStrategy with three responsibilities:
    1. matches()  — decides whether this rule should fire
    2. apply()    — writes style adjustments into ModulationBuilder
    3. priority   — determines override order (higher = overrides lower)

Priority levels:
    0   — PAD-derived base tone (foundation, lowest authority)
    10  — Dominant pattern adjustments (sustained emotion override)
    20  — Trend-based modulation (trajectory override)
    30  — Risk flags: burnout / mania (always override, highest authority)

Open/Closed principle:
    Adding a new strategy = create a new ToneStrategy subclass + register it.
    No changes to ConversationModulator or existing strategies.

    Custom strategies are injected at construction:
        modulator = ConversationModulator(config, extra_strategies=[MyStrategy()])

Separation of concerns:
    - No LLM calls, no DB access, no EmotionEngine logic
    - Accepts duck-typed inputs (EmotionalState, trend dict)
    - Fully pure and testable with minimal setup
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from .conversation_config import ConversationConfig
from .models import ModulationBuilder, ResponseModulation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY BASE
# ─────────────────────────────────────────────────────────────────────────────

class ToneStrategy(ABC):
    """
    Abstract base for all modulation rules.

    Subclass this to add new response strategies.
    No changes to ConversationModulator required.
    """

    #: Override order — higher priority fires last and wins.
    priority: int = 0

    #: Human-readable name recorded in ResponseModulation.active_strategies.
    name: str = "unnamed_strategy"

    @abstractmethod
    def matches(
        self,
        state,                  # EmotionalState (duck-typed)
        trend: dict,
        dominant_pattern: Optional[str],
    ) -> bool:
        """Return True if this strategy should apply."""
        ...

    @abstractmethod
    def apply(
        self,
        builder: ModulationBuilder,
        state,                  # EmotionalState (duck-typed)
        trend: dict,
        dominant_pattern: Optional[str],
        config: ConversationConfig,
    ) -> None:
        """Write modulation adjustments into builder."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE STRATEGIES — priority 0: PAD base mapping
# ─────────────────────────────────────────────────────────────────────────────

class PADGroundingStrategy(ToneStrategy):
    """
    Low valence + high arousal → user is agitated and negative.
    Apply grounding tone to stabilize before anything else.
    """
    priority = 0
    name = "pad_grounding"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        valence = getattr(getattr(state, "pad", None), "valence", 0.0)
        arousal = getattr(getattr(state, "pad", None), "arousal", 0.5)
        return valence < 0.0 and arousal > 0.6  # use hard boundaries here; thresholds applied in config checks

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.pad_grounding_tone
        builder.pacing = config.pad_grounding_pacing


class PADChallengingStrategy(ToneStrategy):
    """
    High valence + high dominance → user is in a powerful, capable state.
    Raise the bar: use challenging tone.
    """
    priority = 0
    name = "pad_challenging"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        pad = getattr(state, "pad", None)
        if pad is None:
            return False
        return pad.valence > 0.1 and pad.dominance > 0.65

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.pad_challenging_tone
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.pad_challenging_motivational_bias,
        )


class PADSupportiveStrategy(ToneStrategy):
    """
    Low dominance → user feels out of control or helpless.
    Provide scaffolding and validation.
    """
    priority = 0
    name = "pad_supportive"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        dominance = getattr(getattr(state, "pad", None), "dominance", 0.5)
        return dominance < 0.35

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.pad_supportive_tone
        builder.emotional_validation = config.pad_supportive_emotional_validation
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.pad_supportive_motivational_bias,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE STRATEGIES — priority 10: Dominant pattern
# ─────────────────────────────────────────────────────────────────────────────

class SustainedFrustrationStrategy(ToneStrategy):
    """
    Prolonged frustration → reduce cognitive load, increase structure.
    User needs clarity and smaller steps, not more information.
    """
    priority = 10
    name = "pattern_frustration"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        return dominant_pattern == "frustration"

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.cognitive_load_limit = min(
            builder.cognitive_load_limit,
            config.frustration_cognitive_load,
        )
        builder.structure_bias = config.frustration_structure_bias
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.frustration_motivational_bias,
        )


class SustainedConfidenceStrategy(ToneStrategy):
    """
    Prolonged confidence → raise challenge, reduce hand-holding.
    User is capable and bored of easy tasks.
    """
    priority = 10
    name = "pattern_confidence"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        return dominant_pattern == "confidence"

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.confidence_motivational_bias,
        )
        builder.cognitive_load_limit = config.confidence_cognitive_load
        builder.emotional_validation = config.confidence_emotional_validation


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE STRATEGIES — priority 20: Trend-based
# ─────────────────────────────────────────────────────────────────────────────

class DecliningTrendStrategy(ToneStrategy):
    """
    Valence declining over recent history → user is getting worse.
    Shift to reassuring tone with gentle validation.
    """
    priority = 20
    name = "trend_declining"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        label = trend.get("trend_label", "stable")
        slope = trend.get("valence_slope", 0.0)
        # Covers both "declining" and "escalating" (negative + rising arousal)
        return label in ("declining", "escalating") or slope < -0.03

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.declining_tone
        builder.verbosity = config.declining_verbosity
        builder.emotional_validation = config.declining_emotional_validation
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.declining_motivational_bias,
        )


class ImprovingTrendStrategy(ToneStrategy):
    """
    Valence improving over recent history → user is getting better.
    Amplify momentum with energizing tone.
    """
    priority = 20
    name = "trend_improving"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        label = trend.get("trend_label", "stable")
        slope = trend.get("valence_slope", 0.0)
        return label == "improving" or slope > 0.03

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.improving_tone
        builder.pacing = config.improving_pacing
        builder.verbosity = config.improving_verbosity
        builder.motivational_bias = max(
            builder.motivational_bias,
            config.improving_motivational_bias,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE STRATEGIES — priority 30: Risk flags (HIGHEST AUTHORITY)
# ─────────────────────────────────────────────────────────────────────────────

class BurnoutRiskStrategy(ToneStrategy):
    """
    Steep valence decline detected → burnout risk.
    Reduce everything: pressure, complexity, speed.
    Always overrides lower-priority strategies.
    """
    priority = 30
    name = "risk_burnout"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        return trend.get("burnout_risk", False)

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.burnout_tone
        builder.pacing = config.burnout_pacing
        builder.verbosity = config.burnout_verbosity
        builder.structure_bias = config.burnout_structure_bias
        builder.emotional_validation = config.burnout_emotional_validation
        builder.cognitive_load_limit = config.burnout_cognitive_load
        # Assign directly — don't use max() since burnout requires gentle positivity
        # even if another strategy pushed bias higher
        builder.motivational_bias = config.burnout_motivational_bias


class ManiaRiskStrategy(ToneStrategy):
    """
    Valence rising + arousal spiking → mania/overexcitement risk.
    Ground and structure the response; slightly dampen motivation.
    Always overrides lower-priority strategies.
    """
    priority = 30
    name = "risk_mania"

    def matches(self, state, trend: dict, dominant_pattern: Optional[str]) -> bool:
        return trend.get("mania_risk", False)

    def apply(self, builder, state, trend, dominant_pattern, config):
        builder.tone = config.mania_tone
        builder.pacing = config.mania_pacing
        builder.verbosity = config.mania_verbosity
        builder.structure_bias = config.mania_structure_bias
        builder.emotional_validation = config.mania_emotional_validation
        builder.cognitive_load_limit = config.mania_cognitive_load
        builder.motivational_bias = config.mania_motivational_bias


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGIES: list[ToneStrategy] = [
    # Priority 0 — PAD base mapping
    PADGroundingStrategy(),
    PADChallengingStrategy(),
    PADSupportiveStrategy(),
    # Priority 10 — Sustained patterns
    SustainedFrustrationStrategy(),
    SustainedConfidenceStrategy(),
    # Priority 20 — Trend direction
    DecliningTrendStrategy(),
    ImprovingTrendStrategy(),
    # Priority 30 — Risk overrides
    BurnoutRiskStrategy(),
    ManiaRiskStrategy(),
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE MODULATOR
# ─────────────────────────────────────────────────────────────────────────────

class ConversationModulator:
    """
    Orchestrates the strategy chain to produce a ResponseModulation.

    Usage:
        modulator = ConversationModulator()
        modulation = modulator.build_modulation(
            state=emotional_state,
            trend=engine.emotional_trend(),
            dominant_pattern=engine.dominant_pattern(),
        )
        print(modulation.tone, modulation.cognitive_load_limit)

    Extending with new strategies (open/closed principle):
        class MyCustomStrategy(ToneStrategy):
            priority = 15
            name = "my_custom"
            def matches(self, ...): ...
            def apply(self, ...): ...

        modulator = ConversationModulator(extra_strategies=[MyCustomStrategy()])
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        extra_strategies: Optional[list[ToneStrategy]] = None,
    ):
        self.config = config or ConversationConfig()

        # Combine defaults + custom strategies, sort by priority ascending
        # (low priority fires first, high priority fires last and wins)
        all_strategies = DEFAULT_STRATEGIES + (extra_strategies or [])
        self._strategies: list[ToneStrategy] = sorted(
            all_strategies, key=lambda s: s.priority
        )

        logger.info(
            f"ConversationModulator ready — "
            f"{len(self._strategies)} strategies, "
            f"priorities: {[s.priority for s in self._strategies]}"
        )

    def build_modulation(
        self,
        state,                          # EmotionalState (duck-typed)
        trend: dict,
        dominant_pattern: Optional[str] = None,
    ) -> ResponseModulation:
        """
        Run the full strategy chain and produce a ResponseModulation.

        Args:
            state:            EmotionalState from EmotionEngine.process_interaction()
            trend:            dict from EmotionEngine.emotional_trend()
            dominant_pattern: str|None from EmotionEngine.dominant_pattern()

        Returns:
            ResponseModulation — immutable style instructions for the LLM layer.
        """
        # Initialise builder with config defaults
        builder = self._make_builder()

        fired: list[str] = []

        # Run strategies in priority order (ascending = lowest first)
        for strategy in self._strategies:
            try:
                if strategy.matches(state, trend, dominant_pattern):
                    strategy.apply(builder, state, trend, dominant_pattern, self.config)
                    builder.record_strategy(strategy.name)
                    fired.append(strategy.name)
                    logger.debug(f"Strategy fired: [{strategy.name}] (priority={strategy.priority})")
            except Exception as exc:
                # Never crash the conversation layer due to a bad strategy
                logger.error(
                    f"Strategy [{strategy.name}] raised an exception and was skipped: {exc}",
                    exc_info=True,
                )

        modulation = builder.build()

        logger.info(
            f"Modulation built: tone={modulation.tone} | "
            f"pacing={modulation.pacing} | "
            f"verbosity={modulation.verbosity} | "
            f"load={modulation.cognitive_load_limit:.2f} | "
            f"strategies={fired}"
        )

        return modulation

    def _make_builder(self) -> ModulationBuilder:
        """Create a fresh ModulationBuilder from config defaults."""
        c = self.config
        return ModulationBuilder(
            tone=c.default_tone,
            pacing=c.default_pacing,
            verbosity=c.default_verbosity,
            structure_bias=c.default_structure_bias,
            emotional_validation=c.default_emotional_validation,
            motivational_bias=c.default_motivational_bias,
            cognitive_load_limit=c.default_cognitive_load_limit,
        )

    def explain(
        self,
        state,
        trend: dict,
        dominant_pattern: Optional[str] = None,
    ) -> str:
        """Debug helper: show which strategies matched and what they would set."""
        lines = ["ConversationModulator — strategy trace:"]
        for strategy in self._strategies:
            try:
                matched = strategy.matches(state, trend, dominant_pattern)
                lines.append(
                    f"  [{strategy.priority:2d}] {strategy.name:<30} "
                    f"{'✅ FIRED' if matched else '⬜ skip'}"
                )
            except Exception as exc:
                lines.append(f"  [{strategy.priority:2d}] {strategy.name:<30} ❌ ERROR: {exc}")
        modulation = self.build_modulation(state, trend, dominant_pattern)
        lines.append(f"\nResult: {modulation}")
        return "\n".join(lines)

    @property
    def strategy_names(self) -> list[str]:
        return [s.name for s in self._strategies]