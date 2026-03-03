"""
A.C.C.E.S.S. — Conversation Layer

Translates EmotionEngine output into style instructions for the LLM prompt layer.

Quick usage:
    from conversation import ConversationModulator

    modulator = ConversationModulator()
    modulation = modulator.build_modulation(
        state=engine.process_interaction(user_text),
        trend=engine.emotional_trend(),
        dominant_pattern=engine.dominant_pattern(),
    )
    print(modulation.tone, modulation.to_dict())

Extending with a custom strategy:
    from conversation.conversation_modulation import ToneStrategy

    class MidnightStrategy(ToneStrategy):
        priority = 5
        name = "midnight_calm"
        def matches(self, state, trend, pattern): return state.primary_emotion == "fatigue"
        def apply(self, builder, state, trend, pattern, config):
            builder.pacing = "slow"
            builder.verbosity = "concise"

    modulator = ConversationModulator(extra_strategies=[MidnightStrategy()])
"""

from .conversation_modulation import ConversationModulator, ToneStrategy, DEFAULT_STRATEGIES
from .conversation_config import ConversationConfig
from .models import ResponseModulation, ModulationBuilder

__all__ = [
    "ConversationModulator",
    "ToneStrategy",
    "DEFAULT_STRATEGIES",
    "ConversationConfig",
    "ResponseModulation",
    "ModulationBuilder",
]