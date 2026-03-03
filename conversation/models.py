"""
A.C.C.E.S.S. — Conversation Layer Models

ResponseModulation is the contract between the Conversation Layer and the LLM prompt
system. It carries style instructions only — no text, no LLM calls.

Design:
    - Immutable after construction (frozen dataclass)
    - All fields are primitives → trivially serializable
    - active_strategies: audit trail of which rules fired (tupled for immutability)
    - ModulationBuilder: mutable accumulator used internally by strategies,
      then frozen into ResponseModulation via build()
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResponseModulation:
    """
    Immutable style instruction set for the LLM prompt layer.

    tone:
        calm         — soft, low-pressure, validating
        energizing   — upbeat, forward-looking, action-oriented
        grounding    — stable, factual, slowing things down
        challenging  — direct, high-expectation, stretch-goal oriented
        reassuring   — empathetic, normalizing, warm
        supportive   — gentle, building-up, scaffolding
        neutral      — default, no strong bias

    pacing:
        slow   — fewer suggestions, longer pauses, space to breathe
        normal — standard conversational tempo
        fast   — efficient, crisp, momentum-building

    verbosity:
        concise  — essentials only, no elaboration
        normal   — balanced explanation
        detailed — thorough, with context and rationale

    structure_bias:
        structured       — bullet points, steps, clear sections
        conversational   — flowing prose, natural dialogue

    motivational_bias:
        -1.0 = fully de-energizing (slow down)
         0.0 = neutral
        +1.0 = fully motivating (push forward)

    cognitive_load_limit:
         0.0 = minimal complexity (burnout / fragile state)
         1.0 = full complexity allowed
    """

    tone: str                       # calm / energizing / grounding / challenging / reassuring / supportive / neutral
    pacing: str                     # slow / normal / fast
    verbosity: str                  # concise / normal / detailed
    structure_bias: str             # structured / conversational
    emotional_validation: bool      # should response acknowledge the user's emotion?
    motivational_bias: float        # [-1.0, +1.0]
    cognitive_load_limit: float     # [0.0, 1.0]
    active_strategies: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """Compact representation for logging and prompt injection."""
        return {
            "tone": self.tone,
            "pacing": self.pacing,
            "verbosity": self.verbosity,
            "structure_bias": self.structure_bias,
            "emotional_validation": self.emotional_validation,
            "motivational_bias": round(self.motivational_bias, 3),
            "cognitive_load_limit": round(self.cognitive_load_limit, 3),
            "active_strategies": list(self.active_strategies),
        }

    def __repr__(self) -> str:
        bias_sign = "+" if self.motivational_bias >= 0 else ""
        return (
            f"ResponseModulation("
            f"tone={self.tone}, "
            f"pacing={self.pacing}, "
            f"verbosity={self.verbosity}, "
            f"structure={self.structure_bias}, "
            f"validation={self.emotional_validation}, "
            f"motivation={bias_sign}{self.motivational_bias:.2f}, "
            f"load_limit={self.cognitive_load_limit:.2f}, "
            f"strategies={list(self.active_strategies)})"
        )


class ModulationBuilder:
    """
    Mutable accumulator used during strategy application.
    Strategies write into this; ConversationModulator freezes it via build().

    Each field starts at its default value (from ConversationConfig).
    Strategies overwrite specific fields in priority order.
    """

    def __init__(
        self,
        tone: str,
        pacing: str,
        verbosity: str,
        structure_bias: str,
        emotional_validation: bool,
        motivational_bias: float,
        cognitive_load_limit: float,
    ):
        self.tone = tone
        self.pacing = pacing
        self.verbosity = verbosity
        self.structure_bias = structure_bias
        self.emotional_validation = emotional_validation
        self.motivational_bias = motivational_bias
        self.cognitive_load_limit = cognitive_load_limit
        self._active: list[str] = []

    def record_strategy(self, name: str) -> None:
        """Mark a strategy as having fired."""
        self._active.append(name)

    def clamp(self) -> None:
        """Safety clamps — called once before build()."""
        self.motivational_bias = max(-1.0, min(1.0, self.motivational_bias))
        self.cognitive_load_limit = max(0.0, min(1.0, self.cognitive_load_limit))

    def build(self) -> ResponseModulation:
        """Freeze into an immutable ResponseModulation."""
        self.clamp()
        return ResponseModulation(
            tone=self.tone,
            pacing=self.pacing,
            verbosity=self.verbosity,
            structure_bias=self.structure_bias,
            emotional_validation=self.emotional_validation,
            motivational_bias=self.motivational_bias,
            cognitive_load_limit=self.cognitive_load_limit,
            active_strategies=tuple(self._active),
        )