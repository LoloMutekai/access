"""
A.C.C.E.S.S. — Prompt Builder

Translates ResponseModulation into a structured system prompt for the LLM.

Architecture: Section Assembler Pattern
────────────────────────────────────────
The system prompt is assembled from independent sections:

    [personality_prefix]  ← Phase 2 hook
    [tone]                ← always present
    [pacing]              ← always present
    [verbosity]           ← always present
    [structure_bias]      ← always present
    [validation]          ← if emotional_validation=True
    [motivational_bias]   ← if bias is non-neutral
    [cognitive_load]      ← if load < low_threshold or > high_threshold
    [memory_context]      ← if memory_context string provided
    [goal_suffix]         ← Phase 2 hook
    [trajectory_suffix]   ← Phase 3 hook

Each section is a pure function: (modulation, config) → str | None.
None means "do not include this section".

This makes the system prompt:
    - Machine-precise (not poetic)
    - Compact (each section is one sentence or two)
    - Auditable (sections list tells exactly what fired)
    - Testable (inspect sections without string-parsing)
    - Extensible (add section = add one function + register it)

Boundaries:
    - NO emotion logic (that is ConversationModulator's job)
    - NO LLM calls
    - NO database access
    - Pure input → output transformation

Usage:
    builder = PromptBuilder()
    prompt = builder.build(
        user_input="I feel stuck on this task.",
        modulation=modulation,
        memory_context="Past event: ...",
    )
    response = llm_client.chat(prompt.to_api_messages())
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .models import BuiltPrompt, PromptContext
from .prompt_config import PromptConfig

logger = logging.getLogger(__name__)

# Type alias for a section builder function
SectionFn = Callable[[object, Optional[str], PromptConfig], Optional[str]]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS — pure functions, no side effects
# Each returns a string to include, or None to skip.
# Signature: (modulation, memory_context, config) → str | None
# ─────────────────────────────────────────────────────────────────────────────

def _section_personality(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    """Phase 2 hook: personality prefix (empty by default)."""
    return cfg.personality_prefix.strip() or None


def _section_tone(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    tone = getattr(mod, "tone", "neutral")
    return cfg.tone_instructions.get(tone, cfg.tone_instructions["neutral"])


def _section_pacing(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    pacing = getattr(mod, "pacing", "normal")
    return cfg.pacing_instructions.get(pacing, cfg.pacing_instructions["normal"])


def _section_verbosity(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    verbosity = getattr(mod, "verbosity", "normal")
    return cfg.verbosity_instructions.get(verbosity, cfg.verbosity_instructions["normal"])


def _section_structure(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    bias = getattr(mod, "structure_bias", "conversational")
    return cfg.structure_instructions.get(bias, cfg.structure_instructions["conversational"])


def _section_validation(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    if getattr(mod, "emotional_validation", False):
        return cfg.validation_instruction
    return None


def _section_motivational_bias(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    bias = getattr(mod, "motivational_bias", 0.0)
    if bias <= cfg.motivational_low_threshold:
        return cfg.motivational_low_instruction
    if bias >= cfg.motivational_high_threshold:
        return cfg.motivational_high_instruction
    # Neutral range → no instruction added
    return None


def _section_cognitive_load(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    load = getattr(mod, "cognitive_load_limit", 1.0)
    if load < cfg.cognitive_low_threshold:
        return cfg.cognitive_low_instruction
    if load > cfg.cognitive_high_threshold:
        return cfg.cognitive_high_instruction
    return None


def _section_memory(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    if not memory_ctx:
        return None
    return f"{cfg.memory_context_header}\n{memory_ctx.strip()}"


def _section_goal(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    """Phase 2 hook: goal awareness suffix."""
    return cfg.goal_suffix.strip() or None


def _section_trajectory(mod, memory_ctx: Optional[str], cfg: PromptConfig) -> Optional[str]:
    """Phase 3 hook: long-term trajectory suffix."""
    return cfg.trajectory_suffix.strip() or None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION REGISTRY — ordered list of (label, fn) pairs
# Order determines prompt structure. Add new sections here only.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SECTIONS: list[tuple[str, SectionFn]] = [
    ("personality",       _section_personality),
    ("tone",              _section_tone),
    ("pacing",            _section_pacing),
    ("verbosity",         _section_verbosity),
    ("structure_bias",    _section_structure),
    ("validation",        _section_validation),
    ("motivational_bias", _section_motivational_bias),
    ("cognitive_load",    _section_cognitive_load),
    ("memory_context",    _section_memory),
    ("goal",              _section_goal),
    ("trajectory",        _section_trajectory),
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles LLM messages from a ResponseModulation + user input.

    Produces:
        [
            {"role": "system", "content": "<assembled system prompt>"},
            {"role": "user",   "content": "<user_input>"},
        ]

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(
            user_input="I feel completely stuck.",
            modulation=modulation,
            memory_context="Previous session: user was frustrated with project X.",
        )
        messages = prompt.to_api_messages()

    Extending (open/closed principle):
        Pass extra_sections to inject new sections at a specific position:

        def my_section(mod, memory_ctx, cfg):
            return "Always respond in French."  # example

        builder = PromptBuilder(
            extra_sections=[("language", my_section)]
        )

    The extra sections are appended after the default set.
    To control ordering precisely, build a custom DEFAULT_SECTIONS list and pass it
    as sections parameter.
    """

    def __init__(
        self,
        config: Optional[PromptConfig] = None,
        sections: Optional[list[tuple[str, SectionFn]]] = None,
        extra_sections: Optional[list[tuple[str, SectionFn]]] = None,
    ):
        self.config = config or PromptConfig()
        self._sections = (sections or DEFAULT_SECTIONS) + (extra_sections or [])

        logger.info(
            f"PromptBuilder ready — "
            f"{len(self._sections)} sections: "
            f"{[name for name, _ in self._sections]}"
        )

    def build(
        self,
        user_input: str,
        modulation: object,             # ResponseModulation (duck-typed)
        memory_context: Optional[str] = None,
    ) -> BuiltPrompt:
        """
        Build a BuiltPrompt from modulation + user input.

        Args:
            user_input:      Raw user message.
            modulation:      ResponseModulation from ConversationModulator.
            memory_context:  Optional formatted string from MemoryManager retrieval.

        Returns:
            BuiltPrompt — immutable, contains messages + audit metadata.
        """
        ctx = PromptContext(
            user_input=user_input,
            modulation=modulation,
            memory_context=memory_context,
        )
        return self._assemble(ctx)

    def build_from_context(self, ctx: PromptContext) -> BuiltPrompt:
        """
        Alternative entry point: build from a pre-constructed PromptContext.
        Useful when callers want to pass the full context as a value object.
        """
        return self._assemble(ctx)

    def _assemble(self, ctx: PromptContext) -> BuiltPrompt:
        """Run all sections, collect non-None outputs, build the system prompt."""
        lines: list[str] = []
        active_sections: list[str] = []

        for section_name, fn in self._sections:
            try:
                result = fn(ctx.modulation, ctx.memory_context, self.config)
                if result is not None and result.strip():
                    lines.append(result.strip())
                    active_sections.append(section_name)
            except Exception as exc:
                # Never crash prompt assembly due to a faulty section
                logger.error(
                    f"Section [{section_name}] raised and was skipped: {exc}",
                    exc_info=True,
                )

        system_prompt = "\n".join(lines)

        # Enforce compactness cap
        if len(system_prompt) > self.config.max_system_prompt_chars:
            system_prompt = system_prompt[: self.config.max_system_prompt_chars]
            logger.warning(
                f"System prompt truncated to {self.config.max_system_prompt_chars} chars"
            )

        messages = (
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ctx.user_input},
        )

        built = BuiltPrompt(
            messages=messages,
            system_prompt=system_prompt,
            sections=tuple(active_sections),
            char_count=len(system_prompt),
        )

        logger.info(
            f"Prompt built — "
            f"sections={active_sections}, "
            f"chars={built.char_count}"
        )

        return built

    def explain(
        self,
        user_input: str,
        modulation: object,
        memory_context: Optional[str] = None,
    ) -> str:
        """Debug helper: show which sections fired and their content."""
        lines = ["PromptBuilder — section trace:"]
        for section_name, fn in self._sections:
            try:
                result = fn(modulation, memory_context, self.config)
                if result is not None and result.strip():
                    lines.append(f"\n  [{section_name}] ✅")
                    for l in result.strip().split("\n"):
                        lines.append(f"    {l}")
                else:
                    lines.append(f"  [{section_name}] ⬜ (skipped)")
            except Exception as exc:
                lines.append(f"  [{section_name}] ❌ ERROR: {exc}")

        prompt = self.build(user_input, modulation, memory_context)
        lines.append(f"\nFinal: {prompt}")
        return "\n".join(lines)

    @property
    def section_names(self) -> list[str]:
        return [name for name, _ in self._sections]