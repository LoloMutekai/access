"""
A.C.C.E.S.S. — Prompt Builder Layer

Translates ResponseModulation into structured LLM messages.

Pipeline position:
    ConversationModulator → ResponseModulation → PromptBuilder → list[dict] → LLM

Quick usage:
    from prompt import PromptBuilder

    builder = PromptBuilder()
    prompt = builder.build(
        user_input="I feel completely stuck on this.",
        modulation=modulation,                  # ResponseModulation from ConversationModulator
        memory_context="Past session: ...",     # optional, from MemoryManager
    )
    messages = prompt.to_api_messages()         # list[dict] → LLM client

Extending with a custom section:
    from prompt.prompt_builder import DEFAULT_SECTIONS

    def my_section(mod, memory_ctx, cfg):
        return "Always reply in French."

    builder = PromptBuilder(extra_sections=[("language", my_section)])

Phased extension hooks (already wired, inactive by default):
    Phase 2 — Personality profile:
        config = PromptConfig(personality_prefix="You are Aya, a direct and loyal companion.")
    Phase 2 — Goal awareness:
        config = PromptConfig(goal_suffix="Current user goal: finish Chapter 3 by Friday.")
    Phase 3 — Trajectory awareness:
        config = PromptConfig(trajectory_suffix="Long-term trajectory: declining focus trend over 3 weeks.")
"""

from .prompt_builder import PromptBuilder, DEFAULT_SECTIONS, SectionFn
from .prompt_config import PromptConfig
from .models import BuiltPrompt, PromptContext

__all__ = [
    "PromptBuilder",
    "DEFAULT_SECTIONS",
    "SectionFn",
    "PromptConfig",
    "BuiltPrompt",
    "PromptContext",
]