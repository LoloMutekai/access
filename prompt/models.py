"""
A.C.C.E.S.S. — Prompt Builder Models

PromptContext:
    Input to PromptBuilder — carries everything needed to build the prompt.
    Immutable after construction. Clean input contract.

BuiltPrompt:
    Output of PromptBuilder — the final messages list + metadata.
    Immutable. Ready to hand directly to the LLM client.

Design rationale for a BuiltPrompt wrapper (vs returning raw list[dict]):
    - The metadata (sections, char count) is useful for logging, auditing, tests
    - Keeps the interface clean without polluting the caller
    - Tests can inspect individual sections without string-parsing the full prompt
    - Future: add token count estimation here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PromptContext:
    """
    Everything PromptBuilder needs to construct a prompt.

    user_input:       Raw message from the user.
    modulation:       ResponseModulation from ConversationModulator (duck-typed).
    memory_context:   Optional pre-formatted string of retrieved memory snippets.
                      Injected as a labelled block in the system prompt.
    session_id:       For future multi-session tracking (not used in prompt yet).
    """
    user_input: str
    modulation: object              # ResponseModulation (duck-typed — no circular import)
    memory_context: Optional[str] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class BuiltPrompt:
    """
    Immutable output of PromptBuilder.

    messages:       List of {role, content} dicts — ready for LLM API.
    system_prompt:  The system message content (for logging/testing).
    sections:       Ordered list of section labels included in the system prompt
                    (for unit-test inspection without string matching).
    char_count:     Total character count of the system prompt.
    """
    messages: tuple[dict, ...]      # frozen as tuple; expose as list via property
    system_prompt: str
    sections: tuple[str, ...]       # e.g. ("tone", "pacing", "cognitive_load", ...)
    char_count: int

    @property
    def message_list(self) -> list[dict]:
        """Return messages as a plain list (for LLM client compatibility)."""
        return list(self.messages)

    def to_api_messages(self) -> list[dict]:
        """Alias for message_list — explicit name for LLM call sites."""
        return list(self.messages)

    def has_section(self, section: str) -> bool:
        """Check if a named section was included in the system prompt."""
        return section in self.sections

    def __repr__(self) -> str:
        return (
            f"BuiltPrompt("
            f"sections={list(self.sections)}, "
            f"chars={self.char_count}, "
            f"messages={len(self.messages)})"
        )