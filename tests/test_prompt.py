"""
A.C.C.E.S.S. — PromptBuilder Test Suite

Test strategy:
- Zero real EmotionEngine / ConversationModulator instantiation
- All inputs are duck-typed minimal stubs
- PromptConfig overridden per-test for deterministic assertions
- Each test validates exactly one logical rule
- No network, no DB, no embedder

Coverage:
    TestLowCognitiveLoadPrompt       — load < 0.5 → constraint instruction present
    TestHighCognitiveLoadPrompt      — load > 0.8 → full reasoning instruction present
    TestMidCognitiveLoadPrompt       — 0.5 <= load <= 0.8 → no cognitive instruction
    TestMotivationalBiasMapping      — three tiers: de-energize / neutral / forward
    TestValidationInstruction        — emotional_validation=True → instruction present
    TestStructureBiasInjection       — structured / conversational mapping
    TestPromptIsCompact              — char count within configured cap
    TestMemoryContextInjection       — memory_context → injected with header
    TestToneMapping                  — each tone label maps to a non-empty instruction
    TestPacingMapping                — slow / normal / fast each map correctly
    TestVerbosityMapping             — concise / normal / detailed each map correctly
    TestMessageFormat                — output is [system, user] dict list
    TestSectionAuditTrail            — sections tuple reflects what fired
    TestExtensibility                — custom sections injectable
    TestPhaseHooks                   — personality / goal / trajectory hooks wired
    TestFaultTolerance               — crashing section does not crash builder
    TestBuiltPromptContract          — BuiltPrompt immutability + API shape
    TestConfigOverride               — all thresholds configurable
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Optional

from prompt.prompt_builder import PromptBuilder, DEFAULT_SECTIONS
from prompt.prompt_config import PromptConfig
from prompt.models import BuiltPrompt, PromptContext


# ─────────────────────────────────────────────────────────────────────────────
# STUBS — minimal duck-typed ResponseModulation (no conversation imports needed)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeModulation:
    tone: str = "neutral"
    pacing: str = "normal"
    verbosity: str = "normal"
    structure_bias: str = "conversational"
    emotional_validation: bool = False
    motivational_bias: float = 0.0
    cognitive_load_limit: float = 1.0
    active_strategies: tuple = ()


def _make_mod(**kwargs) -> FakeModulation:
    return FakeModulation(**kwargs)


def _builder(**config_kwargs) -> PromptBuilder:
    cfg = PromptConfig(**config_kwargs) if config_kwargs else PromptConfig()
    return PromptBuilder(config=cfg)


USER_INPUT = "I feel stuck on this task."


# ─────────────────────────────────────────────────────────────────────────────
# TestLowCognitiveLoadPrompt
# ─────────────────────────────────────────────────────────────────────────────

class TestLowCognitiveLoadPrompt:
    """cognitive_load_limit < 0.5 → constraint instruction must appear in system prompt."""

    def test_low_load_injects_cognitive_section(self):
        b = _builder()
        mod = _make_mod(cognitive_load_limit=0.3)
        result = b.build(USER_INPUT, mod)
        assert result.has_section("cognitive_load")

    def test_low_load_system_contains_step_limit_keyword(self):
        cfg = PromptConfig(cognitive_low_instruction="COGNITIVE LIMIT: max 1 step.")
        b = PromptBuilder(config=cfg)
        mod = _make_mod(cognitive_load_limit=0.3)
        result = b.build(USER_INPUT, mod)
        assert "COGNITIVE LIMIT" in result.system_prompt

    def test_boundary_just_below_threshold_triggers(self):
        cfg = PromptConfig(cognitive_low_threshold=0.5)
        b = PromptBuilder(config=cfg)
        mod = _make_mod(cognitive_load_limit=0.49)
        result = b.build(USER_INPUT, mod)
        assert result.has_section("cognitive_load")

    def test_boundary_exactly_at_threshold_does_not_trigger_low(self):
        """load == threshold is NOT below threshold → no cognitive_load section."""
        cfg = PromptConfig(cognitive_low_threshold=0.5)
        b = PromptBuilder(config=cfg)
        mod = _make_mod(cognitive_load_limit=0.5)
        result = b.build(USER_INPUT, mod)
        # Exactly at the boundary: neither low nor high
        assert "cognitive_load" not in result.sections


# ─────────────────────────────────────────────────────────────────────────────
# TestHighCognitiveLoadPrompt
# ─────────────────────────────────────────────────────────────────────────────

class TestHighCognitiveLoadPrompt:
    """cognitive_load_limit > 0.8 → full reasoning instruction must appear."""

    def test_high_load_injects_cognitive_section(self):
        b = _builder()
        mod = _make_mod(cognitive_load_limit=0.9)
        result = b.build(USER_INPUT, mod)
        assert result.has_section("cognitive_load")

    def test_high_load_system_contains_full_reasoning_keyword(self):
        cfg = PromptConfig(cognitive_high_instruction="COGNITIVE MODE: Full reasoning.")
        b = PromptBuilder(config=cfg)
        mod = _make_mod(cognitive_load_limit=0.9)
        result = b.build(USER_INPUT, mod)
        assert "COGNITIVE MODE" in result.system_prompt

    def test_boundary_just_above_high_threshold_triggers(self):
        cfg = PromptConfig(cognitive_high_threshold=0.8)
        b = PromptBuilder(config=cfg)
        mod = _make_mod(cognitive_load_limit=0.81)
        result = b.build(USER_INPUT, mod)
        assert result.has_section("cognitive_load")

    def test_low_load_instruction_differs_from_high_load_instruction(self):
        b = _builder()
        low_result = b.build(USER_INPUT, _make_mod(cognitive_load_limit=0.2))
        high_result = b.build(USER_INPUT, _make_mod(cognitive_load_limit=1.0))
        # Both fire cognitive_load section but with different content
        assert low_result.system_prompt != high_result.system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# TestMidCognitiveLoadPrompt
# ─────────────────────────────────────────────────────────────────────────────

class TestMidCognitiveLoadPrompt:
    """0.5 <= load <= 0.8 → no cognitive_load section (neutral zone, no instruction)."""

    def test_mid_load_no_cognitive_section(self):
        b = _builder()
        for load in [0.5, 0.6, 0.7, 0.8]:
            mod = _make_mod(cognitive_load_limit=load)
            result = b.build(USER_INPUT, mod)
            assert "cognitive_load" not in result.sections, f"Failed at load={load}"

    def test_default_load_no_cognitive_section(self):
        """Default cognitive_load_limit=1.0 triggers HIGH, not absent."""
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(cognitive_load_limit=1.0))
        # 1.0 > 0.8 → cognitive_load fires
        assert result.has_section("cognitive_load")


# ─────────────────────────────────────────────────────────────────────────────
# TestMotivationalBiasMapping
# ─────────────────────────────────────────────────────────────────────────────

class TestMotivationalBiasMapping:
    """Three tiers: <= -0.3 / -0.3..0.3 / >= 0.3."""

    def test_strong_negative_bias_triggers_de_energize_instruction(self):
        cfg = PromptConfig(
            motivational_low_threshold=-0.3,
            motivational_low_instruction="MOMENTUM: Slow down.",
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(motivational_bias=-0.5))
        assert result.has_section("motivational_bias")
        assert "MOMENTUM: Slow down." in result.system_prompt

    def test_strong_positive_bias_triggers_forward_instruction(self):
        cfg = PromptConfig(
            motivational_high_threshold=0.3,
            motivational_high_instruction="MOMENTUM: Push forward.",
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(motivational_bias=0.5))
        assert result.has_section("motivational_bias")
        assert "MOMENTUM: Push forward." in result.system_prompt

    def test_neutral_bias_no_motivational_section(self):
        b = _builder()
        for bias in [0.0, 0.1, -0.1, 0.29, -0.29]:
            result = b.build(USER_INPUT, _make_mod(motivational_bias=bias))
            assert "motivational_bias" not in result.sections, f"Failed at bias={bias}"

    def test_exact_negative_threshold_triggers(self):
        cfg = PromptConfig(motivational_low_threshold=-0.3)
        b = PromptBuilder(config=cfg)
        # Exactly at -0.3 → triggers (<=)
        result = b.build(USER_INPUT, _make_mod(motivational_bias=-0.3))
        assert result.has_section("motivational_bias")

    def test_exact_positive_threshold_triggers(self):
        cfg = PromptConfig(motivational_high_threshold=0.3)
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(motivational_bias=0.3))
        assert result.has_section("motivational_bias")

    def test_negative_and_positive_instructions_are_different(self):
        b = _builder()
        neg = b.build(USER_INPUT, _make_mod(motivational_bias=-1.0))
        pos = b.build(USER_INPUT, _make_mod(motivational_bias=1.0))
        assert neg.system_prompt != pos.system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# TestValidationInstruction
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationInstruction:
    """emotional_validation=True → validation instruction injected."""

    def test_validation_true_injects_section(self):
        b = _builder()
        mod = _make_mod(emotional_validation=True)
        result = b.build(USER_INPUT, mod)
        assert result.has_section("validation")

    def test_validation_false_no_section(self):
        b = _builder()
        mod = _make_mod(emotional_validation=False)
        result = b.build(USER_INPUT, mod)
        assert "validation" not in result.sections

    def test_validation_instruction_content_configurable(self):
        cfg = PromptConfig(validation_instruction="VALIDATION: Acknowledge feelings first.")
        b = PromptBuilder(config=cfg)
        mod = _make_mod(emotional_validation=True)
        result = b.build(USER_INPUT, mod)
        assert "VALIDATION: Acknowledge feelings first." in result.system_prompt

    def test_validation_appears_before_user_message(self):
        """Validation instruction must be in system prompt, not user turn."""
        b = _builder()
        mod = _make_mod(emotional_validation=True)
        result = b.build(USER_INPUT, mod)
        system_msg = result.messages[0]
        assert system_msg["role"] == "system"
        cfg_default = PromptConfig()
        assert cfg_default.validation_instruction in system_msg["content"]


# ─────────────────────────────────────────────────────────────────────────────
# TestStructureBiasInjection
# ─────────────────────────────────────────────────────────────────────────────

class TestStructureBiasInjection:
    """structure_bias maps to distinct instructions."""

    def test_structured_bias_present_in_prompt(self):
        cfg = PromptConfig(
            structure_instructions={
                "structured": "STRUCTURE: Use numbered steps.",
                "conversational": "STRUCTURE: Write in prose.",
            }
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(structure_bias="structured"))
        assert "STRUCTURE: Use numbered steps." in result.system_prompt

    def test_conversational_bias_present_in_prompt(self):
        cfg = PromptConfig(
            structure_instructions={
                "structured": "STRUCTURE: Use numbered steps.",
                "conversational": "STRUCTURE: Write in prose.",
            }
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(structure_bias="conversational"))
        assert "STRUCTURE: Write in prose." in result.system_prompt

    def test_structure_section_always_fires(self):
        """structure_bias section is always present (not conditional)."""
        b = _builder()
        for bias in ["structured", "conversational"]:
            result = b.build(USER_INPUT, _make_mod(structure_bias=bias))
            assert result.has_section("structure_bias"), f"Failed for bias={bias}"

    def test_structured_and_conversational_produce_different_prompts(self):
        b = _builder()
        s = b.build(USER_INPUT, _make_mod(structure_bias="structured"))
        c = b.build(USER_INPUT, _make_mod(structure_bias="conversational"))
        assert s.system_prompt != c.system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# TestPromptIsCompact
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptIsCompact:
    """System prompt must stay within the configured character cap."""

    def test_default_char_limit_respected(self):
        b = _builder()
        mod = _make_mod(
            tone="calm",
            pacing="slow",
            verbosity="concise",
            emotional_validation=True,
            motivational_bias=-0.5,
            cognitive_load_limit=0.2,
        )
        result = b.build(USER_INPUT, mod, memory_context="A long memory context " * 20)
        assert result.char_count <= PromptConfig().max_system_prompt_chars

    def test_custom_char_limit_respected(self):
        cfg = PromptConfig(max_system_prompt_chars=200)
        b = PromptBuilder(config=cfg)
        mod = _make_mod(
            emotional_validation=True,
            motivational_bias=0.8,
            cognitive_load_limit=0.1,
        )
        result = b.build(USER_INPUT, mod, memory_context="Context " * 100)
        assert result.char_count <= 200

    def test_char_count_matches_actual_system_prompt_length(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert result.char_count == len(result.system_prompt)

    def test_minimum_prompt_is_non_empty(self):
        """Even a fully neutral modulation produces a non-empty system prompt."""
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert result.char_count > 0
        assert result.system_prompt.strip() != ""


# ─────────────────────────────────────────────────────────────────────────────
# TestMemoryContextInjection
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryContextInjection:
    """memory_context string must be injected into system prompt with header."""

    def test_memory_context_present_in_system_prompt(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(), memory_context="Past: user worked on project X.")
        assert "Past: user worked on project X." in result.system_prompt

    def test_memory_context_header_present(self):
        cfg = PromptConfig(memory_context_header="CONTEXT FROM MEMORY:")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(), memory_context="Something.")
        assert "CONTEXT FROM MEMORY:" in result.system_prompt

    def test_memory_context_section_fires(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(), memory_context="Anything.")
        assert result.has_section("memory_context")

    def test_no_memory_context_section_absent(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(), memory_context=None)
        assert "memory_context" not in result.sections

    def test_empty_string_memory_context_not_injected(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(), memory_context="")
        assert "memory_context" not in result.sections

    def test_memory_context_in_system_not_user_message(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(), memory_context="Private memory.")
        user_msg = result.messages[1]
        assert "Private memory." not in user_msg["content"]


# ─────────────────────────────────────────────────────────────────────────────
# TestToneMapping
# ─────────────────────────────────────────────────────────────────────────────

class TestToneMapping:
    """Every defined tone label maps to a non-empty instruction."""

    def test_all_default_tones_produce_non_empty_instruction(self):
        b = _builder()
        cfg = PromptConfig()
        for tone in cfg.tone_instructions.keys():
            result = b.build(USER_INPUT, _make_mod(tone=tone))
            assert result.has_section("tone"), f"tone section missing for: {tone}"

    def test_unknown_tone_falls_back_to_neutral(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(tone="nonexistent_tone"))
        # Falls back to neutral instruction, section still fires
        assert result.has_section("tone")
        cfg = PromptConfig()
        assert cfg.tone_instructions["neutral"] in result.system_prompt

    def test_each_tone_produces_distinct_system_prompt(self):
        b = _builder()
        cfg = PromptConfig()
        prompts = set()
        for tone in cfg.tone_instructions.keys():
            result = b.build(USER_INPUT, _make_mod(tone=tone))
            prompts.add(result.system_prompt)
        # Each tone must produce a different prompt (they have different instructions)
        assert len(prompts) == len(cfg.tone_instructions)


# ─────────────────────────────────────────────────────────────────────────────
# TestPacingMapping
# ─────────────────────────────────────────────────────────────────────────────

class TestPacingMapping:
    """slow / normal / fast all map to distinct, non-empty instructions."""

    def test_pacing_section_always_fires(self):
        b = _builder()
        for pacing in ["slow", "normal", "fast"]:
            result = b.build(USER_INPUT, _make_mod(pacing=pacing))
            assert result.has_section("pacing"), f"Failed for pacing={pacing}"

    def test_all_pacing_values_produce_distinct_prompts(self):
        b = _builder()
        slow = b.build(USER_INPUT, _make_mod(pacing="slow")).system_prompt
        normal = b.build(USER_INPUT, _make_mod(pacing="normal")).system_prompt
        fast = b.build(USER_INPUT, _make_mod(pacing="fast")).system_prompt
        assert len({slow, normal, fast}) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestVerbosityMapping
# ─────────────────────────────────────────────────────────────────────────────

class TestVerbosityMapping:
    """concise / normal / detailed all map to distinct, non-empty instructions."""

    def test_verbosity_section_always_fires(self):
        b = _builder()
        for verbosity in ["concise", "normal", "detailed"]:
            result = b.build(USER_INPUT, _make_mod(verbosity=verbosity))
            assert result.has_section("verbosity"), f"Failed for verbosity={verbosity}"

    def test_all_verbosity_values_produce_distinct_prompts(self):
        b = _builder()
        concise = b.build(USER_INPUT, _make_mod(verbosity="concise")).system_prompt
        normal  = b.build(USER_INPUT, _make_mod(verbosity="normal")).system_prompt
        detailed = b.build(USER_INPUT, _make_mod(verbosity="detailed")).system_prompt
        assert len({concise, normal, detailed}) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestMessageFormat
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageFormat:
    """Output must be exactly [system_msg, user_msg] in correct format."""

    def test_message_list_has_two_elements(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        msgs = result.message_list
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert result.message_list[0]["role"] == "system"

    def test_second_message_is_user(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert result.message_list[1]["role"] == "user"

    def test_user_message_contains_exact_input(self):
        b = _builder()
        user_text = "My very specific message that should appear verbatim."
        result = b.build(user_text, _make_mod())
        assert result.message_list[1]["content"] == user_text

    def test_to_api_messages_returns_list(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        api_msgs = result.to_api_messages()
        assert isinstance(api_msgs, list)
        assert len(api_msgs) == 2

    def test_messages_have_content_key(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        for msg in result.message_list:
            assert "role" in msg
            assert "content" in msg

    def test_build_from_context_produces_same_result(self):
        b = _builder()
        mod = _make_mod(tone="calm", emotional_validation=True)
        ctx = PromptContext(user_input=USER_INPUT, modulation=mod, memory_context="Some context.")
        result_a = b.build(USER_INPUT, mod, memory_context="Some context.")
        result_b = b.build_from_context(ctx)
        assert result_a.system_prompt == result_b.system_prompt
        assert result_a.message_list[1]["content"] == result_b.message_list[1]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# TestSectionAuditTrail
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionAuditTrail:
    """sections tuple must accurately reflect which sections fired."""

    def test_core_sections_always_present(self):
        """tone, pacing, verbosity, structure_bias always fire for any modulation."""
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        always_on = {"tone", "pacing", "verbosity", "structure_bias"}
        assert always_on.issubset(set(result.sections))

    def test_conditional_sections_absent_by_default(self):
        """validation, motivational_bias, memory_context absent unless triggered."""
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        conditional = {"validation", "motivational_bias", "memory_context"}
        assert set(result.sections).isdisjoint(conditional)

    def test_sections_ordered_consistently(self):
        """sections must reflect insertion order (same as section registry order)."""
        b = _builder()
        result = b.build(USER_INPUT, _make_mod(emotional_validation=True))
        section_list = list(result.sections)
        # tone must appear before validation
        assert section_list.index("tone") < section_list.index("validation")

    def test_has_section_method(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert result.has_section("tone") is True
        assert result.has_section("validation") is False
        assert result.has_section("nonexistent") is False


# ─────────────────────────────────────────────────────────────────────────────
# TestExtensibility
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensibility:
    """Custom sections must be injectable without modifying core code."""

    def test_extra_section_fires_when_returning_string(self):
        def always_on_section(mod, memory_ctx, cfg):
            return "Always respond in French."

        b = PromptBuilder(extra_sections=[("language", always_on_section)])
        result = b.build(USER_INPUT, _make_mod())
        assert result.has_section("language")
        assert "Always respond in French." in result.system_prompt

    def test_extra_section_skipped_when_returning_none(self):
        def conditional_section(mod, memory_ctx, cfg):
            return None

        b = PromptBuilder(extra_sections=[("conditional", conditional_section)])
        result = b.build(USER_INPUT, _make_mod())
        assert "conditional" not in result.sections

    def test_extra_sections_appended_after_defaults(self):
        def my_section(mod, memory_ctx, cfg):
            return "Custom suffix."

        b = PromptBuilder(extra_sections=[("custom", my_section)])
        result = b.build(USER_INPUT, _make_mod())
        sections = list(result.sections)
        # Custom section appears after core sections
        if "custom" in sections and "tone" in sections:
            assert sections.index("custom") > sections.index("tone")

    def test_full_section_replacement_via_sections_param(self):
        """Passing sections= replaces entire registry."""
        def only_section(mod, memory_ctx, cfg):
            return "Only this."

        b = PromptBuilder(sections=[("only", only_section)])
        result = b.build(USER_INPUT, _make_mod())
        assert list(result.sections) == ["only"]
        assert result.system_prompt == "Only this."

    def test_section_names_accessible(self):
        b = _builder()
        names = b.section_names
        assert "tone" in names
        assert "pacing" in names
        assert "verbosity" in names
        assert "cognitive_load" in names
        assert "memory_context" in names


# ─────────────────────────────────────────────────────────────────────────────
# TestPhaseHooks
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseHooks:
    """Phase 2 / Phase 3 extension hooks must be wired but silent by default."""

    def test_personality_prefix_silent_by_default(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert "personality" not in result.sections

    def test_goal_suffix_silent_by_default(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert "goal" not in result.sections

    def test_trajectory_suffix_silent_by_default(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert "trajectory" not in result.sections

    def test_personality_prefix_activates_when_set(self):
        cfg = PromptConfig(personality_prefix="You are Aya, a direct companion.")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod())
        assert result.has_section("personality")
        assert "You are Aya" in result.system_prompt

    def test_goal_suffix_activates_when_set(self):
        cfg = PromptConfig(goal_suffix="Current goal: finish Chapter 3.")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod())
        assert result.has_section("goal")
        assert "finish Chapter 3" in result.system_prompt

    def test_trajectory_suffix_activates_when_set(self):
        cfg = PromptConfig(trajectory_suffix="Long-term trend: improving focus.")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod())
        assert result.has_section("trajectory")
        assert "improving focus" in result.system_prompt

    def test_personality_appears_first_in_sections(self):
        """Personality prefix must be the first section in the prompt."""
        cfg = PromptConfig(personality_prefix="You are Aya.")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod())
        assert result.sections[0] == "personality"

    def test_goal_appears_before_trajectory(self):
        cfg = PromptConfig(
            goal_suffix="Goal: ship feature.",
            trajectory_suffix="Trend: improving."
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod())
        sections = list(result.sections)
        assert sections.index("goal") < sections.index("trajectory")


# ─────────────────────────────────────────────────────────────────────────────
# TestFaultTolerance
# ─────────────────────────────────────────────────────────────────────────────

class TestFaultTolerance:
    """A crashing section must be silently skipped — never crash the builder."""

    def test_crashing_section_does_not_raise(self):
        def crash_section(mod, memory_ctx, cfg):
            raise RuntimeError("Section explosion!")

        b = PromptBuilder(extra_sections=[("crash", crash_section)])
        # Must not raise
        result = b.build(USER_INPUT, _make_mod())
        assert "crash" not in result.sections

    def test_crashing_section_does_not_affect_other_sections(self):
        def crash_section(mod, memory_ctx, cfg):
            raise ValueError("boom")

        b = PromptBuilder(extra_sections=[("crash", crash_section)])
        result = b.build(USER_INPUT, _make_mod())
        # Core sections must still be present
        assert result.has_section("tone")
        assert result.has_section("pacing")

    def test_output_still_valid_after_crash(self):
        def crash_section(mod, memory_ctx, cfg):
            raise Exception("random error")

        b = PromptBuilder(extra_sections=[("crash", crash_section)])
        result = b.build(USER_INPUT, _make_mod())
        assert len(result.message_list) == 2
        assert result.message_list[0]["role"] == "system"
        assert result.message_list[1]["role"] == "user"


# ─────────────────────────────────────────────────────────────────────────────
# TestBuiltPromptContract
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltPromptContract:
    """BuiltPrompt must be immutable and expose the correct API."""

    def test_built_prompt_is_frozen(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        try:
            result.system_prompt = "hacked"
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass  # expected

    def test_sections_is_tuple(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert isinstance(result.sections, tuple)

    def test_messages_is_tuple_internally(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert isinstance(result.messages, tuple)

    def test_message_list_returns_list(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert isinstance(result.message_list, list)

    def test_to_api_messages_returns_list(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        assert isinstance(result.to_api_messages(), list)

    def test_repr_is_informative(self):
        b = _builder()
        result = b.build(USER_INPUT, _make_mod())
        r = repr(result)
        assert "BuiltPrompt" in r
        assert "chars=" in r
        assert "sections=" in r

    def test_deterministic_for_same_input(self):
        """Identical inputs must always produce identical output."""
        b = _builder()
        mod = _make_mod(tone="calm", emotional_validation=True, motivational_bias=0.5)
        r1 = b.build("test input", mod, memory_context="ctx")
        r2 = b.build("test input", mod, memory_context="ctx")
        assert r1.system_prompt == r2.system_prompt
        assert r1.sections == r2.sections


# ─────────────────────────────────────────────────────────────────────────────
# TestConfigOverride
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigOverride:
    """Every threshold and instruction string must be configurable."""

    def test_cognitive_low_threshold_configurable(self):
        for threshold in [0.3, 0.5, 0.7]:
            cfg = PromptConfig(cognitive_low_threshold=threshold)
            b = PromptBuilder(config=cfg)
            # Just below threshold → fires
            result = b.build(USER_INPUT, _make_mod(cognitive_load_limit=threshold - 0.01))
            assert result.has_section("cognitive_load"), f"Failed at threshold={threshold}"

    def test_cognitive_high_threshold_configurable(self):
        for threshold in [0.7, 0.8, 0.9]:
            cfg = PromptConfig(cognitive_high_threshold=threshold)
            b = PromptBuilder(config=cfg)
            result = b.build(USER_INPUT, _make_mod(cognitive_load_limit=threshold + 0.01))
            assert result.has_section("cognitive_load"), f"Failed at threshold={threshold}"

    def test_motivational_thresholds_configurable(self):
        cfg = PromptConfig(
            motivational_low_threshold=-0.1,
            motivational_high_threshold=0.1,
        )
        b = PromptBuilder(config=cfg)
        # Now -0.15 should trigger low
        result = b.build(USER_INPUT, _make_mod(motivational_bias=-0.15))
        assert result.has_section("motivational_bias")

    def test_tone_instructions_fully_replaceable(self):
        cfg = PromptConfig(
            tone_instructions={
                "neutral": "Parle de façon neutre.",
                "calm": "Reste calme.",
            }
        )
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(tone="calm"))
        assert "Reste calme." in result.system_prompt

    def test_memory_context_header_configurable(self):
        cfg = PromptConfig(memory_context_header="MÉMOIRE:")
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(), memory_context="Quelque chose.")
        assert "MÉMOIRE:" in result.system_prompt

    def test_char_cap_configurable(self):
        cfg = PromptConfig(max_system_prompt_chars=50)
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(emotional_validation=True))
        assert result.char_count <= 50

    def test_validation_instruction_string_configurable(self):
        custom = "RÈGLE: Valide l'émotion avant de répondre."
        cfg = PromptConfig(validation_instruction=custom)
        b = PromptBuilder(config=cfg)
        result = b.build(USER_INPUT, _make_mod(emotional_validation=True))
        assert custom in result.system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER — executed directly (no pytest needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestLowCognitiveLoadPrompt,
        TestHighCognitiveLoadPrompt,
        TestMidCognitiveLoadPrompt,
        TestMotivationalBiasMapping,
        TestValidationInstruction,
        TestStructureBiasInjection,
        TestPromptIsCompact,
        TestMemoryContextInjection,
        TestToneMapping,
        TestPacingMapping,
        TestVerbosityMapping,
        TestMessageFormat,
        TestSectionAuditTrail,
        TestExtensibility,
        TestPhaseHooks,
        TestFaultTolerance,
        TestBuiltPromptContract,
        TestConfigOverride,
    ]

    passed = []
    failed = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method_name in methods:
            label = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed.append(label)
                print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e))
                print(f"  ❌ {label}")
                traceback.print_exc()

    total = len(passed) + len(failed)
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{total} passed")
    if failed:
        print("\nFAILED:")
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")