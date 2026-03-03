"""
A.C.C.E.S.S. — Prompt Builder Configuration

Single source of truth for:
- Motivational bias thresholds and their language mappings
- Cognitive load tier boundaries and their constraints
- Tone / pacing / verbosity language instructions
- System prompt structure constants

All values are overridable. Zero hard-coded numbers in prompt_builder.py.
"""

from dataclasses import dataclass, field


@dataclass
class PromptConfig:

    # ── Motivational bias thresholds ──────────────────────────────────────
    # [-1.0 ... low_threshold ... neutral_threshold ... 1.0]
    motivational_low_threshold: float = -0.3     # below → de-energize
    motivational_high_threshold: float = 0.3     # above → push forward

    # ── Cognitive load tiers ──────────────────────────────────────────────
    cognitive_low_threshold: float = 0.5         # below → constrained mode
    cognitive_high_threshold: float = 0.8        # above → full mode

    # ── System prompt structure ───────────────────────────────────────────
    max_system_prompt_chars: int = 1200          # hard cap for compactness

    # ── Tone instructions ─────────────────────────────────────────────────
    # Maps tone label → compact instruction fragment
    tone_instructions: dict = field(default_factory=lambda: {
        "calm":        "Respond with a calm, low-pressure tone. Avoid urgency.",
        "energizing":  "Respond with energy and forward momentum. Be action-oriented.",
        "grounding":   "Respond with grounded, factual stability. Slow down the pace.",
        "challenging": "Respond with directness and high expectations. Push the user.",
        "reassuring":  "Respond with warmth and empathy. Normalize the user's experience.",
        "supportive":  "Respond with gentle encouragement. Scaffold and build up.",
        "neutral":     "Respond in a balanced, professional tone.",
    })

    # ── Pacing instructions ───────────────────────────────────────────────
    pacing_instructions: dict = field(default_factory=lambda: {
        "slow":   "Offer fewer points. Leave space. Do not overwhelm.",
        "normal": "Maintain a standard conversational rhythm.",
        "fast":   "Be crisp and efficient. Respect the user's momentum.",
    })

    # ── Verbosity instructions ────────────────────────────────────────────
    verbosity_instructions: dict = field(default_factory=lambda: {
        "concise":  "Be brief. Essentials only. No elaboration.",
        "normal":   "Provide balanced explanation. Include context where useful.",
        "detailed": "Go deep. Offer thorough reasoning and advanced context.",
    })

    # ── Structure bias ────────────────────────────────────────────────────
    structure_instructions: dict = field(default_factory=lambda: {
        "structured":     "Use bullet points or numbered steps when presenting options.",
        "conversational": "Write in natural flowing prose. Avoid bullet points.",
    })

    # ── Cognitive load tiers — constraint instructions ────────────────────
    cognitive_low_instruction: str = (
        "COGNITIVE LIMIT: Offer at most 1–2 steps or suggestions. "
        "Avoid complex explanations or multi-part reasoning."
    )
    cognitive_high_instruction: str = (
        "COGNITIVE MODE: Full reasoning allowed. "
        "You may provide advanced analysis, multi-step plans, and deep context."
    )
    # Between low and high → no explicit cognitive instruction added

    # ── Motivational bias instructions ───────────────────────────────────
    motivational_low_instruction: str = (
        "MOMENTUM: Reduce excitement. Avoid hype or urgency. "
        "Focus on stabilization, not acceleration."
    )
    motivational_neutral_instruction: str = ""   # nothing added for neutral
    motivational_high_instruction: str = (
        "MOMENTUM: Use forward-looking language. "
        "Emphasize progress, next steps, and achievable wins."
    )

    # ── Emotional validation instruction ─────────────────────────────────
    validation_instruction: str = (
        "VALIDATION: Before giving advice or information, briefly acknowledge "
        "the user's emotional state in 1 sentence."
    )

    # ── Memory context section header ─────────────────────────────────────
    memory_context_header: str = "CONTEXT FROM MEMORY:"

    # ── Extensibility hooks (for future layers) ───────────────────────────
    # Personality profile prefix — prepended before tone instructions
    # Set to non-empty string to activate persona (Phase 2)
    personality_prefix: str = ""

    # Goal awareness suffix — appended to system prompt (Phase 2)
    goal_suffix: str = ""

    # Trajectory awareness suffix — appended after goal (Phase 3)
    trajectory_suffix: str = ""