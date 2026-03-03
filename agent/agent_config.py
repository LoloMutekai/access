"""
A.C.C.E.S.S. — Agent Configuration (Phase 4)

Phase 4 additions:
    - enable_relationship_tracking: bool
    - enable_personality_drift: bool
    - enable_self_model: bool
    - enable_goal_queue: bool
    - enable_persistence: bool
    - identity_data_dir: str
    - persist_every_n_turns: int

Backward compatible: all new flags default to False.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:

    # ── Session ────────────────────────────────────────────────────────────
    default_session_id: Optional[str] = None

    # ── Memory write-back ──────────────────────────────────────────────────
    write_user_turn_to_memory: bool = True
    write_assistant_turn_to_memory: bool = False
    auto_memory_importance: float = 0.5
    auto_memory_source: str = "interaction"

    # ── RAG ───────────────────────────────────────────────────────────────
    enable_rag: bool = True
    rag_top_k: int = 3
    rag_min_importance: float = 0.3
    rag_emotion_aware: bool = True

    # ── Emotional analysis ─────────────────────────────────────────────────
    apply_emotional_protection: bool = True

    # ── Tool Use (Phase 2) ────────────────────────────────────────────────
    enable_tool_use: bool = False
    tool_registry: Optional[object] = field(default=None, repr=False)
    max_tool_iterations: int = 5

    # ── Conversation History (Phase 2) ────────────────────────────────────
    enable_conversation_history: bool = False
    conversation_history_max_turns: int = 10

    # ── Self-Reflection (Phase 3) ─────────────────────────────────────────
    enable_reflection: bool = True
    use_reflection_summary_for_memory: bool = False
    adaptive_importance: bool = False

    # ── Adaptive Memory Loop (Phase 3) ────────────────────────────────────
    memory_decay_enabled: bool = True
    consolidation_threshold: float = 0.3

    # ── Trajectory (Phase 3) ──────────────────────────────────────────────
    trajectory_window_size: int = 10

    # ── Structured Logging (Phase 3) ──────────────────────────────────────
    enable_structured_logging: bool = False

    # ── LLM timeout ───────────────────────────────────────────────────────
    llm_timeout_seconds: float = 30.0

    # ── Response post-processing ──────────────────────────────────────────
    strip_response: bool = True
    max_response_chars: int = 0

    # ── Autonomous actions (Phase 3+) ─────────────────────────────────────
    enable_autonomous_actions: bool = False

    # ── Logging / Debug ───────────────────────────────────────────────────
    log_full_prompt: bool = False
    log_emotional_state: bool = True
    log_modulation: bool = True

    # ═════════════════════════════════════════════════════════════════════
    # Phase 4 — Persistent Cognitive Identity
    # ═════════════════════════════════════════════════════════════════════

    # Master switches (all default OFF for backward compatibility)
    enable_relationship_tracking: bool = False
    enable_personality_drift: bool = False
    enable_self_model: bool = False
    enable_goal_queue: bool = False

    # Persistence
    enable_persistence: bool = False
    identity_data_dir: str = "data/identity"
    persist_every_n_turns: int = 5
    create_identity_backups: bool = True

    # Goal queue
    goal_queue_max_goals: int = 20
    goal_priority_decay_rate: float = 0.02

    # ═════════════════════════════════════════════════════════════════════
    # Phase 4.5 — Meta-Cognitive Stability Layer
    # ═════════════════════════════════════════════════════════════════════

    enable_meta_cognition: bool = False
    meta_coherence_threshold: float = 0.6
    meta_window_size: int = 20
    meta_ema_alpha: float = 0.3

    # ═════════════════════════════════════════════════════════════════════
    # Phase 5 — Adaptive Meta-Control
    #
    # Self-calibrating control loops that replace static meta-cognitive
    # parameters with bounded adaptive dynamics.
    # When enable_adaptive_meta=False, none of this code executes.
    # System behaves identically to Phase 4.6.
    # ═════════════════════════════════════════════════════════════════════

    enable_adaptive_meta: bool = False           # master switch

    # 5.1 Adaptive Coherence Threshold
    adaptive_threshold_k: float = 1.5            # sigma multiplier (1.5σ ≈ 93rd %ile)
    adaptive_threshold_floor: float = 0.30       # constitutional minimum
    adaptive_threshold_ceiling: float = 0.85     # practical maximum
    adaptive_hysteresis_min: float = 0.03        # minimum dead zone band
    adaptive_freq_dampen: float = 0.05           # threshold depression from frequent adj

    # 5.2 Dynamic Weight Rebalancing
    adaptive_weight_lambda: float = 0.02         # blend speed toward attention target
    adaptive_weight_min: float = 0.08            # per-weight floor (no irrelevance)
    adaptive_weight_max: float = 0.40            # per-weight ceiling (no dominance)

    # 5.3 Self-Tuning EMA Alpha
    adaptive_alpha_min: float = 0.10             # heavy smoothing (volatile regime)
    adaptive_alpha_max: float = 0.50             # light smoothing (stable regime)

    # 5.4 Fatigue Control
    adaptive_fatigue_rate: float = 0.15          # accumulation per adj turn
    adaptive_fatigue_recovery: float = 0.08      # recovery per quiet turn
    adaptive_fatigue_max_dampening: float = 0.70 # max intensity reduction
    adaptive_circuit_breaker_threshold: int = 10 # emergency stop after N consecutive

    # 5.5 Stability²
    adaptive_stability_kappa: float = 10.0       # S² scaling factor
    adaptive_stability_gate: float = 0.25        # below this, gate non-safety adj


    # ═════════════════════════════════════════════════════════════════════
    # Phase 6 — Static Self-Inspection + Patch Proposal
    # ═════════════════════════════════════════════════════════════════════

    enable_structural_inspection: bool = False    # master switch
    structural_inspection_interval: int = 10      # run every N turns
    enable_patch_proposals: bool = False           # Phase 6.2
    structural_gate_threshold_engage: float = 0.55
    structural_gate_threshold_disengage: float = 0.45