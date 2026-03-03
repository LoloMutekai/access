"""
A.C.C.E.S.S. — Cognitive Identity Manager (Phase 4 + 4.5 + 4.6)

Thin orchestrator for all Phase 4 subsystems.
Injected into AgentCore — keeps AgentCore clean and Phase 4 isolated.

Phase 4.6 additions:
    - Scoped override pattern: meta-adjustments modify engine params per-turn only
    - EMA-smoothed coherence: avoids overreaction to single volatile turns
    - Meta-history persistence: survives across sessions via IdentityStore
    - Telemetry hook: emits meta_diagnostics events (optional, never crashes)
    - current_coherence(): exposes smoothed coherence to AgentCore

Design rationale:
    Adjustments computed on turn N are applied to turn N+1.
    This avoids retroactive mutation — the agent reacts to instability
    on the NEXT turn, not the current one.

Scoped override pattern:
    TraitDriftEngine.compute_drift(drift_cap_override=...)
    RelationshipEngine.update_from_turn(dependency_mitigation_multiplier=...)
    These params affect only one call — no permanent config mutation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from .agent_config import AgentConfig
from .relationship_state import (
    RelationshipState, RelationshipEngine, RelationshipConfig,
)
from .personality_state import (
    PersonalityTraits, TraitDriftEngine, PersonalityConfig,
)
from .self_model import SelfModel, SelfModelEngine, SelfModelConfig
from .goal_queue import GoalQueue, GoalQueueConfig, InternalGoal
from .persistence import IdentityStore, PersistenceConfig
from .meta_diagnostics import (
    MetaDiagnosticsEngine, MetaDiagnosticsConfig, MetaDiagnostics,
    IdentitySnapshot,
)
from .meta_strategy import (
    StrategicAdjustmentEngine, StrategyConfig, StrategicAdjustment,
)
from .adaptive_meta import (
    AdaptiveMetaController, AdaptiveMetaConfig, AdaptiveMetaState,
    AdaptedContext, SAFETY_FLAGS,
)

logger = logging.getLogger(__name__)

# ── Goal description templates by signal ──────────────────────────────────
_GOAL_TEMPLATES = {
    "push_forward": "Continue momentum on current task",
    "execute": "Complete in-progress execution",
    "stabilize": "Restore stability before advancing",
    "recover": "Allow recovery before new challenges",
    "explore": "Explore alternative approaches",
}


class CognitiveIdentityManager:
    """
    Single integration point for all Phase 4 subsystems.
    Fully optional — AgentCore works identically without it.
    """

    def __init__(
        self,
        config: AgentConfig,
        telemetry_hook: Optional[Callable[[str, dict], None]] = None,
    ):
        self._config = config
        self._telemetry_hook = telemetry_hook

        # Engines (stateless — pure computation)
        self._rel_engine = RelationshipEngine()
        self._trait_engine = TraitDriftEngine()
        self._self_engine = SelfModelEngine()

        # State (loaded from disk or defaults)
        self._relationship = RelationshipState()
        self._personality = PersonalityTraits()
        self._self_model = SelfModel()
        self._goal_queue = GoalQueue(GoalQueueConfig(
            max_goals=config.goal_queue_max_goals,
            priority_decay_rate=config.goal_priority_decay_rate,
        ))

        # Persistence
        self._store: Optional[IdentityStore] = None
        if config.enable_persistence:
            self._store = IdentityStore(PersistenceConfig(
                data_dir=Path(config.identity_data_dir),
                create_backups=config.create_identity_backups,
            ))

        # Auto-save counter
        self._turns_since_save: int = 0

        # ── Phase 4.5: Meta-Cognitive Layer ───────────────────────────────
        self._meta_engine: Optional[MetaDiagnosticsEngine] = None
        self._strategy_engine: Optional[StrategicAdjustmentEngine] = None
        self._meta_history: list[IdentitySnapshot] = []
        self._meta_window: int = config.meta_window_size
        self._last_diagnostics: Optional[MetaDiagnostics] = None
        self._last_adjustment: Optional[StrategicAdjustment] = None

        # ── Phase 4.6: EMA coherence buffer ───────────────────────────────
        self._coherence_history: list[float] = []
        self._smoothed_coherence: float = 1.0
        self._ema_alpha: float = config.meta_ema_alpha

        # ── Phase 5: Adaptive Meta-Controller ─────────────────────────────
        self._adaptive_ctrl: Optional[AdaptiveMetaController] = None
        self._last_adapted_context: Optional[AdaptedContext] = None

        if config.enable_meta_cognition:
            self._meta_engine = MetaDiagnosticsEngine()
            self._strategy_engine = StrategicAdjustmentEngine(
                StrategyConfig(coherence_threshold=config.meta_coherence_threshold)
            )
            # Phase 5: instantiate adaptive controller if enabled
            if config.enable_adaptive_meta:
                self._adaptive_ctrl = AdaptiveMetaController(
                    AdaptiveMetaConfig(
                        threshold_k=config.adaptive_threshold_k,
                        threshold_floor=config.adaptive_threshold_floor,
                        threshold_ceiling=config.adaptive_threshold_ceiling,
                        hysteresis_min=config.adaptive_hysteresis_min,
                        freq_dampen=config.adaptive_freq_dampen,
                        weight_lambda=config.adaptive_weight_lambda,
                        weight_min=config.adaptive_weight_min,
                        weight_max=config.adaptive_weight_max,
                        alpha_min=config.adaptive_alpha_min,
                        alpha_max=config.adaptive_alpha_max,
                        fatigue_rate=config.adaptive_fatigue_rate,
                        fatigue_recovery=config.adaptive_fatigue_recovery,
                        fatigue_max_dampening=config.adaptive_fatigue_max_dampening,
                        circuit_breaker_threshold=config.adaptive_circuit_breaker_threshold,
                        stability_kappa=config.adaptive_stability_kappa,
                        stability_gate=config.adaptive_stability_gate,
                    )
                )

        logger.info(
            f"CognitiveIdentityManager ready — "
            f"relationship={'ON' if config.enable_relationship_tracking else 'OFF'}, "
            f"personality={'ON' if config.enable_personality_drift else 'OFF'}, "
            f"self_model={'ON' if config.enable_self_model else 'OFF'}, "
            f"goals={'ON' if config.enable_goal_queue else 'OFF'}, "
            f"persistence={'ON' if self._store else 'OFF'}, "
            f"meta={'ON' if self._meta_engine else 'OFF'}, "
            f"adaptive={'ON' if self._adaptive_ctrl else 'OFF'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all states from disk. Safe — uses defaults on failure."""
        if self._store is None:
            return
        try:
            self._relationship = self._store.load_relationship_state()
            self._personality = self._store.load_personality_state()
            self._self_model = self._store.load_self_model()
            self._goal_queue = self._store.load_goal_queue(GoalQueueConfig(
                max_goals=self._config.goal_queue_max_goals,
                priority_decay_rate=self._config.goal_priority_decay_rate,
            ))
            logger.info("Identity state loaded from disk")
        except Exception as exc:
            logger.error(f"Identity load failed (using defaults): {exc}")

        # Phase 4.6: load meta history
        if self._config.enable_meta_cognition:
            try:
                self._meta_history = self._store.load_meta_history()
                # Trim to window
                if len(self._meta_history) > self._meta_window:
                    self._meta_history = self._meta_history[-self._meta_window:]
                logger.info(f"Meta history loaded: {len(self._meta_history)} snapshots")
            except Exception as exc:
                logger.error(f"Meta history load failed (using empty): {exc}")
                self._meta_history = []

        # Phase 5: load adaptive controller state
        if self._adaptive_ctrl is not None:
            try:
                self._adaptive_ctrl.state = self._store.load_adaptive_meta()
                logger.info(
                    f"Adaptive meta state loaded: "
                    f"turns={self._adaptive_ctrl.state.total_adaptive_turns}"
                )
            except Exception as exc:
                logger.error(f"Adaptive meta load failed (using defaults): {exc}")
                self._adaptive_ctrl.state = AdaptiveMetaState()

    def save(self) -> bool:
        """Save all states to disk. Returns False on failure."""
        if self._store is None:
            return True
        try:
            ok = self._store.save_all(
                self._relationship, self._personality,
                self._self_model, self._goal_queue,
            )
            # Phase 4.6: persist meta history
            if self._config.enable_meta_cognition and self._meta_history:
                ok &= self._store.save_meta_history(
                    self._meta_history, self._meta_window,
                )
            # Phase 5: persist adaptive controller state
            if self._adaptive_ctrl is not None:
                ok &= self._store.save_adaptive_meta(
                    self._adaptive_ctrl.state,
                )
            if ok:
                self._turns_since_save = 0
                logger.debug("Identity state saved to disk")
            return ok
        except Exception as exc:
            logger.error(f"Identity save failed: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # PER-TURN UPDATE (called by AgentCore after each finalized turn)
    # ─────────────────────────────────────────────────────────────────────────

    def update_from_turn(
        self,
        reflection_result,
        trajectory_state,
        emotional_state,
        modulation,
    ) -> None:
        """
        Update all Phase 4 subsystems from a finalized turn's signals.

        Phase 4.6 scoped override pattern:
            Adjustment computed on turn N-1 is applied to turn N's engine calls.
            After updates, new meta-cognition runs to produce adjustment for turn N+1.
        """
        try:
            # 0. Extract previous turn's adjustment for scoped overrides
            drift_cap_override: Optional[float] = None
            dep_mitigation_mult: float = 1.0
            allow_goal_injection: bool = True

            if (self._meta_engine is not None
                    and self._last_adjustment is not None
                    and self._last_adjustment.any_adjustment):
                drift_cap_override = self._last_adjustment.suggested_drift_cap
                dep_mitigation_mult = self._last_adjustment.dependency_mitigation_multiplier
                allow_goal_injection = self._last_adjustment.allow_goal_injection

            # 1. Relationship (with scoped dep mitigation multiplier)
            if self._config.enable_relationship_tracking:
                self._relationship = self._rel_engine.update_from_turn(
                    self._relationship, reflection_result,
                    trajectory_state, emotional_state,
                    dependency_mitigation_multiplier=dep_mitigation_mult,
                )

            # 2. Personality (with scoped drift cap override)
            if self._config.enable_personality_drift:
                self._personality = self._trait_engine.compute_drift(
                    self._personality, reflection_result,
                    trajectory_state, emotional_state, modulation,
                    drift_cap_override=drift_cap_override,
                )

            # 3. Self-model
            if self._config.enable_self_model:
                self._self_model = self._self_engine.observe_turn(
                    self._self_model, reflection_result,
                    trajectory_state, emotional_state, modulation,
                )

            # 4. Goal queue (with scoped injection throttle)
            if self._config.enable_goal_queue:
                if allow_goal_injection:
                    self._maybe_add_goal(reflection_result, trajectory_state)
                self._goal_queue.decay_priorities()

            # 5. Auto-save
            self._turns_since_save += 1
            if (self._config.persist_every_n_turns > 0
                    and self._turns_since_save >= self._config.persist_every_n_turns):
                self.save()

            # 6. Meta-cognitive layer (Phase 4.5+4.6)
            if self._meta_engine is not None:
                self._run_meta_cognition()

        except Exception as exc:
            logger.error(
                f"CognitiveIdentityManager.update_from_turn failed: {exc}",
                exc_info=True,
            )

    def _maybe_add_goal(self, reflection, trajectory) -> None:
        """Heuristically inject a goal from reflection signals."""
        from .relationship_state import _getattr
        goal_signal = _getattr(reflection, "goal_signal", None)
        importance = _getattr(reflection, "importance_score", 0.5)
        drift = _getattr(trajectory, "drift_score", 0.0)

        if goal_signal is None:
            return

        # Only add goals for significant turns
        if importance < 0.5:
            return

        desc = _GOAL_TEMPLATES.get(goal_signal)
        if desc is None:
            return

        # Priority based on importance and drift
        priority = importance * 0.7 + (1.0 - drift) * 0.3
        self._goal_queue.add_goal(desc, priority=priority, origin="reflection")

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION BOUNDARY (time-based decay between sessions)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_session_decay(self, hours_elapsed: float) -> None:
        """Apply time-based decay between sessions."""
        try:
            if self._config.enable_relationship_tracking:
                self._relationship = self._rel_engine.decay_over_time(
                    self._relationship, hours_elapsed,
                )
            if self._config.enable_personality_drift:
                self._personality = self._trait_engine.apply_homeostasis(
                    self._personality, hours_elapsed,
                )
        except Exception as exc:
            logger.error(f"Session decay failed: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # READ-ONLY ACCESSORS
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def relationship(self) -> RelationshipState:
        return self._relationship

    @property
    def personality(self) -> PersonalityTraits:
        return self._personality

    @property
    def self_model(self) -> SelfModel:
        return self._self_model

    @property
    def goal_queue(self) -> GoalQueue:
        return self._goal_queue

    # ─────────────────────────────────────────────────────────────────────────
    # META-COGNITIVE LAYER (Phase 4.5 + 4.6)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_meta_cognition(self) -> None:
        """
        Phase 4.5 + 4.6 + 5 meta-cognitive pipeline.

        Data flow:
            1. Capture snapshot → append to history
            2. MetaDiagnosticsEngine.analyze(history) → raw MetaDiagnostics
            3. EMA smooth raw coherence
            4. If adaptive: AdaptiveMetaController.adapt() → AdaptedContext
               Else: use static parameters
            5. StrategicAdjustmentEngine.evaluate(diagnostics, adapted_context)
            6. Inject meta-goals
            7. Phase 5: update EMA alpha for next turn
            8. Telemetry

        Never raises — errors are caught and logged.
        """
        try:
            # 1. Capture current state as snapshot
            goal_stats = self._goal_queue_stats()
            snapshot = IdentitySnapshot(
                personality=self._personality,
                relationship=self._relationship,
                dominant_mode=self._self_model.dominant_interaction_mode,
                dependency_risk=self._relationship.dependency_risk,
                active_goals=goal_stats["active"],
                completed_goals=goal_stats["completed"],
                expired_goals=goal_stats["expired"],
            )
            self._meta_history.append(snapshot)

            # Trim history to window
            if len(self._meta_history) > self._meta_window:
                self._meta_history = self._meta_history[-self._meta_window:]

            # 2. Run diagnostics (raw coherence)
            self._last_diagnostics = self._meta_engine.analyze(self._meta_history)
            raw_coherence = self._last_diagnostics.coherence_score

            # 3. EMA smoothing (Phase 4.6 — alpha may be adaptive)
            self._coherence_history.append(raw_coherence)
            if len(self._coherence_history) > self._meta_window:
                self._coherence_history = self._coherence_history[-self._meta_window:]
            self._smoothed_coherence = self._compute_ema(raw_coherence)

            # 4. Adaptive control (Phase 5) — sits between diagnostics and strategy
            adapted_ctx = None
            if self._adaptive_ctrl is not None:
                # Determine if previous turn had an adjustment
                prev_adj = (
                    self._last_adjustment is not None
                    and self._last_adjustment.any_adjustment
                )

                adapted_ctx = self._adaptive_ctrl.adapt(
                    raw_coherence=raw_coherence,
                    raw_risk_flags=self._last_diagnostics.risk_flags,
                    coherence_history=list(self._coherence_history),
                    adjustment_triggered_this_turn=prev_adj,
                )
                self._last_adapted_context = adapted_ctx

                # Phase 5: update EMA alpha for NEXT turn (one-step delay)
                self._ema_alpha = adapted_ctx.adapted_alpha

            # 5. Run strategic adjustment using SMOOTHED coherence
            if self._strategy_engine is not None:
                # Create a modified diagnostics with smoothed coherence for strategy
                smoothed_diag = MetaDiagnostics(
                    personality_volatility=self._last_diagnostics.personality_volatility,
                    relationship_volatility=self._last_diagnostics.relationship_volatility,
                    goal_resolution_rate=self._last_diagnostics.goal_resolution_rate,
                    dependency_trend=self._last_diagnostics.dependency_trend,
                    mode_instability=self._last_diagnostics.mode_instability,
                    coherence_score=self._smoothed_coherence,
                    risk_flags=self._last_diagnostics.risk_flags,
                    data_points=self._last_diagnostics.data_points,
                )
                # Phase 5: pass adapted context to strategy engine
                self._last_adjustment = self._strategy_engine.evaluate(
                    smoothed_diag, adapted_context=adapted_ctx,
                )

                # Inject meta-goals if any
                if self._last_adjustment.any_adjustment:
                    for mg in self._last_adjustment.meta_goals:
                        self._goal_queue.add_goal(
                            mg.description,
                            priority=mg.priority,
                            origin="reflection",
                        )
                    logger.debug(
                        f"Meta-cognition: raw={raw_coherence:.2f}, "
                        f"smoothed={self._smoothed_coherence:.2f}, "
                        f"flags={list(self._last_diagnostics.risk_flags)}, "
                        f"adjustment={self._last_adjustment.any_adjustment}"
                        + (f", adaptive={adapted_ctx}" if adapted_ctx else "")
                    )

            # 6. Telemetry (Phase 4.6 + 5)
            self._fire_meta_telemetry()

        except Exception as exc:
            logger.error(f"Meta-cognition failed: {exc}", exc_info=True)

    def _compute_ema(self, raw_value: float) -> float:
        """
        Exponential Moving Average.

        EMA_t = α × raw_t + (1 - α) × EMA_{t-1}

        α = 0.3 → new data has 30% weight, history has 70% weight.
        This prevents overreaction to single volatile turns.
        """
        alpha = self._ema_alpha
        if len(self._coherence_history) <= 1:
            return raw_value
        return alpha * raw_value + (1.0 - alpha) * self._smoothed_coherence

    def _goal_queue_stats(self) -> dict:
        """Count goals by status for snapshot."""
        active = completed = expired = 0
        for g in self._goal_queue.list_all():
            if g.status == "active":
                active += 1
            elif g.status == "completed":
                completed += 1
            elif g.status == "expired":
                expired += 1
        return {"active": active, "completed": completed, "expired": expired}

    def _fire_meta_telemetry(self) -> None:
        """Emit meta_diagnostics event via telemetry hook. Never crashes."""
        if self._telemetry_hook is None:
            return
        if self._last_diagnostics is None:
            return
        try:
            payload = {
                "coherence": round(self._smoothed_coherence, 4),
                "raw_coherence": round(self._last_diagnostics.coherence_score, 4),
                "flags": list(self._last_diagnostics.risk_flags),
                "history_size": len(self._meta_history),
            }
            # Phase 5: include adaptive context
            if self._last_adapted_context is not None:
                ctx = self._last_adapted_context
                payload["adaptive"] = {
                    "threshold": round(ctx.adapted_threshold, 4),
                    "alpha": round(ctx.adapted_alpha, 4),
                    "stability_sq": round(ctx.stability_of_stability, 4),
                    "dampening": round(ctx.dampening_factor, 4),
                    "circuit_breaker": ctx.circuit_breaker_active,
                    "gate_open": ctx.stability_gate_open,
                    "fatigue": round(ctx.fatigue_level, 4),
                }
            self._telemetry_hook("meta_diagnostics", payload)
        except Exception as exc:
            logger.warning(f"Meta telemetry error (suppressed): {exc}")

    # ── Read-only meta accessors ──────────────────────────────────────────

    @property
    def last_diagnostics(self) -> Optional[MetaDiagnostics]:
        """Most recent meta-diagnostics. None if meta-cognition is OFF or no data yet."""
        return self._last_diagnostics

    @property
    def last_adjustment(self) -> Optional[StrategicAdjustment]:
        """Most recent strategic adjustment. None if not triggered."""
        return self._last_adjustment

    @property
    def meta_history(self) -> list[IdentitySnapshot]:
        """Read-only access to meta history buffer."""
        return list(self._meta_history)

    @property
    def smoothed_coherence(self) -> float:
        """EMA-smoothed coherence. 1.0 if no data yet."""
        return self._smoothed_coherence

    @property
    def coherence_history(self) -> list[float]:
        """Raw coherence values for debugging. Read-only copy."""
        return list(self._coherence_history)

    def current_coherence(self) -> Optional[float]:
        """
        Smoothed coherence score for external consumption (AgentCore).
        Returns None if meta-cognition is OFF.
        """
        if self._meta_engine is None:
            return None
        return self._smoothed_coherence

    def meta_snapshot(self) -> Optional[dict]:
        """Complete meta-cognitive diagnostic snapshot. None if meta-cognition is OFF."""
        if self._meta_engine is None:
            return None
        diag = self._last_diagnostics
        adj = self._last_adjustment
        result = {
            "diagnostics": diag.to_dict() if diag else None,
            "adjustment": adj.to_dict() if adj else None,
            "smoothed_coherence": round(self._smoothed_coherence, 4),
            "history_size": len(self._meta_history),
            "window_size": self._meta_window,
        }
        # Phase 5: include adaptive context
        if self._last_adapted_context is not None:
            result["adaptive"] = self._last_adapted_context.to_dict()
        return result

    # ── Phase 5: Adaptive Meta Accessors ─────────────────────────────────

    @property
    def last_adapted_context(self) -> Optional[AdaptedContext]:
        """Most recent adaptive context. None if adaptive meta is OFF."""
        return self._last_adapted_context

    @property
    def adaptive_state(self) -> Optional[AdaptiveMetaState]:
        """Current adaptive controller state. None if adaptive meta is OFF."""
        if self._adaptive_ctrl is None:
            return None
        return self._adaptive_ctrl.state

    @property
    def adaptive_enabled(self) -> bool:
        """Whether Phase 5 adaptive meta-control is active."""
        return self._adaptive_ctrl is not None

    # ─────────────────────────────────────────────────────────────────────────
    # FULL SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────────

    def identity_snapshot(self) -> dict:
        """Complete diagnostic snapshot of all Phase 4 + 4.5 + 4.6 state."""
        dep = self._rel_engine.detect_dependency_risk(self._relationship)
        result = {
            "relationship": self._relationship.to_dict(),
            "personality": self._personality.to_dict(),
            "self_model": self._self_model.to_dict(),
            "goals": self._goal_queue.to_dict(),
            "dependency_analysis": dep,
        }
        meta = self.meta_snapshot()
        if meta is not None:
            result["meta_cognition"] = meta
        return result