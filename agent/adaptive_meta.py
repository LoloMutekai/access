"""
A.C.C.E.S.S. — Adaptive Meta-Controller (Phase 5)

Self-calibrating control layer that replaces static meta-cognitive parameters
with bounded adaptive loops.

This is NOT machine learning. This is gain scheduling with hysteresis —
a well-understood technique from industrial control theory.

Subsystems:
    5.1  AdaptiveThreshold    — sigma-based intervention with hysteresis
    5.2  WeightRebalancer     — slow attention-weighted redistribution
    5.3  AlphaTuner           — regime-aware EMA responsiveness
    5.4  FatigueController    — asymmetric dampening + circuit breaker
    5.5  StabilityScorer      — second-order volatility-of-coherence

Design contracts:
    - All parameters bounded with hard floors and ceilings
    - All transitions deterministic (pure functions of state + input)
    - No global mutation — scoped per-turn only
    - Persistence-safe via to_dict()/from_dict()
    - Backward compatible: disabled → Phase 4.6 behavior exactly
    - No external libraries — pure algorithmic design

Mathematical invariants (proven in test suite):
    P1: 0.30 ≤ adapted_threshold ≤ 0.85
    P2: |sum(weights) - 1.0| < 1e-10 ∧ ∀i: 0.08 ≤ w_i ≤ 0.40
    P3: 0.10 ≤ adapted_alpha ≤ 0.50
    P4: 0.0 ≤ fatigue ≤ 1.0 ∧ (1 - max_dampening) ≤ dampening ≤ 1.0
    P5: 0.0 ≤ stability_of_stability ≤ 1.0
    P6: deterministic(same inputs → same outputs)
    P7: cold_start ≈ Phase 4.6 static behavior
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# Risk flags that MUST bypass fatigue dampening and S² gating.
# These represent real safety concerns (dependency, goal failure).
SAFETY_FLAGS = frozenset({"dependency_rising", "goals_failing"})

# All known risk flag names, indexed by dimension for weight rebalancing.
DIMENSION_FLAGS = (
    "personality_volatile",     # index 0 → w_personality
    "relationship_volatile",    # index 1 → w_relationship
    "mode_unstable",            # index 2 → w_mode
    "dependency_rising",        # index 3 → w_dependency
    "goals_failing",            # index 4 → w_goals
)

# Default diagnostic weights (Phase 4.5 baseline)
DEFAULT_WEIGHTS = (0.20, 0.20, 0.20, 0.25, 0.15)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdaptiveMetaConfig:
    """
    All constants for the adaptive meta-controller.
    Frozen — never mutated after construction.

    Every constant has a control-theoretic justification:

    threshold_alpha_mean (0.05):
        EMA smoothing for long-term coherence mean.
        Half-life = ln(2)/ln(1/0.95) ≈ 14 turns.
        Mean should shift very slowly — it represents the system's
        "natural operating point", not recent perturbations.

    threshold_alpha_var (0.08):
        EMA smoothing for coherence variance.
        Half-life ≈ 9 turns.
        Variance can adapt faster than mean because detecting
        regime changes (stable → volatile) is time-critical.

    threshold_k (1.5):
        Number of standard deviations below mean for intervention.
        At 1.5σ, the system intervenes when coherence drops to
        approximately the bottom 7% of its natural range (assuming
        Gaussian-like distribution around the mean).

    hysteresis_min (0.03):
        Minimum dead zone between activate and deactivate thresholds.
        Prevents chattering when coherence hovers near the threshold.
        Value chosen: 3% of full scale — a standard hysteresis band
        for process control systems.

    freq_dampen (0.05):
        Maximum threshold depression from frequent adjustments.
        When adjustments fire every turn, the threshold drops by
        up to 0.05, making the system less sensitive.

    weight_lambda (0.02):
        Blending speed for weight adaptation.
        At λ=0.02, each step moves weights 2% toward the
        attention-weighted target. The per-weight cap [0.08, 0.40]
        is the primary safety bound, not the speed.

    flag_epsilon (1.0):
        Regularization term in attention computation.
        Prevents division by zero and biases toward uniform
        weights when few flags have been observed.

    flag_decay (0.98):
        Per-turn decay of historical flag counts.
        Half-life = ln(2)/ln(1/0.98) ≈ 35 turns.
        Balances memory (recent patterns matter) with forgetting
        (ancient instability should not dominate).

    alpha_min (0.10), alpha_max (0.50):
        EMA alpha bounds.
        At α=0.10: ~90% weight on history (heavy smoothing, volatile regime)
        At α=0.50: balanced new/old (light smoothing, stable regime)
        Both are valid EMA parameters: |1-α| < 1 → contraction mapping.

    fatigue_rate (0.15):
        Fatigue accumulation per consecutive adjustment turn.
        Saturation in ~7 consecutive adjustments: 7 × 0.15 = 1.05 → clamped to 1.0.

    fatigue_recovery (0.08):
        Fatigue recovery per quiet turn.
        Full recovery from saturation: 1.0 / 0.08 = 12.5 turns.
        Asymmetric by design: faster to detect oscillation than to
        recover trust in system stability.

    fatigue_max_dampening (0.70):
        Maximum reduction of adjustment intensity at full fatigue.
        At fatigue=1.0: dampening = 1.0 - 0.70 = 0.30
        Adjustments are never fully silenced (30% minimum intensity).

    circuit_breaker_threshold (10):
        Emergency stop after N consecutive adjustments.
        If the meta-layer has been correcting for 10 straight turns,
        continued correction is likely making things worse.

    stability_kappa (10.0):
        Scaling factor for stability_of_stability metric.
        S² = 1/(1 + κ·μ_Δ) where μ_Δ is mean absolute coherence change.
        At κ=10: μ_Δ=0.1 → S²=0.5 (moderate volatility).
        Coherence ∈ [0,1], max Δ=1.0, typical Δ=0.01–0.05.
        This calibrates S² to have resolution in the normal range.

    stability_gate (0.25):
        Below this S² value, non-safety adjustments are gated off.
        When S²<0.25, mean inter-turn coherence change exceeds 0.3,
        meaning the signal is too noisy for threshold decisions.
    """

    # 5.1 Adaptive Threshold
    threshold_alpha_mean: float = 0.05
    threshold_alpha_var: float = 0.08
    threshold_k: float = 1.5
    threshold_floor: float = 0.30
    threshold_ceiling: float = 0.85
    hysteresis_min: float = 0.03
    freq_dampen: float = 0.05

    # 5.2 Weight Rebalancing
    weight_lambda: float = 0.02
    weight_min: float = 0.08
    weight_max: float = 0.40
    flag_epsilon: float = 1.0
    flag_decay: float = 0.98

    # 5.3 Alpha Tuning
    alpha_min: float = 0.10
    alpha_max: float = 0.50

    # 5.4 Fatigue Control
    fatigue_rate: float = 0.15
    fatigue_recovery: float = 0.08
    fatigue_max_dampening: float = 0.70
    circuit_breaker_threshold: int = 10

    # 5.5 Stability²
    stability_kappa: float = 10.0
    stability_gate: float = 0.25

    # Window size for adjustment frequency tracking
    adjustment_window: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE STATE — persisted between sessions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveMetaState:
    """
    Mutable state for the adaptive meta-controller.
    Persisted to disk. Loaded on boot. Defaults on missing.

    Non-persisted computed values:
        adapted_threshold, stability_of_stability, dampening_factor
        — these are re-derived each turn from persisted state + inputs.
    """

    # 5.1 Threshold state
    coherence_mean: float = 0.85         # optimistic cold start
    coherence_variance: float = 0.005    # low initial variance
    adjustments_active: bool = False      # hysteresis flag

    # 5.2 Weight state
    flag_counts: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    adapted_weights: list = field(default_factory=lambda: list(DEFAULT_WEIGHTS))

    # 5.3 Alpha state
    adapted_alpha: float = 0.3           # matches Phase 4.6 default

    # 5.4 Fatigue state
    fatigue_level: float = 0.0
    consecutive_adjustments: int = 0

    # Adjustment history ring buffer (for frequency computation)
    adjustment_history: list = field(default_factory=list)

    # Turn counter for diagnostics
    total_adaptive_turns: int = 0

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "coherence_mean": round(self.coherence_mean, 6),
            "coherence_variance": round(self.coherence_variance, 6),
            "adjustments_active": self.adjustments_active,
            "flag_counts": [round(c, 4) for c in self.flag_counts],
            "adapted_weights": [round(w, 6) for w in self.adapted_weights],
            "adapted_alpha": round(self.adapted_alpha, 4),
            "fatigue_level": round(self.fatigue_level, 4),
            "consecutive_adjustments": self.consecutive_adjustments,
            "adjustment_history": self.adjustment_history[-40:],  # cap stored size
            "total_adaptive_turns": self.total_adaptive_turns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveMetaState":
        """Deserialize from JSON dict. Graceful defaults on missing fields."""
        defaults = cls()
        return cls(
            coherence_mean=d.get("coherence_mean", defaults.coherence_mean),
            coherence_variance=d.get("coherence_variance", defaults.coherence_variance),
            adjustments_active=d.get("adjustments_active", defaults.adjustments_active),
            flag_counts=d.get("flag_counts", list(defaults.flag_counts))[:5],
            adapted_weights=d.get("adapted_weights", list(defaults.adapted_weights))[:5],
            adapted_alpha=d.get("adapted_alpha", defaults.adapted_alpha),
            fatigue_level=d.get("fatigue_level", defaults.fatigue_level),
            consecutive_adjustments=d.get("consecutive_adjustments", defaults.consecutive_adjustments),
            adjustment_history=d.get("adjustment_history", list(defaults.adjustment_history)),
            total_adaptive_turns=d.get("total_adaptive_turns", defaults.total_adaptive_turns),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTED CONTEXT — output of one adaptive cycle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdaptedContext:
    """
    Immutable output from AdaptiveMetaController.adapt().
    Consumed by StrategicAdjustmentEngine and CognitiveIdentityManager.
    """
    # 5.1: Threshold
    adapted_threshold: float = 0.6
    threshold_activate: float = 0.6
    threshold_deactivate: float = 0.63
    adjustments_active: bool = False

    # 5.2: Weights — recomputed coherence using adapted weights
    adapted_weights: tuple = DEFAULT_WEIGHTS
    adapted_coherence: float = 1.0

    # 5.3: Alpha for NEXT turn's EMA
    adapted_alpha: float = 0.3

    # 5.4: Fatigue
    dampening_factor: float = 1.0
    circuit_breaker_active: bool = False
    fatigue_level: float = 0.0

    # 5.5: Stability²
    stability_of_stability: float = 1.0
    stability_gate_open: bool = True  # True = non-safety adjustments allowed

    def to_dict(self) -> dict:
        return {
            "adapted_threshold": round(self.adapted_threshold, 4),
            "threshold_activate": round(self.threshold_activate, 4),
            "threshold_deactivate": round(self.threshold_deactivate, 4),
            "adjustments_active": self.adjustments_active,
            "adapted_weights": [round(w, 4) for w in self.adapted_weights],
            "adapted_coherence": round(self.adapted_coherence, 4),
            "adapted_alpha": round(self.adapted_alpha, 4),
            "dampening_factor": round(self.dampening_factor, 4),
            "circuit_breaker_active": self.circuit_breaker_active,
            "fatigue_level": round(self.fatigue_level, 4),
            "stability_of_stability": round(self.stability_of_stability, 4),
            "stability_gate_open": self.stability_gate_open,
        }

    def __repr__(self) -> str:
        parts = [
            f"θ={self.adapted_threshold:.3f}",
            f"α={self.adapted_alpha:.2f}",
            f"S²={self.stability_of_stability:.2f}",
            f"damp={self.dampening_factor:.2f}",
        ]
        if self.circuit_breaker_active:
            parts.append("CB=ON")
        if not self.stability_gate_open:
            parts.append("GATED")
        return f"AdaptedContext({', '.join(parts)})"


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMetaController:
    """
    Self-calibrating meta-controller. All 5 subsystems in one class.

    Usage:
        ctrl = AdaptiveMetaController(config)
        ctx = ctrl.adapt(raw_diagnostics, coherence_history, adjustment_triggered)
        # ctx.adapted_threshold replaces static threshold
        # ctx.adapted_alpha used for next EMA computation
        # ctx.dampening_factor scales adjustment intensity
        # ctx.stability_gate_open gates non-safety corrections

    Thread safety: NOT thread-safe. Single-threaded agent use only.
    """

    def __init__(self, config: Optional[AdaptiveMetaConfig] = None):
        self._cfg = config or AdaptiveMetaConfig()
        self._state = AdaptiveMetaState()

    @property
    def state(self) -> AdaptiveMetaState:
        return self._state

    @state.setter
    def state(self, value: AdaptiveMetaState) -> None:
        self._state = value

    def adapt(
        self,
        raw_coherence: float,
        raw_risk_flags: tuple,
        coherence_history: list,
        adjustment_triggered_this_turn: bool,
        raw_stability_scores: Optional[tuple] = None,
    ) -> AdaptedContext:
        """
        Run one full adaptive cycle. Returns immutable AdaptedContext.

        Args:
            raw_coherence: Raw coherence score from MetaDiagnosticsEngine
            raw_risk_flags: Risk flags from MetaDiagnosticsEngine
            coherence_history: List of raw coherence values (for S²)
            adjustment_triggered_this_turn: Whether StrategicAdjustment was active
            raw_stability_scores: Optional (p_stab, r_stab, m_stab, d_health, g_health)
                for weight-adjusted coherence recomputation.
                If None, adapted_coherence = raw_coherence.

        Returns:
            AdaptedContext with all adaptive parameters for this turn.
        """
        cfg = self._cfg
        s = self._state
        s.total_adaptive_turns += 1

        # ── 5.4: Fatigue (updated FIRST — affects all other decisions) ────
        dampening, circuit_break = self._update_fatigue(adjustment_triggered_this_turn)

        # ── 5.5: Stability² (independent of other subsystems) ─────────────
        s2 = self._compute_stability_sq(coherence_history)
        gate_open = s2 >= cfg.stability_gate

        # ── 5.2: Weight rebalancing ───────────────────────────────────────
        weights = self._rebalance_weights(raw_risk_flags)

        # Recompute coherence with adapted weights
        adapted_coherence = raw_coherence
        if raw_stability_scores is not None and len(raw_stability_scores) == 5:
            adapted_coherence = sum(
                w * sc for w, sc in zip(weights, raw_stability_scores)
            )
            adapted_coherence = _clamp(adapted_coherence)

        # ── 5.1: Adaptive threshold ───────────────────────────────────────
        threshold, th_activate, th_deactivate = self._compute_threshold(raw_coherence)

        # Apply hysteresis using SMOOTHED coherence (not raw)
        # Note: we use adapted_coherence here, which incorporates weight changes
        if adapted_coherence < th_activate:
            s.adjustments_active = True
        elif adapted_coherence > th_deactivate:
            s.adjustments_active = False
        # else: retain previous state (hysteresis dead zone)

        # ── 5.3: Alpha tuning (for NEXT turn) ────────────────────────────
        alpha = self._tune_alpha(s2)

        return AdaptedContext(
            adapted_threshold=threshold,
            threshold_activate=th_activate,
            threshold_deactivate=th_deactivate,
            adjustments_active=s.adjustments_active,
            adapted_weights=tuple(weights),
            adapted_coherence=adapted_coherence,
            adapted_alpha=alpha,
            dampening_factor=dampening,
            circuit_breaker_active=circuit_break,
            fatigue_level=s.fatigue_level,
            stability_of_stability=s2,
            stability_gate_open=gate_open,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5.1 ADAPTIVE THRESHOLD
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_threshold(self, raw_coherence: float) -> tuple:
        """
        Update running mean/variance and compute adaptive threshold.

        Returns: (threshold, activate_level, deactivate_level)

        Math:
            μ ← α_μ · c + (1 - α_μ) · μ
            σ² ← α_σ · (c - μ)² + (1 - α_σ) · σ²
            θ_raw = μ - k · σ
            θ = clamp(θ_raw - freq_penalty, floor, ceiling)
            activate = θ
            deactivate = θ + max(h_min, 0.5·σ)
        """
        cfg = self._cfg
        s = self._state

        # Update long-term mean (EMA, very slow)
        s.coherence_mean = (
            cfg.threshold_alpha_mean * raw_coherence
            + (1.0 - cfg.threshold_alpha_mean) * s.coherence_mean
        )

        # Update rolling variance (EMA of squared deviations)
        deviation_sq = (raw_coherence - s.coherence_mean) ** 2
        s.coherence_variance = (
            cfg.threshold_alpha_var * deviation_sq
            + (1.0 - cfg.threshold_alpha_var) * s.coherence_variance
        )

        # Standard deviation
        sigma = math.sqrt(max(0.0, s.coherence_variance))

        # Raw threshold: mean minus k standard deviations
        theta_raw = s.coherence_mean - cfg.threshold_k * sigma

        # Adjustment frequency penalty
        window = s.adjustment_history[-cfg.adjustment_window:]
        freq_ratio = sum(1 for x in window if x) / max(1, len(window)) if window else 0.0
        theta_adjusted = theta_raw - cfg.freq_dampen * freq_ratio

        # Hard bounds
        threshold = _clamp(theta_adjusted, cfg.threshold_floor, cfg.threshold_ceiling)

        # Hysteresis band
        h = max(cfg.hysteresis_min, 0.5 * sigma)
        activate = threshold
        deactivate = min(cfg.threshold_ceiling, threshold + h)

        return threshold, activate, deactivate

    # ─────────────────────────────────────────────────────────────────────────
    # 5.2 DYNAMIC WEIGHT REBALANCING
    # ─────────────────────────────────────────────────────────────────────────

    def _rebalance_weights(self, risk_flags: tuple) -> list:
        """
        Adapt diagnostic weights based on which dimensions trigger flags.

        Returns: list of 5 weights (normalized, bounded).

        Math:
            flag_counts_i ← 0.98 · flag_counts_i + new_i
            attention_i = flag_counts_i / (Σ flag_counts + ε)
            w_i ← (1 - λ) · w_i + λ · attention_i
            w = bounded_normalize(w, w_min, w_max)
        """
        cfg = self._cfg
        s = self._state

        # Decay historical counts
        s.flag_counts = [c * cfg.flag_decay for c in s.flag_counts]

        # Increment for new flags
        flags_set = set(risk_flags)
        for i, flag_name in enumerate(DIMENSION_FLAGS):
            if flag_name in flags_set:
                s.flag_counts[i] += 1.0

        # Compute attention distribution
        total_flags = sum(s.flag_counts) + cfg.flag_epsilon
        attention = [c / total_flags for c in s.flag_counts]

        # Blend toward attention-weighted target
        lam = cfg.weight_lambda
        s.adapted_weights = [
            (1.0 - lam) * w + lam * a
            for w, a in zip(s.adapted_weights, attention)
        ]

        # Bounded normalization
        s.adapted_weights = _bounded_normalize(
            s.adapted_weights, cfg.weight_min, cfg.weight_max
        )

        return list(s.adapted_weights)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3 SELF-TUNING EMA ALPHA
    # ─────────────────────────────────────────────────────────────────────────

    def _tune_alpha(self, stability_sq: float) -> float:
        """
        Compute adapted EMA alpha for the NEXT turn.

        Math:
            α = α_min + (α_max - α_min) · S²

        When stable (S²→1): α→α_max=0.50 (responsive, light smoothing)
        When volatile (S²→0): α→α_min=0.10 (dampened, heavy smoothing)

        One-step delay: computed from turn N's S², applied at turn N+1.
        This breaks the feedback loop S² → α → EMA → coherence → S².

        Stability proof:
            For any α ∈ [0.10, 0.50]:
            EMA_t = α·x_t + (1-α)·EMA_{t-1}
            |1-α| < 1 → contraction mapping → bounded by [min(x), max(x)]
            Changing α between turns changes convergence rate, not range.
            System cannot diverge.
        """
        cfg = self._cfg
        s = self._state
        s.adapted_alpha = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * _clamp(stability_sq)
        return s.adapted_alpha

    # ─────────────────────────────────────────────────────────────────────────
    # 5.4 ADJUSTMENT FATIGUE CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    def _update_fatigue(self, adjustment_triggered: bool) -> tuple:
        """
        Update fatigue state. Returns (dampening_factor, circuit_breaker_active).

        Math:
            If adjustment: fatigue ← min(1, fatigue + rate)
                          consecutive += 1
            Else:          fatigue ← max(0, fatigue - recovery)
                          consecutive = 0

            dampening = 1.0 - fatigue · max_dampening
            circuit_break = consecutive ≥ threshold

        Asymmetry: rate=0.15 > recovery=0.08
            Build to saturation: ~7 turns
            Recovery from saturation: ~13 turns
            This is deliberate: detecting oscillation is faster than
            recovering trust in stability (control-theoretic conservatism).

        No deadlock:
            - Circuit breaker forces adjustment=False for the blocked turn
            - This resets consecutive_adjustments to 0
            - Recovery rate reduces fatigue
            - System always recovers given quiet turns
        """
        cfg = self._cfg
        s = self._state

        if adjustment_triggered:
            s.consecutive_adjustments += 1
            s.fatigue_level = min(1.0, s.fatigue_level + cfg.fatigue_rate)
        else:
            s.consecutive_adjustments = 0
            s.fatigue_level = max(0.0, s.fatigue_level - cfg.fatigue_recovery)

        # Record in history ring buffer
        s.adjustment_history.append(adjustment_triggered)
        if len(s.adjustment_history) > cfg.adjustment_window * 2:
            s.adjustment_history = s.adjustment_history[-cfg.adjustment_window:]

        # Circuit breaker
        circuit_break = s.consecutive_adjustments >= cfg.circuit_breaker_threshold

        # Dampening factor
        dampening = 1.0 - s.fatigue_level * cfg.fatigue_max_dampening

        return dampening, circuit_break

    # ─────────────────────────────────────────────────────────────────────────
    # 5.5 META-STABILITY SCORING
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_stability_sq(self, coherence_history: list) -> float:
        """
        Second-order metric: volatility of coherence itself.

        Math:
            Δ_i = |c_i - c_{i-1}|        for i = 2..n
            μ_Δ = mean(Δ)
            S² = 1 / (1 + κ · μ_Δ)

        S² ∈ [0, 1]:
            S² = 1.0  : coherence never changes (perfect stability)
            S² ≈ 0.50 : mean Δ ≈ 0.1 (moderate volatility)
            S² ≈ 0.09 : mean Δ ≈ 1.0 (maximum theoretical volatility)

        κ = 10.0 justification:
            Coherence ∈ [0,1], max Δ = 1.0.
            Typical inter-turn changes: 0.01–0.05.
            At κ=10: Δ=0.1 → S²=0.5 (center of useful range).
            At κ=10: Δ=0.03 → S²=0.77 (healthy, responsive).
            At κ=10: Δ=0.3 → S²=0.25 (gate threshold, very noisy).
        """
        if len(coherence_history) < 2:
            return 1.0  # insufficient data → assume stable

        deltas = [
            abs(coherence_history[i] - coherence_history[i - 1])
            for i in range(1, len(coherence_history))
        ]
        mean_delta = sum(deltas) / len(deltas)

        return 1.0 / (1.0 + self._cfg.stability_kappa * mean_delta)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, v))


def _bounded_normalize(
    weights: list,
    lo: float,
    hi: float,
    max_iter: int = 10,
) -> list:
    """
    Normalize weights to sum=1.0 while respecting per-weight bounds [lo, hi].

    Algorithm: iterative projection.
        1. Clamp all weights to [lo, hi]
        2. If sum ≈ 1.0: done
        3. Compute residual = 1.0 - sum(w)
        4. Distribute residual uniformly among non-saturated weights
        5. Repeat until convergence

    Convergence guaranteed: each iteration either fixes at least one weight
    to a bound or reduces residual. For N weights, at most N iterations needed.

    Simple sum-and-divide normalization is INCORRECT here:
        If w=[0.5, 0.1, 0.1, 0.1, 0.1] with hi=0.40,
        divide-by-sum gives w[0]=0.556 > hi. Violation.
        Iterative projection handles this correctly.
    """
    n = len(weights)
    w = [max(lo, min(hi, x)) for x in weights]

    for _ in range(max_iter):
        total = sum(w)
        if abs(total - 1.0) < 1e-12:
            break

        residual = 1.0 - total
        if residual > 0:
            adjustable = [i for i in range(n) if w[i] < hi - 1e-12]
        else:
            adjustable = [i for i in range(n) if w[i] > lo + 1e-12]

        if not adjustable:
            break

        share = residual / len(adjustable)
        for i in adjustable:
            w[i] = max(lo, min(hi, w[i] + share))

    return w