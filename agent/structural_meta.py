"""
A.C.C.E.S.S. — Structural Meta-State (Phase 6.1 → Meta Integration)

Bridge between static self-inspection (Phase 6.1) and the meta-cognitive
regulation layer (Phase 4.5/5). Converts raw InspectionReport snapshots
into a smoothed, bounded structural health signal that can influence
system-level decision-making.

This is NOT a feedback loop into code.
This is a SIGNAL LAYER that informs meta-regulation.

Design:
    - StructuralMetaState is FROZEN — immutable snapshot
    - StructuralMetaTracker is the stateful updater (EMA-smoothed)
    - StructuralGate is pure gating logic (no state)
    - All computations deterministic
    - JSON-serializable throughout
    - Cold-start safe (sensible defaults, no NaN)
    - No external dependencies beyond stdlib

Mathematical properties:
    P1: structural_instability_index ∈ [0.0, 1.0]   (bounded)
    P2: EMA smoothing contracts toward true mean      (no divergence)
    P3: Gating is monotonic: higher instability → stricter gates
    P4: No oscillation: hysteresis on gate transitions
    P5: Deterministic: same inputs → same outputs

Integration points:
    - meta_diagnostics  → structural health feeds into coherence scoring
    - adaptive_meta     → instability reduces adaptive sensitivity
    - meta_strategy     → patch proposal aggressiveness gating
    - persistence       → to_dict()/from_dict() for session survival
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructuralMetaConfig:
    """
    All constants for structural meta-state tracking.
    Frozen — never mutated after construction.

    EMA alpha (0.15):
        Half-life ≈ ln(2)/ln(1/0.85) ≈ 4.3 inspections.
        Structural changes happen slowly (code doesn't mutate between turns),
        so we want moderate smoothing. Not as aggressive as Phase 5's coherence
        alpha (0.05–0.50) because inspection frequency is much lower.

    instability_w_* weights (sum to 1.0):
        Composite index weights for each structural dimension.
        Coupling and complexity are weighted highest because they have
        the strongest correlation with maintenance difficulty.

    gate_threshold_engage (0.55):
        When instability exceeds this, gating activates.
        Chosen at ~halfway point because structural instability is
        harder to recover from than behavioral volatility.

    gate_threshold_disengage (0.45):
        Hysteresis: gate deactivates only when instability drops below this.
        Band of 0.10 prevents chattering near the threshold.

    max_patch_suggestions (20):
        Hard cap on patch suggestions per run. Prevents suggestion spam.
        At high instability, this is further reduced via aggressiveness modulation.
    """

    # EMA smoothing
    ema_alpha: float = 0.15
    ema_alpha_min: float = 0.05
    ema_alpha_max: float = 0.30

    # Instability index weights (must sum to 1.0)
    instability_w_risk: float = 0.30
    instability_w_smells: float = 0.20
    instability_w_cycles: float = 0.25
    instability_w_violations: float = 0.15
    instability_w_io: float = 0.10

    # Smell density normalization (number of smells at which density = 1.0)
    smell_density_saturation: int = 20

    # Cycle normalization (number of cycles at which raw signal = 1.0)
    cycle_saturation: int = 5

    # Layer violation normalization
    violation_saturation: int = 8

    # Gating thresholds (with hysteresis band)
    gate_threshold_engage: float = 0.55
    gate_threshold_disengage: float = 0.45

    # Patch proposal aggressiveness range
    max_patch_suggestions: int = 20
    min_patch_suggestions: int = 3

    # Adaptive sensitivity reduction (applied to Phase 5 alpha)
    max_sensitivity_reduction: float = 0.30

    # Cold start: how many inspections before EMA is trusted
    cold_start_inspections: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL META STATE (immutable snapshot)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructuralMetaState:
    """
    Immutable snapshot of structural health.
    Produced by StructuralMetaTracker.update().

    All fields are bounded and JSON-serializable.
    """

    # Raw metrics from latest inspection
    last_composite_risk: float = 0.0
    last_smell_count: int = 0
    last_cycle_count: int = 0
    last_violation_count: int = 0
    last_health_grade: str = "A"
    last_files_analyzed: int = 0
    last_lines_analyzed: int = 0

    # EMA-smoothed values
    ema_composite_risk: float = 0.0
    ema_smell_density: float = 0.0
    ema_cycle_signal: float = 0.0

    # Composite instability index ∈ [0.0, 1.0]
    structural_instability_index: float = 0.0

    # Trend: positive = degrading, negative = improving
    trend: float = 0.0

    # Gate state
    intervention_gate_active: bool = False

    # Meta
    inspection_count: int = 0
    inspection_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Enforce bounds via frozen object mutation."""
        object.__setattr__(
            self, "structural_instability_index",
            _clamp(self.structural_instability_index),
        )
        object.__setattr__(
            self, "ema_composite_risk",
            _clamp(self.ema_composite_risk),
        )

    @property
    def is_healthy(self) -> bool:
        return self.structural_instability_index < 0.3

    @property
    def is_degraded(self) -> bool:
        return self.structural_instability_index >= 0.5

    @property
    def is_cold_start(self) -> bool:
        return self.inspection_count < 3

    def to_dict(self) -> dict:
        return {
            "last_composite_risk": round(self.last_composite_risk, 4),
            "last_smell_count": self.last_smell_count,
            "last_cycle_count": self.last_cycle_count,
            "last_violation_count": self.last_violation_count,
            "last_health_grade": self.last_health_grade,
            "last_files_analyzed": self.last_files_analyzed,
            "last_lines_analyzed": self.last_lines_analyzed,
            "ema_composite_risk": round(self.ema_composite_risk, 4),
            "ema_smell_density": round(self.ema_smell_density, 4),
            "ema_cycle_signal": round(self.ema_cycle_signal, 4),
            "structural_instability_index": round(self.structural_instability_index, 4),
            "trend": round(self.trend, 4),
            "intervention_gate_active": self.intervention_gate_active,
            "inspection_count": self.inspection_count,
            "inspection_timestamp": self.inspection_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralMetaState":
        """Deserialize from JSON dict. Safe against missing/malformed keys."""
        return cls(
            last_composite_risk=_safe_dict_float(d, "last_composite_risk", 0.0),
            last_smell_count=_safe_dict_int(d, "last_smell_count", 0),
            last_cycle_count=_safe_dict_int(d, "last_cycle_count", 0),
            last_violation_count=_safe_dict_int(d, "last_violation_count", 0),
            last_health_grade=str(d.get("last_health_grade", "A")),
            last_files_analyzed=_safe_dict_int(d, "last_files_analyzed", 0),
            last_lines_analyzed=_safe_dict_int(d, "last_lines_analyzed", 0),
            ema_composite_risk=_safe_dict_float(d, "ema_composite_risk", 0.0),
            ema_smell_density=_safe_dict_float(d, "ema_smell_density", 0.0),
            ema_cycle_signal=_safe_dict_float(d, "ema_cycle_signal", 0.0),
            structural_instability_index=_safe_dict_float(d, "structural_instability_index", 0.0),
            trend=_safe_dict_float(d, "trend", 0.0),
            intervention_gate_active=bool(d.get("intervention_gate_active", False)),
            inspection_count=_safe_dict_int(d, "inspection_count", 0),
            inspection_timestamp=_parse_dt(d.get("inspection_timestamp")),
        )

    def __repr__(self) -> str:
        gate = " GATED" if self.intervention_gate_active else ""
        return (
            f"StructuralMetaState("
            f"instability={self.structural_instability_index:.3f}, "
            f"risk={self.ema_composite_risk:.3f}, "
            f"grade={self.last_health_grade}, "
            f"trend={self.trend:+.4f}, "
            f"n={self.inspection_count}{gate})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL META TRACKER (stateful updater)
# ─────────────────────────────────────────────────────────────────────────────

class StructuralMetaTracker:
    """
    Maintains the structural meta-state across inspection cycles.

    Lifecycle:
        1. Create with optional config and previous state
        2. Call update(report) after each StaticInspector.inspect()
        3. Read .state for current StructuralMetaState
        4. Persist .state.to_dict() for session survival

    EMA update rule:
        ema_new = α · raw + (1 - α) · ema_old

    Cold start behavior:
        First inspection: EMA initialized to raw values (no smoothing).
        Inspections 2–3: EMA applies but instability index is conservative.
        After cold_start_inspections: full tracking active.
    """

    def __init__(
        self,
        config: Optional[StructuralMetaConfig] = None,
        initial_state: Optional[StructuralMetaState] = None,
    ):
        self._cfg = config or StructuralMetaConfig()
        self._state = initial_state or StructuralMetaState()
        self._previous_instability: float = self._state.structural_instability_index

    @property
    def state(self) -> StructuralMetaState:
        return self._state

    @property
    def config(self) -> StructuralMetaConfig:
        return self._cfg

    def update(self, report) -> StructuralMetaState:
        """
        Update structural meta-state from a new InspectionReport.

        Args:
            report: InspectionReport from StaticInspector.inspect()
                    Must have: files_analyzed, lines_analyzed, composite_risk,
                    health_grade, smells (iterable), cycles (iterable),
                    layer_violations (iterable)

        Returns:
            New StructuralMetaState (frozen).
        """
        cfg = self._cfg
        prev = self._state

        # ── Extract raw metrics ───────────────────────────────────────────
        files_analyzed = _safe_int(report, "files_analyzed")
        lines_analyzed = _safe_int(report, "lines_analyzed")
        composite_risk = _clamp(_safe_float(report, "composite_risk"))
        health_grade = _safe_str(report, "health_grade", "A")
        smell_count = _safe_len(report, "smells")
        cycle_count = _safe_len(report, "cycles")
        violation_count = _safe_len(report, "layer_violations")

        # ── Normalize raw signals to [0, 1] ──────────────────────────────
        smell_density = min(1.0, smell_count / max(1, cfg.smell_density_saturation))
        cycle_signal = min(1.0, cycle_count / max(1, cfg.cycle_saturation))
        violation_signal = min(1.0, violation_count / max(1, cfg.violation_saturation))

        # ── Compute IO risk from report if available ──────────────────────
        io_signal = 0.0
        try:
            files_data = report.files if hasattr(report, "files") else ()
            if files_data:
                io_densities = [
                    f.io_density for f in files_data
                    if hasattr(f, "io_density")
                ]
                if io_densities:
                    io_signal = min(1.0, sum(io_densities) / len(io_densities) * 3.0)
        except (TypeError, AttributeError):
            io_signal = 0.0

        # ── EMA update ────────────────────────────────────────────────────
        count = prev.inspection_count + 1
        alpha = cfg.ema_alpha

        if count == 1:
            # Cold start: initialize EMA to raw values
            ema_risk = composite_risk
            ema_smell = smell_density
            ema_cycle = cycle_signal
        else:
            ema_risk = alpha * composite_risk + (1.0 - alpha) * prev.ema_composite_risk
            ema_smell = alpha * smell_density + (1.0 - alpha) * prev.ema_smell_density
            ema_cycle = alpha * cycle_signal + (1.0 - alpha) * prev.ema_cycle_signal

        # ── Structural Instability Index ──────────────────────────────────
        instability = (
            cfg.instability_w_risk * ema_risk
            + cfg.instability_w_smells * ema_smell
            + cfg.instability_w_cycles * ema_cycle
            + cfg.instability_w_violations * violation_signal
            + cfg.instability_w_io * io_signal
        )
        instability = _clamp(instability)

        # ── Trend (EMA of instability change) ────────────────────────────
        raw_delta = instability - self._previous_instability
        trend_alpha = 0.3  # moderate smoothing for trend
        trend = trend_alpha * raw_delta + (1.0 - trend_alpha) * prev.trend

        # ── Gating with hysteresis ────────────────────────────────────────
        gate_active = prev.intervention_gate_active
        if not gate_active and instability > cfg.gate_threshold_engage:
            gate_active = True
        elif gate_active and instability < cfg.gate_threshold_disengage:
            gate_active = False

        # Store for next trend computation
        self._previous_instability = instability

        # ── Build new state ───────────────────────────────────────────────
        new_state = StructuralMetaState(
            last_composite_risk=composite_risk,
            last_smell_count=smell_count,
            last_cycle_count=cycle_count,
            last_violation_count=violation_count,
            last_health_grade=health_grade,
            last_files_analyzed=files_analyzed,
            last_lines_analyzed=lines_analyzed,
            ema_composite_risk=ema_risk,
            ema_smell_density=ema_smell,
            ema_cycle_signal=ema_cycle,
            structural_instability_index=instability,
            trend=trend,
            intervention_gate_active=gate_active,
            inspection_count=count,
            inspection_timestamp=datetime.now(UTC),
        )

        self._state = new_state
        logger.info(f"Structural meta updated: {new_state}")
        return new_state


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL GATE (pure gating logic)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructuralGateDecision:
    """
    Immutable gating decision based on structural instability.
    Consumed by PatchProposalEngine, meta_strategy, and adaptive_meta.
    """

    # Patch proposal aggressiveness ∈ [0.0, 1.0]
    # 1.0 = max suggestions, 0.0 = suppress all non-critical suggestions
    patch_aggressiveness: float = 1.0

    # Maximum number of suggestions allowed this run
    max_suggestions: int = 20

    # Sensitivity reduction for adaptive_meta (Phase 5)
    # 0.0 = no reduction, 0.30 = reduce alpha sensitivity by 30%
    sensitivity_reduction: float = 0.0

    # Meta warning flag: True when structural health is degraded
    meta_warning: bool = False

    # Explanation for logging
    reason: str = "healthy"

    def to_dict(self) -> dict:
        return {
            "patch_aggressiveness": round(self.patch_aggressiveness, 4),
            "max_suggestions": self.max_suggestions,
            "sensitivity_reduction": round(self.sensitivity_reduction, 4),
            "meta_warning": self.meta_warning,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return (
            f"StructuralGateDecision("
            f"aggr={self.patch_aggressiveness:.2f}, "
            f"max={self.max_suggestions}, "
            f"sens_red={self.sensitivity_reduction:.2f}, "
            f"warn={self.meta_warning}, "
            f"reason={self.reason!r})"
        )


class StructuralGate:
    """
    Pure function: StructuralMetaState → StructuralGateDecision.
    No state. No side effects.

    Gating rules:
        instability < 0.30 → healthy: full aggressiveness
        0.30 ≤ inst < 0.55 → caution: moderate reduction
        0.55 ≤ inst < 0.75 → degraded: major reduction + meta warning
        inst ≥ 0.75        → critical: minimal suggestions + max restriction

    Aggressiveness formula:
        aggressiveness = 1.0 - instability^1.5
        (concave curve: gentle reduction at low instability, aggressive at high)

    Sensitivity reduction:
        Linear scale: reduction = instability × max_reduction
        Only applied when instability > 0.30 (below that = no Phase 5 impact)
    """

    def __init__(self, config: Optional[StructuralMetaConfig] = None):
        self._cfg = config or StructuralMetaConfig()

    def evaluate(self, state: StructuralMetaState) -> StructuralGateDecision:
        """
        Compute gating decision from structural meta-state.
        Pure function. Never raises.
        """
        cfg = self._cfg
        inst = state.structural_instability_index

        # ── Aggressiveness (concave decay) ────────────────────────────────
        aggressiveness = max(0.0, 1.0 - inst ** 1.5)

        # ── Max suggestions (linear interpolation) ────────────────────────
        max_sugg = int(
            cfg.max_patch_suggestions
            - (cfg.max_patch_suggestions - cfg.min_patch_suggestions) * inst
        )
        max_sugg = max(cfg.min_patch_suggestions, min(cfg.max_patch_suggestions, max_sugg))

        # ── Sensitivity reduction (only when degraded) ────────────────────
        if inst > 0.30:
            sens_reduction = min(
                cfg.max_sensitivity_reduction,
                (inst - 0.30) / 0.70 * cfg.max_sensitivity_reduction,
            )
        else:
            sens_reduction = 0.0

        # ── Meta warning ──────────────────────────────────────────────────
        meta_warning = state.intervention_gate_active

        # ── Reason classification ─────────────────────────────────────────
        if inst < 0.30:
            reason = "healthy"
        elif inst < 0.55:
            reason = "caution"
        elif inst < 0.75:
            reason = "degraded"
        else:
            reason = "critical"

        return StructuralGateDecision(
            patch_aggressiveness=aggressiveness,
            max_suggestions=max_sugg,
            sensitivity_reduction=sens_reduction,
            meta_warning=meta_warning,
            reason=reason,
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def structural_state_to_meta_snapshot(state: StructuralMetaState) -> dict:
    """
    Extract structural health data for inclusion in meta_snapshot telemetry.

    Returns a dict suitable for merging into the CognitiveIdentityManager's
    periodic meta-snapshot output.
    """
    return {
        "structural_health": {
            "instability_index": round(state.structural_instability_index, 4),
            "composite_risk": round(state.ema_composite_risk, 4),
            "health_grade": state.last_health_grade,
            "trend": round(state.trend, 4),
            "gate_active": state.intervention_gate_active,
            "inspection_count": state.inspection_count,
        }
    }


def apply_structural_sensitivity_reduction(
    base_alpha: float,
    gate_decision: StructuralGateDecision,
) -> float:
    """
    Apply structural gating to Phase 5 adaptive alpha.

    When structural health is degraded, the system should be LESS responsive
    to behavioral noise (reduce alpha toward alpha_min).

    Formula:
        adjusted_alpha = base_alpha × (1.0 - sensitivity_reduction)

    This is a multiplicative reduction, bounded by the original alpha range.
    """
    reduction = gate_decision.sensitivity_reduction
    adjusted = base_alpha * (1.0 - reduction)
    return max(0.10, adjusted)  # never below Phase 5 alpha_min


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]. Returns lo on NaN, Inf, or non-numeric input."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, v))


def _parse_dt(raw) -> datetime:
    """Parse datetime from ISO string. Returns now() on failure."""
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            pass
    return datetime.now(UTC)


def _safe_dict_float(d: dict, key: str, default: float = 0.0) -> float:
    """Safely extract a float from a dict. Returns default on any error."""
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_dict_int(d: dict, key: str, default: int = 0) -> int:
    """Safely extract an int from a dict. Returns default on any error."""
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_int(obj, attr: str) -> int:
    """Safely extract an integer attribute from an object."""
    try:
        val = getattr(obj, attr, 0)
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(obj, attr: str) -> float:
    """Safely extract a float attribute from an object."""
    try:
        val = getattr(obj, attr, 0.0)
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_str(obj, attr: str, default: str = "") -> str:
    """Safely extract a string attribute from an object."""
    try:
        val = getattr(obj, attr, default)
        return str(val)
    except (TypeError, ValueError):
        return default


def _safe_len(obj, attr: str) -> int:
    """Safely get length of an iterable attribute."""
    try:
        val = getattr(obj, attr, ())
        return len(val)
    except (TypeError, AttributeError):
        return 0