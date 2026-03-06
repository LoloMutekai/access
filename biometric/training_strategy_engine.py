"""
A.C.C.E.S.S. — Training Strategy Engine (Phase B8 / 8.0.0)
biometric/training_strategy_engine.py

Transforms physiological state into an actionable, explainable training
strategy by evaluating several hypothetical training scenarios through the
Athlete Digital Twin simulation capability and selecting the safest and most
effective option.

───────────────────────────────────────────────────────────────────────────────
Pipeline position
───────────────────────────────────────────────────────────────────────────────

    raw signals
        → biometric metrics
        → recovery prediction
        → recovery memory correction
        → rule evolution
        → performance forecast
        → athlete digital twin
        → training strategy engine    ← THIS MODULE  (Phase 8.0.0)

───────────────────────────────────────────────────────────────────────────────
Inputs
───────────────────────────────────────────────────────────────────────────────

    digital_twin_state : dict   — from AthleteDigitalTwin.get_state()
    performance_forecast: dict  — from AthleteDigitalTwin.forecast()

    Core state variables consumed:
        fatigue_state       ∈ [0, 1]
        injury_risk_state   ∈ [0, 1]
        readiness_state     ∈ [0, 1]
        adaptation_factor   ∈ [0, 1]

    Forecast variables consumed:
        fatigue_24h, fatigue_48h, fatigue_72h
        injury_risk, readiness_score

───────────────────────────────────────────────────────────────────────────────
Training scenarios
───────────────────────────────────────────────────────────────────────────────

    REST            load = 0       recovery_hours = 24
    LIGHT_SESSION   load = 2000    recovery_hours = 18
    MODERATE_SESSION load = 4000   recovery_hours = 16
    INTENSE_SESSION  load = 7000   recovery_hours = 12

    Each scenario is simulated via AthleteDigitalTwin.simulate_training().

───────────────────────────────────────────────────────────────────────────────
Safety constraints  (applied before scoring, in priority order)
───────────────────────────────────────────────────────────────────────────────

    Priority 1 — injury_risk_state > INJURY_RISK_THRESHOLD (0.7)
        → Force REST.  No other scenario considered.

    Priority 2 — fatigue_state > FATIGUE_THRESHOLD (0.8)
        → Allow only REST and LIGHT_SESSION.

    Priority 3 (extends allowed set) — readiness_state > READINESS_THRESHOLD (0.75)
        → Also allow INTENSE_SESSION (when not blocked by P1 or P2).

    Default (no constraint triggered):
        → REST, LIGHT_SESSION, MODERATE_SESSION allowed.
           INTENSE_SESSION added only when readiness_state > READINESS_THRESHOLD.

───────────────────────────────────────────────────────────────────────────────
Scenario score formula
───────────────────────────────────────────────────────────────────────────────

    strategy_score =
        clamp(
            future_readiness
            − future_injury_risk * W_INJURY_RISK
            − fatigue_projection  * W_FATIGUE_48H,
            0.0, 1.0
        )

    Where (from simulate_training() output):
        future_readiness   = simulation["readiness_score"]
        future_injury_risk = simulation["injury_risk"]
        fatigue_projection = simulation["fatigue_48h"]

    Constants:
        W_INJURY_RISK = 0.8    — penalises scenarios that raise injury risk
        W_FATIGUE_48H = 0.5    — penalises scenarios that raise mid-term fatigue

───────────────────────────────────────────────────────────────────────────────
Decision selection
───────────────────────────────────────────────────────────────────────────────

    The scenario with the highest strategy_score among the allowed set is
    selected.  Ties are broken by scenario order (REST > LIGHT > MODERATE >
    INTENSE), which is inherently conservative.

───────────────────────────────────────────────────────────────────────────────
Confidence
───────────────────────────────────────────────────────────────────────────────

    confidence = clamp(best_score − second_best_score, 0.0, 1.0)

    When only one scenario is allowed (forced REST), confidence = 1.0.
    When best_score == second_best_score, confidence = 0.0.

───────────────────────────────────────────────────────────────────────────────
Reasoning
───────────────────────────────────────────────────────────────────────────────

    A deterministic set of human-readable strings is produced from the
    current state variables and the decision outcome.  The same inputs always
    produce the same reasoning strings (G6 determinism).

───────────────────────────────────────────────────────────────────────────────
Fallback (invalid inputs)
───────────────────────────────────────────────────────────────────────────────

    If the digital_twin_state or performance_forecast contain non-finite
    values, or if the AthleteDigitalTwin instance is unavailable, the engine
    returns a safe REST recommendation with confidence = 0.0.

───────────────────────────────────────────────────────────────────────────────
Mathematical guarantees
───────────────────────────────────────────────────────────────────────────────

    G1  strategy_score ∈ [0.0, 1.0] for every scenario
    G2  confidence ∈ [0.0, 1.0]
    G3  expected_readiness, expected_injury_risk ∈ [0.0, 1.0]
    G4  All output floats finite
    G5  Inputs never mutated
    G6  Deterministic — identical state inputs → identical decision
    G7  Bounded execution — O(S × N) where S = scenario count ≤ 4
    G8  Invalid inputs → safe REST fallback, never exception
    G9  Output is JSON-serialisable
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from biometric.athlete_digital_twin import AthleteDigitalTwin


# =============================================================================
# VERSION
# =============================================================================

ENGINE_NAME:    str = "TrainingStrategyEngine"
ENGINE_VERSION: str = "8.0.0"


# =============================================================================
# CONSTANTS
# =============================================================================

# ── Scenario definitions ───────────────────────────────────────────────────────
#: Number of built-in scenarios.
SCENARIO_COUNT: int = 4

# ── Score formula weights ──────────────────────────────────────────────────────
#: Penalty weight applied to future_injury_risk in the strategy score.
W_INJURY_RISK: float = 0.8

#: Penalty weight applied to fatigue_projection (fatigue_48h) in the strategy score.
W_FATIGUE_48H: float = 0.5

# ── Safety thresholds ─────────────────────────────────────────────────────────
#: injury_risk_state above this value forces REST (highest priority constraint).
INJURY_RISK_THRESHOLD: float = 0.7

#: fatigue_state above this value restricts scenarios to REST or LIGHT_SESSION.
FATIGUE_THRESHOLD: float = 0.8

#: readiness_state above this value unlocks INTENSE_SESSION.
READINESS_THRESHOLD: float = 0.75

# ── Confidence ────────────────────────────────────────────────────────────────
#: Confidence when only one scenario is available (forced decision).
FORCED_CONFIDENCE: float = 1.0

# ── Fallback load ─────────────────────────────────────────────────────────────
#: Recommended load for a REST decision.
REST_LOAD: float = 0.0


# =============================================================================
# TRAINING SCENARIO  (immutable definition)
# =============================================================================

@dataclass(frozen=True)
class TrainingScenario:
    """
    Immutable definition of one training scenario to be evaluated.

    Fields
    ──────
    name           : str    — human-readable identifier
    load           : float  — sprint load (arbitrary units, ≥ 0)
    recovery_hours : float  — recovery hours (≥ 0)

    The dataclass is frozen and fully hashable.
    """

    name:           str
    load:           float
    recovery_hours: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name":           self.name,
            "load":           self.load,
            "recovery_hours": self.recovery_hours,
        }

    def is_valid(self) -> bool:
        """Return True when load ≥ 0, recovery_hours ≥ 0 and both are finite."""
        return (
            isinstance(self.name, str) and len(self.name) > 0
            and math.isfinite(self.load)           and self.load >= 0.0
            and math.isfinite(self.recovery_hours) and self.recovery_hours >= 0.0
        )


# ── Built-in scenario catalogue ───────────────────────────────────────────────

#: Complete rest — no load, maximum recovery.
REST_SCENARIO: TrainingScenario = TrainingScenario(
    name="REST", load=0.0, recovery_hours=24.0
)

#: Light session — minimal load, good recovery.
LIGHT_SESSION_SCENARIO: TrainingScenario = TrainingScenario(
    name="LIGHT_SESSION", load=2_000.0, recovery_hours=18.0
)

#: Moderate session — balanced load and recovery.
MODERATE_SESSION_SCENARIO: TrainingScenario = TrainingScenario(
    name="MODERATE_SESSION", load=4_000.0, recovery_hours=16.0
)

#: Intense session — high load, shorter recovery.
INTENSE_SESSION_SCENARIO: TrainingScenario = TrainingScenario(
    name="INTENSE_SESSION", load=7_000.0, recovery_hours=12.0
)

#: Ordered catalogue of all built-in scenarios (REST first → most conservative).
ALL_SCENARIOS: tuple[TrainingScenario, ...] = (
    REST_SCENARIO,
    LIGHT_SESSION_SCENARIO,
    MODERATE_SESSION_SCENARIO,
    INTENSE_SESSION_SCENARIO,
)


# =============================================================================
# SCENARIO RESULT  (immutable evaluation of one scenario)
# =============================================================================

@dataclass(frozen=True)
class ScenarioResult:
    """
    Immutable evaluation of one training scenario.

    Fields
    ──────
    scenario           : TrainingScenario  — scenario definition
    future_readiness   : float             — readiness_score from simulation ∈ [0,1]
    future_injury_risk : float             — injury_risk from simulation      ∈ [0,1]
    fatigue_projection : float             — fatigue_48h from simulation      ∈ [0,1]
    strategy_score     : float             — composite score                  ∈ [0,1]

    Invariants
    ──────────
    - All floats finite and ∈ [0.0, 1.0].
    """

    scenario:           TrainingScenario
    future_readiness:   float
    future_injury_risk: float
    fatigue_projection: float
    strategy_score:     float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "scenario":           self.scenario.to_dict(),
            "future_readiness":   self.future_readiness,
            "future_injury_risk": self.future_injury_risk,
            "fatigue_projection": self.fatigue_projection,
            "strategy_score":     self.strategy_score,
        }

    def is_valid(self) -> bool:
        """Return True when all float fields are finite and ∈ [0, 1]."""
        floats = (
            self.future_readiness, self.future_injury_risk,
            self.fatigue_projection, self.strategy_score,
        )
        return self.scenario.is_valid() and all(
            math.isfinite(f) and 0.0 <= f <= 1.0 for f in floats
        )


# =============================================================================
# STRATEGY DECISION  (immutable final output)
# =============================================================================

@dataclass(frozen=True)
class StrategyDecision:
    """
    Immutable record of the engine's final training recommendation.

    Fields
    ──────
    recommended_session  : str            — scenario name
    recommended_load     : float          — load for the recommended session
    expected_readiness   : float          — projected readiness ∈ [0, 1]
    expected_injury_risk : float          — projected injury risk ∈ [0, 1]
    confidence           : float          — decision confidence ∈ [0, 1]
    reasoning            : tuple[str, ...]— deterministic explanation strings

    The dataclass is frozen so decisions can be safely cached or logged.

    Invariants
    ──────────
    - All floats finite and ∈ [0.0, 1.0].
    - reasoning is a non-empty tuple of non-empty strings.
    """

    recommended_session:  str
    recommended_load:     float
    expected_readiness:   float
    expected_injury_risk: float
    confidence:           float
    reasoning:            tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable dict.

        Note: ``reasoning`` is serialised as a list (JSON has no tuple type).
        """
        return {
            "recommended_session":  self.recommended_session,
            "recommended_load":     self.recommended_load,
            "expected_readiness":   self.expected_readiness,
            "expected_injury_risk": self.expected_injury_risk,
            "confidence":           self.confidence,
            "reasoning":            list(self.reasoning),
        }

    def is_valid(self) -> bool:
        """Return True when all fields satisfy the documented invariants."""
        floats = (
            self.expected_readiness, self.expected_injury_risk,
            self.confidence,
        )
        return (
            isinstance(self.recommended_session, str)
            and len(self.recommended_session) > 0
            and math.isfinite(self.recommended_load)
            and self.recommended_load >= 0.0
            and all(math.isfinite(f) and 0.0 <= f <= 1.0 for f in floats)
            and isinstance(self.reasoning, tuple)
            and len(self.reasoning) > 0
        )


# =============================================================================
# TRAINING STRATEGY ENGINE  (stateless)
# =============================================================================

class TrainingStrategyEngine:
    """
    Deterministic, stateless training strategy engine.

    Evaluates the four built-in training scenarios against the current
    Athlete Digital Twin state, applies safety constraints, and returns the
    highest-scoring allowed recommendation with a human-readable explanation.

    Stateless design
    ────────────────
    ``TrainingStrategyEngine`` holds no mutable state.  Every call to
    ``evaluate()`` derives its decision entirely from the supplied twin
    instance and does not alter that instance.

    Example
    ───────
        engine = TrainingStrategyEngine()
        twin   = AthleteDigitalTwin()
        twin.update({...})
        result = engine.evaluate(twin)
        print(result["recommended_session"])   # e.g. "LIGHT_SESSION"
    """

    def __init__(self) -> None:
        pass   # fully stateless

    # ── Primary API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        twin:     "AthleteDigitalTwin",
        forecast: dict | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate all allowed training scenarios and return the best decision.

        The method:
        1. Reads ``twin.get_state()`` for physiological state variables.
        2. Reads ``twin.forecast()`` (or the supplied ``forecast`` dict) for
           24/48/72 h projections.
        3. Determines the allowed scenario set via safety constraints.
        4. Simulates each allowed scenario via ``twin.simulate_training()``.
        5. Scores each scenario with the strategy formula.
        6. Selects the highest-scoring scenario (ties → most conservative).
        7. Computes confidence from the score margin.
        8. Generates deterministic reasoning strings.

        Parameters
        ──────────
        twin : AthleteDigitalTwin
            Live twin instance.  Never mutated.
        forecast : dict | None
            Pre-computed forecast dict.  When ``None`` the engine calls
            ``twin.forecast()`` internally.

        Returns
        ───────
        dict[str, Any]
            Keys: ``"recommended_session"``, ``"recommended_load"``,
            ``"expected_readiness"``, ``"expected_injury_risk"``,
            ``"confidence"``, ``"reasoning"``.
            All float values ∈ [0.0, 1.0] and finite.
        """
        # ── 1. Read twin state ─────────────────────────────────────────────
        try:
            state = twin.get_state()
        except Exception:
            return _safe_fallback_dict("could not read twin state")

        fatigue_state     = _safe_float(state.get("fatigue_state",     0.0))
        injury_risk_state = _safe_float(state.get("injury_risk_state", 0.0))
        readiness_state   = _safe_float(state.get("readiness_state",   1.0))
        adaptation_factor = _safe_float(state.get("adaptation_factor", 0.0))

        # Validate core state
        if not _state_is_valid(fatigue_state, injury_risk_state, readiness_state):
            return _safe_fallback_dict("invalid twin state values")

        # ── 2. Read forecast ───────────────────────────────────────────────
        if forecast is None:
            try:
                forecast = twin.forecast()
            except Exception:
                forecast = {}

        # ── 3. Filter allowed scenarios ────────────────────────────────────
        allowed = _filter_scenarios(
            fatigue_state, injury_risk_state, readiness_state
        )
        forced = len(allowed) == 1  # only REST

        # ── 4 & 5. Simulate + score each allowed scenario ──────────────────
        results: list[ScenarioResult] = []
        for scenario in allowed:
            try:
                sim = twin.simulate_training(scenario.load, scenario.recovery_hours)
            except Exception:
                sim = {}

            future_readiness   = _safe_float(sim.get("readiness_score", 0.0))
            future_injury_risk = _safe_float(sim.get("injury_risk",      1.0))
            fatigue_projection = _safe_float(sim.get("fatigue_48h",      1.0))

            score = _compute_score(future_readiness, future_injury_risk, fatigue_projection)

            results.append(ScenarioResult(
                scenario           = scenario,
                future_readiness   = future_readiness,
                future_injury_risk = future_injury_risk,
                fatigue_projection = fatigue_projection,
                strategy_score     = score,
            ))

        # ── 6. Select best scenario ────────────────────────────────────────
        # Sort descending by score; stable sort preserves original order on ties
        # (REST first → most conservative wins ties).
        sorted_results = sorted(results, key=lambda r: r.strategy_score, reverse=True)
        best = sorted_results[0]

        # ── 7. Confidence ──────────────────────────────────────────────────
        if forced or len(sorted_results) == 1:
            confidence = FORCED_CONFIDENCE
        else:
            confidence = _clamp(
                best.strategy_score - sorted_results[1].strategy_score,
                0.0, 1.0,
            )

        # ── 8. Reasoning ───────────────────────────────────────────────────
        reasoning = _generate_reasoning(
            fatigue_state     = fatigue_state,
            injury_risk_state = injury_risk_state,
            readiness_state   = readiness_state,
            adaptation_factor = adaptation_factor,
            allowed_scenarios = allowed,
            best_result       = best,
            forced            = forced,
        )

        return StrategyDecision(
            recommended_session  = best.scenario.name,
            recommended_load     = best.scenario.load,
            expected_readiness   = best.future_readiness,
            expected_injury_risk = best.future_injury_risk,
            confidence           = confidence,
            reasoning            = reasoning,
        ).to_dict()

    def evaluate_scenarios(
        self,
        twin: "AthleteDigitalTwin",
    ) -> list[dict[str, Any]]:
        """
        Return scored results for **all four** scenarios regardless of safety
        constraints.

        Useful for diagnostics, ranking visualisation, and testing.

        Parameters
        ──────────
        twin : AthleteDigitalTwin
            Live twin instance.  Never mutated.

        Returns
        ───────
        list[dict]
            One ``ScenarioResult.to_dict()`` per scenario, in the order
            ``(REST, LIGHT, MODERATE, INTENSE)``.
        """
        output: list[dict[str, Any]] = []
        for scenario in ALL_SCENARIOS:
            try:
                sim = twin.simulate_training(scenario.load, scenario.recovery_hours)
            except Exception:
                sim = {}

            future_readiness   = _safe_float(sim.get("readiness_score", 0.0))
            future_injury_risk = _safe_float(sim.get("injury_risk",      1.0))
            fatigue_projection = _safe_float(sim.get("fatigue_48h",      1.0))
            score = _compute_score(future_readiness, future_injury_risk, fatigue_projection)

            output.append(ScenarioResult(
                scenario           = scenario,
                future_readiness   = future_readiness,
                future_injury_risk = future_injury_risk,
                fatigue_projection = fatigue_projection,
                strategy_score     = score,
            ).to_dict())

        return output

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def self_test(self) -> dict:
        """
        Run six named invariant checks on synthetic probe twins.

        Checks
        ──────
        1. ``output_keys_always_present``
        2. ``all_outputs_in_unit_interval``
        3. ``forced_rest_on_high_injury_risk``
        4. ``fatigue_constraint_limits_scenarios``
        5. ``json_serialisable``
        6. ``deterministic_output``

        Returns
        ───────
        dict
            ``{"engine": str, "version": str, "checks": list[dict], "passed": bool}``
        """
        from biometric.athlete_digital_twin import AthleteDigitalTwin

        checks: list[dict] = []

        # 1 — required keys always present
        def _keys() -> bool:
            twin = AthleteDigitalTwin()
            r = TrainingStrategyEngine().evaluate(twin)
            return set(r) == {
                "recommended_session", "recommended_load",
                "expected_readiness", "expected_injury_risk",
                "confidence", "reasoning",
            }
        checks.append(_run_check("output_keys_always_present", _keys))

        # 2 — unit interval
        def _bounds() -> bool:
            twin = _make_probe_twin(15)
            r = TrainingStrategyEngine().evaluate(twin)
            for k in ("expected_readiness", "expected_injury_risk", "confidence"):
                if not (0.0 <= r[k] <= 1.0):
                    return False
            return True
        checks.append(_run_check("all_outputs_in_unit_interval", _bounds))

        # 3 — high injury risk → REST
        def _forced_rest() -> bool:
            twin = AthleteDigitalTwin()
            for _ in range(40):
                twin.update({"fatigue_index": 0.95, "sprint_load": 8000.0,
                             "recovery_hours": 0.0, "injury_flag": 1})
            r = TrainingStrategyEngine().evaluate(twin)
            return r["recommended_session"] == "REST"
        checks.append(_run_check("forced_rest_on_high_injury_risk", _forced_rest))

        # 4 — fatigue constraint
        def _fat_constraint() -> bool:
            twin = AthleteDigitalTwin()
            for _ in range(30):
                twin.update({"fatigue_index": 0.9, "sprint_load": 5000.0,
                             "recovery_hours": 8.0, "injury_flag": 0})
            r = TrainingStrategyEngine().evaluate(twin)
            return r["recommended_session"] in ("REST", "LIGHT_SESSION")
        checks.append(_run_check("fatigue_constraint_limits_scenarios", _fat_constraint))

        # 5 — JSON round-trip
        def _json_rt() -> bool:
            twin = _make_probe_twin(10)
            r = TrainingStrategyEngine().evaluate(twin)
            back = json.loads(json.dumps(r))
            return set(back) == set(r)
        checks.append(_run_check("json_serialisable", _json_rt))

        # 6 — determinism
        def _det() -> bool:
            twin = _make_probe_twin(20)
            eng  = TrainingStrategyEngine()
            return eng.evaluate(twin) == eng.evaluate(twin)
        checks.append(_run_check("deterministic_output", _det))

        all_passed = all(c["passed"] for c in checks)
        return {
            "engine":  ENGINE_NAME,
            "version": ENGINE_VERSION,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Run the canonical probe twin through evaluate() twice and compare.

        Returns
        ───────
        bool
            ``True`` when both calls produce bit-for-bit identical output.
        """
        twin = _make_probe_twin(25)
        eng  = TrainingStrategyEngine()
        return eng.evaluate(twin) == eng.evaluate(twin)


# =============================================================================
# PRIVATE — safety constraints
# =============================================================================

def _filter_scenarios(
    fatigue_state:     float,
    injury_risk_state: float,
    readiness_state:   float,
) -> list[TrainingScenario]:
    """
    Return the list of allowed scenarios given the current safety constraints.

    Priority order (highest first):
        P1: injury_risk_state > INJURY_RISK_THRESHOLD → only REST
        P2: fatigue_state > FATIGUE_THRESHOLD         → REST + LIGHT_SESSION
        P3: readiness_state > READINESS_THRESHOLD     → unlock INTENSE_SESSION
        Default: REST + LIGHT_SESSION + MODERATE_SESSION
                 (+ INTENSE if P3 satisfied)

    Parameters
    ──────────
    fatigue_state, injury_risk_state, readiness_state : float
        Current physiological state values ∈ [0, 1].

    Returns
    ───────
    list[TrainingScenario]
        Non-empty ordered list; REST is always present.
    """
    # P1 — critical injury risk overrides everything
    if injury_risk_state > INJURY_RISK_THRESHOLD:
        return [REST_SCENARIO]

    # P2 — high fatigue severely restricts
    if fatigue_state > FATIGUE_THRESHOLD:
        return [REST_SCENARIO, LIGHT_SESSION_SCENARIO]

    # Default allowed set
    allowed: list[TrainingScenario] = [
        REST_SCENARIO,
        LIGHT_SESSION_SCENARIO,
        MODERATE_SESSION_SCENARIO,
    ]

    # P3 — high readiness unlocks intense session
    if readiness_state > READINESS_THRESHOLD:
        allowed.append(INTENSE_SESSION_SCENARIO)

    return allowed


# =============================================================================
# PRIVATE — scoring
# =============================================================================

def _compute_score(
    future_readiness:   float,
    future_injury_risk: float,
    fatigue_projection: float,
) -> float:
    """
    Compute the composite strategy score for one scenario simulation result.

    Formula:
        clamp(
            future_readiness
            − future_injury_risk * W_INJURY_RISK
            − fatigue_projection  * W_FATIGUE_48H,
            0.0, 1.0
        )

    Parameters
    ──────────
    future_readiness   : float  — sim["readiness_score"]  ∈ [0, 1]
    future_injury_risk : float  — sim["injury_risk"]       ∈ [0, 1]
    fatigue_projection : float  — sim["fatigue_48h"]       ∈ [0, 1]

    Returns
    ───────
    float
        Strategy score ∈ [0.0, 1.0].
    """
    raw = (
        future_readiness
        - future_injury_risk * W_INJURY_RISK
        - fatigue_projection  * W_FATIGUE_48H
    )
    return _clamp(raw, 0.0, 1.0)


# =============================================================================
# PRIVATE — reasoning
# =============================================================================

def _generate_reasoning(
    fatigue_state:     float,
    injury_risk_state: float,
    readiness_state:   float,
    adaptation_factor: float,
    allowed_scenarios: list[TrainingScenario],
    best_result:       ScenarioResult,
    forced:            bool,
) -> tuple[str, ...]:
    """
    Generate a deterministic ordered tuple of human-readable reasoning strings.

    The strings are produced by evaluating each condition in a fixed order,
    so identical inputs always produce identical reasoning (G6 guarantee).

    Returns
    ───────
    tuple[str, ...]
        Non-empty tuple of non-empty strings.
    """
    reasons: list[str] = []

    # ── Fatigue assessment ────────────────────────────────────────────────────
    if fatigue_state > FATIGUE_THRESHOLD:
        reasons.append(
            f"fatigue_state critical ({fatigue_state:.2f}): heavy restriction applied"
        )
    elif fatigue_state > 0.5:
        reasons.append(f"fatigue_state elevated ({fatigue_state:.2f})")
    else:
        reasons.append(f"fatigue_state low ({fatigue_state:.2f}): athlete well-rested")

    # ── Injury risk assessment ────────────────────────────────────────────────
    if injury_risk_state > INJURY_RISK_THRESHOLD:
        reasons.append(
            f"injury risk critical ({injury_risk_state:.2f}): REST enforced"
        )
    elif injury_risk_state > 0.4:
        reasons.append(f"injury risk moderate ({injury_risk_state:.2f})")
    else:
        reasons.append(f"injury risk low ({injury_risk_state:.2f})")

    # ── Readiness assessment ──────────────────────────────────────────────────
    if readiness_state > READINESS_THRESHOLD:
        reasons.append(
            f"readiness high ({readiness_state:.2f}): intense training unlocked"
        )
    elif readiness_state > 0.4:
        reasons.append(f"readiness moderate ({readiness_state:.2f})")
    else:
        reasons.append(f"readiness low ({readiness_state:.2f}): conservative approach applied")

    # ── Adaptation factor note ────────────────────────────────────────────────
    if adaptation_factor > 0.6:
        reasons.append(
            f"adaptation factor high ({adaptation_factor:.2f}): athlete tolerates high load"
        )
    elif adaptation_factor > 0.3:
        reasons.append(f"adaptation factor moderate ({adaptation_factor:.2f})")
    else:
        reasons.append(
            f"adaptation factor low ({adaptation_factor:.2f}): training load tolerance limited"
        )

    # ── Constraint summary ────────────────────────────────────────────────────
    if forced:
        reasons.append("safety constraint active: only REST permitted")
    else:
        scenario_names = [s.name for s in allowed_scenarios]
        reasons.append(
            f"{len(allowed_scenarios)} scenario(s) evaluated: {', '.join(scenario_names)}"
        )

    # ── Decision rationale ────────────────────────────────────────────────────
    reasons.append(
        f"{best_result.scenario.name} selected "
        f"(score={best_result.strategy_score:.3f}, "
        f"readiness={best_result.future_readiness:.3f}, "
        f"injury_risk={best_result.future_injury_risk:.3f})"
    )

    return tuple(reasons)


# =============================================================================
# PRIVATE — fallback
# =============================================================================

def _safe_fallback_dict(reason: str = "invalid input") -> dict[str, Any]:
    """
    Return a safe REST recommendation when inputs are invalid or raise.

    confidence = 0.0 signals that this is a fallback, not a scored decision.
    """
    return StrategyDecision(
        recommended_session  = REST_SCENARIO.name,
        recommended_load     = REST_LOAD,
        expected_readiness   = 0.0,
        expected_injury_risk = 1.0,
        confidence           = 0.0,
        reasoning            = (f"safe fallback: {reason}",),
    ).to_dict()


# =============================================================================
# PRIVATE — validation
# =============================================================================

def _state_is_valid(
    fatigue_state:     float,
    injury_risk_state: float,
    readiness_state:   float,
) -> bool:
    """Return True when all three core state values are finite."""
    return (
        math.isfinite(fatigue_state)
        and math.isfinite(injury_risk_state)
        and math.isfinite(readiness_state)
    )


# =============================================================================
# PRIVATE — probe twin for diagnostics
# =============================================================================

def _make_probe_twin(n: int) -> "AthleteDigitalTwin":
    """
    Create a pre-loaded deterministic probe twin for self_test / check methods.

    No randomness — linear sweep from low-to-high load.
    """
    from biometric.athlete_digital_twin import AthleteDigitalTwin
    twin = AthleteDigitalTwin(athlete_id="probe")
    for i in range(n):
        t = i / max(n - 1, 1)
        twin.update({
            "fatigue_index":  round(0.2 + 0.4 * t, 6),
            "sprint_load":    round(2_000.0 + 3_000.0 * t, 2),
            "recovery_hours": round(30.0 - 18.0 * t, 2),
            "injury_flag":    0,
        })
    return twin


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp ``value`` to ``[lo, hi]``.

    Returns ``lo`` for any non-finite input so a corrupt intermediate cannot
    propagate beyond the module boundary.
    """
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    """
    Convert ``value`` to a finite float, returning ``default`` on failure.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _run_check(name: str, fn) -> dict:
    """Execute a named boolean check, catching all exceptions."""
    try:
        passed = bool(fn())
        detail = "pass" if passed else "assertion returned False"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": passed, "detail": detail}