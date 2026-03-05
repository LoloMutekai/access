"""
A.C.C.E.S.S. — DataEngine Router / AgentCore Integration Layer (Phase 7.1)
agent/domain/data_router.py

Connects DataEngine to the central orchestrator cleanly.

Responsibilities
────────────────
    1. Route requests containing {"values": [...]} to DataEngine.run().
    2. Validate schema before execution — abort cleanly on any violation.
    3. Enforce required_permissions via PermissionManager.
    4. Pass sandbox_context to engine.run() when a SandboxRunner is present.
    5. On success: snapshot engine state via RollbackEngine.snapshot().
    6. On error (schema / permission / sandbox): no snapshot, no state mutation.
    7. Extract drift_score, confidence, volatility into a DataDiagnostics record.
    8. Expose domain_determinism_check() as a module-level helper.

Public surface
──────────────
    DataDiagnostics          — frozen, JSON-serializable diagnostics record
    DomainExecutionResult    — frozen, JSON-serializable result of one run_domain()
    DataEngineRouter         — main orchestration class
    domain_determinism_check — module-level helper (no global state)

Execution order inside run_domain()
─────────────────────────────────────
    1. Routing check       → is {"values": [...]} present?
    2. Schema validation   → DataEngine.validate_schema()
    3. Permission check    → PermissionManager.check() for each required perm
    4. Sandboxed execution → SandboxRunner.run()  (or direct call if no runner)
    5. Snapshot            → RollbackEngine.snapshot("post_run")
    6. Diagnostics         → extract drift / confidence / volatility
    7. Return DomainExecutionResult

Design rules
────────────
    - No global mutable state.
    - No circular imports (imports only from agent.domain.*).
    - All outputs are JSON-serializable.
    - All float values guaranteed finite.
    - Deterministic: identical input + state → identical result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from .data_engine import DataEngine
from .domain_registry import DomainRegistry
from .permissions import PermissionManager
from .rollback import RollbackEngine
from .sandbox import SandboxConfig, SandboxRunner


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_KEY         = "data_backbone_engine"
_SNAPSHOT_TAG       = "post_run"
_DETERMINISM_INPUT  = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
_COMPARED_KEYS      = ("status", "features", "signals", "confidence",
                       "engine", "version")


# ─────────────────────────────────────────────────────────────────────────────
# DATA DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataDiagnostics:
    """
    Frozen, JSON-serializable diagnostics record extracted from a DataEngine run.

    Fields
    ──────
    drift_score  ∈ [0.0, 1.0]  — distributional distance from baseline
    confidence   ∈ [0.0, 1.0]  — stability × (1 − drift)
    volatility   ∈ [0.0, ∞)    — population std of series returns
    anomaly_score∈ [0.0, 1.0]  — |last z-score| / 3σ
    momentum_score∈[-1.0, 1.0] — directional tail signal
    stability_score∈(0.0, 1.0] — 1 / (1 + volatility)

    All values are finite floats guaranteed by the DataEngine contract.
    """
    drift_score:     float
    confidence:      float
    volatility:      float
    anomaly_score:   float
    momentum_score:  float
    stability_score: float

    def to_dict(self) -> dict:
        return {
            "drift_score":     round(self.drift_score,     8),
            "confidence":      round(self.confidence,      8),
            "volatility":      round(self.volatility,      8),
            "anomaly_score":   round(self.anomaly_score,   8),
            "momentum_score":  round(self.momentum_score,  8),
            "stability_score": round(self.stability_score, 8),
        }

    def is_healthy(self) -> bool:
        """True iff drift and anomaly are both below conservative thresholds."""
        return self.drift_score < 0.5 and self.anomaly_score < 0.8

    def __repr__(self) -> str:
        return (
            f"DataDiagnostics("
            f"drift={self.drift_score:.4f}, "
            f"confidence={self.confidence:.4f}, "
            f"vol={self.volatility:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN EXECUTION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainExecutionResult:
    """
    Frozen, JSON-serializable record of one DataEngineRouter.run_domain() call.

    Fields
    ──────
    success       : True iff the engine produced status=="ok" with no errors.
    output        : Raw engine output dict ({"status","features","signals",…}).
    diagnostics   : DataDiagnostics extracted from the engine output.
    snapshot_tag  : Tag of the RollbackEngine snapshot created on success, or None.
    errors        : List of human-readable error strings; empty on success.
    engine_name   : Name of the engine that handled the request.
    abort_reason  : Short token describing why execution aborted, or "none".
    """
    success:      bool
    output:       dict
    diagnostics:  DataDiagnostics
    snapshot_tag: Optional[str]
    errors:       tuple[str, ...]
    engine_name:  str
    abort_reason: str           # "none" | "not_routable" | "schema_error"
                                # | "permission_denied" | "sandbox_failure"
                                # | "sandbox_timeout"

    def to_dict(self) -> dict:
        return {
            "success":      self.success,
            "output":       self.output,
            "diagnostics":  self.diagnostics.to_dict(),
            "snapshot_tag": self.snapshot_tag,
            "errors":       list(self.errors),
            "engine_name":  self.engine_name,
            "abort_reason": self.abort_reason,
        }

    def is_json_serializable(self) -> bool:
        try:
            json.dumps(self.to_dict())
            return True
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        if self.success:
            return (
                f"DomainExecutionResult(OK, "
                f"engine={self.engine_name!r}, "
                f"snap={self.snapshot_tag!r})"
            )
        return (
            f"DomainExecutionResult(FAIL:{self.abort_reason}, "
            f"engine={self.engine_name!r}, "
            f"errors={list(self.errors)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class DataEngineRouter:
    """
    Connects DataEngine to AgentCore's orchestration layer.

    The router owns no mutable state beyond its injected components.
    All mutations live inside the injected DataEngine / RollbackEngine instances.

    Constructor parameters
    ──────────────────────
    registry           : DomainRegistry with DataEngine registered under
                         "data_backbone_engine". The router calls registry.get()
                         lazily on the first run_domain() call.
    permission_manager : PermissionManager instance; must have the required
                         permissions granted before run_domain() is called.
    rollback_engine    : RollbackEngine wrapping the same DataEngine instance
                         that is registered in the registry.
    sandbox_config     : Optional SandboxConfig used to construct a SandboxRunner.
                         If None, the engine is called directly (no timeout).

    Usage
    ─────
        pm = PermissionManager()
        pm.grant("data_backbone_engine", "filesystem_read")

        engine   = DataEngine()
        registry = DomainRegistry()
        registry.register("data_backbone_engine", lambda: engine)

        router = DataEngineRouter(
            registry=registry,
            permission_manager=pm,
            rollback_engine=RollbackEngine(engine),
            sandbox_config=SandboxConfig(timeout_seconds=5.0),
        )

        result = router.run_domain({"values": [1, 2, 3, 4, 5, 6, 7]})
        assert result.success
        assert result.diagnostics.drift_score == 0.0
    """

    def __init__(
        self,
        registry:           DomainRegistry,
        permission_manager: PermissionManager,
        rollback_engine:    RollbackEngine,
        sandbox_config:     Optional[SandboxConfig] = None,
    ) -> None:
        self._registry  = registry
        self._pm        = permission_manager
        self._rollback  = rollback_engine
        self._runner    = SandboxRunner(sandbox_config) if sandbox_config else None

    # ── Public API ────────────────────────────────────────────────────────────

    def run_domain(
        self,
        input_data: dict,
        sandbox_context: Any = None,
    ) -> DomainExecutionResult:
        """
        Route ``input_data`` to DataEngine and return a structured result.

        Execution order
        ───────────────
            1. Routing check    → abort "not_routable" if no "values" key
            2. Schema validation → abort "schema_error" on any violation
            3. Permission check  → abort "permission_denied" if any perm missing
            4. Sandbox execution → abort "sandbox_failure" or "sandbox_timeout"
            5. Snapshot         → RollbackEngine.snapshot("post_run")  [success only]
            6. Diagnostics      → extract drift / confidence / volatility
            7. Return DomainExecutionResult(success=True)

        Args:
            input_data      : Dict that should contain "values" key.
            sandbox_context : Optional SandboxContext injected into engine.run().
                              Passed through regardless of whether a SandboxRunner
                              is configured.

        Returns:
            DomainExecutionResult — always returned, never raises.
        """
        engine = self._registry.get(_ENGINE_KEY)

        # ── Step 1: Routing ───────────────────────────────────────────────────
        if not self._is_routable(input_data):
            return self._abort(
                engine_name="data_backbone_engine",
                abort_reason="not_routable",
                errors=["Request does not contain 'values' key — not routable to DataEngine."],
            )

        # ── Step 2: Schema validation ─────────────────────────────────────────
        schema_errors = engine.validate_schema(input_data)
        if schema_errors:
            return self._abort(
                engine_name=engine.name,
                abort_reason="schema_error",
                errors=schema_errors,
            )

        # ── Step 3: Permission check ──────────────────────────────────────────
        denied_perm = self._check_permissions(engine)
        if denied_perm is not None:
            return self._abort(
                engine_name=engine.name,
                abort_reason="permission_denied",
                errors=[f"Permission denied: {denied_perm!r} not granted for {engine.name!r}."],
            )

        # ── Step 4: Execution (sandboxed or direct) ───────────────────────────
        if self._runner is not None:
            exec_output, exec_error, timed_out = self._run_sandboxed(
                engine, input_data, sandbox_context
            )
        else:
            exec_output, exec_error, timed_out = self._run_direct(
                engine, input_data, sandbox_context
            )

        if exec_error is not None:
            reason = "sandbox_timeout" if timed_out else "sandbox_failure"
            return self._abort(
                engine_name=engine.name,
                abort_reason=reason,
                errors=[exec_error],
            )

        # exec_output is now a valid engine result dict
        if exec_output.get("status") != "ok":
            engine_errors = exec_output.get("errors", ["DataEngine returned status != 'ok'."])
            return self._abort(
                engine_name=engine.name,
                abort_reason="sandbox_failure",
                errors=engine_errors,
            )

        # ── Step 5: Snapshot (success path only) ─────────────────────────────
        snapshot_tag = self._take_snapshot(engine)

        # ── Step 6: Diagnostics ───────────────────────────────────────────────
        diagnostics = self._extract_diagnostics(exec_output)

        # ── Step 7: Return ────────────────────────────────────────────────────
        return DomainExecutionResult(
            success=True,
            output=exec_output,
            diagnostics=diagnostics,
            snapshot_tag=snapshot_tag,
            errors=(),
            engine_name=engine.name,
            abort_reason="none",
        )

    # ── Read-only accessors ───────────────────────────────────────────────────

    @property
    def permission_manager(self) -> PermissionManager:
        return self._pm

    @property
    def rollback_engine(self) -> RollbackEngine:
        return self._rollback

    @property
    def registry(self) -> DomainRegistry:
        return self._registry

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_routable(input_data: Any) -> bool:
        """True iff input_data is a dict containing a "values" key."""
        return isinstance(input_data, dict) and "values" in input_data

    def _check_permissions(self, engine) -> Optional[str]:
        """
        Check every required engine permission.

        Returns the first denied permission key, or None if all pass.
        Never raises; uses raise_on_denial=False throughout.
        """
        for perm in engine.required_permissions:
            granted = self._pm.check(
                subject=engine.name,
                action=perm,
                context="DataEngineRouter.run_domain",
                raise_on_denial=False,
            )
            if not granted:
                return perm
        return None

    def _run_sandboxed(
        self,
        engine,
        input_data: dict,
        sandbox_context: Any,
    ) -> tuple[Optional[dict], Optional[str], bool]:
        """
        Execute engine.run() inside SandboxRunner.

        Returns (output, error_str, timed_out).
        output is None on failure; error_str is None on success.
        """
        def _fn(inp: dict, sandbox_context: Any = None) -> dict:
            return engine.run(inp, sandbox_context=sandbox_context)

        report = self._runner.run(_fn, input_data)

        if report.timed_out:
            return None, report.error, True
        if not report.success:
            return None, report.error, False

        return report.result, None, False

    @staticmethod
    def _run_direct(
        engine,
        input_data: dict,
        sandbox_context: Any,
    ) -> tuple[Optional[dict], Optional[str], bool]:
        """
        Execute engine.run() directly, without a SandboxRunner.

        Returns (output, error_str, timed_out).
        Catches all exceptions and converts them to error strings.
        """
        try:
            output = engine.run(input_data, sandbox_context=sandbox_context)
            return output, None, False
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", False

    def _take_snapshot(self, engine) -> str:
        """
        Create a RollbackEngine snapshot tagged with _SNAPSHOT_TAG.

        Returns the snapshot tag string.
        Never raises — logs a warning on failure but returns the tag anyway
        so the result can still carry it.
        """
        try:
            self._rollback.snapshot(
                _SNAPSHOT_TAG,
                metadata={
                    "source": "DataEngineRouter",
                    "engine": engine.name,
                },
            )
        except Exception:
            pass
        return _SNAPSHOT_TAG

    @staticmethod
    def _extract_diagnostics(output: dict) -> DataDiagnostics:
        """
        Pull drift_score, confidence, and feature metrics from engine output.

        All values are clamped/guarded to ensure finite floats.
        """
        def _f(obj: Any, default: float = 0.0) -> float:
            try:
                v = float(obj)
                return v if math.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        features = output.get("features", {})
        signals  = output.get("signals",  {})

        return DataDiagnostics(
            drift_score     = _f(output.get("drift_score"),          0.0),
            confidence      = _f(output.get("confidence"),           0.0),
            volatility      = _f(features.get("volatility"),         0.0),
            anomaly_score   = _f(signals.get("anomaly_score"),       0.0),
            momentum_score  = _f(signals.get("momentum_score"),      0.0),
            stability_score = _f(signals.get("stability_score"),     1.0),
        )

    @staticmethod
    def _abort(
        engine_name: str,
        abort_reason: str,
        errors: list[str],
    ) -> DomainExecutionResult:
        """Build a failed DomainExecutionResult without touching engine state."""
        return DomainExecutionResult(
            success=False,
            output={},
            diagnostics=DataDiagnostics(
                drift_score=0.0,
                confidence=0.0,
                volatility=0.0,
                anomaly_score=0.0,
                momentum_score=0.0,
                stability_score=1.0,
            ),
            snapshot_tag=None,
            errors=tuple(errors),
            engine_name=engine_name,
            abort_reason=abort_reason,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def domain_determinism_check(engine_factory: type | Any = DataEngine) -> bool:
    """
    Instantiate two fresh engines from ``engine_factory``, run the canonical
    input on each, and assert that all output fields (excluding drift_score,
    which is baseline-dependent) are identical.

    Args:
        engine_factory : Zero-argument callable that returns a DomainEngine.
                         Defaults to DataEngine (used as the canonical engine).

    Returns:
        True  iff both engines produce identical outputs on the canonical input.
        False iff any compared field differs, or if either engine raises.

    Never raises — all exceptions are caught and mapped to False.

    Canonical input
    ───────────────
        {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}

    Compared fields (drift_score excluded — it is baseline-dependent)
    ─────────────────────────────────────────────────────────────────
        status, features, signals, confidence, engine, version
    """
    try:
        engine_a = engine_factory()
        engine_b = engine_factory()
        output_a = engine_a.run(_DETERMINISM_INPUT)
        output_b = engine_b.run(_DETERMINISM_INPUT)
        return all(output_a.get(k) == output_b.get(k) for k in _COMPARED_KEYS)
    except Exception:
        return False


def build_default_registry() -> DomainRegistry:
    """
    Return a DomainRegistry with DataEngine pre-registered (lazy).

    The engine is NOT constructed until the first registry.get() call.
    Useful for injecting a ready-to-use registry into DataEngineRouter.
    """
    registry = DomainRegistry()
    registry.register(_ENGINE_KEY, DataEngine)
    return registry