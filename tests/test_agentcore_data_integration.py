"""
A.C.C.E.S.S. — AgentCore / DataEngine Integration Tests (Phase 7.1)
tests/test_agentcore_data_integration.py

Validates the full integration layer introduced in Phase 7.1:
    DomainRegistry  — lazy engine construction and caching
    DataEngineRouter — orchestration: routing, schema, permissions,
                       sandbox, snapshot, diagnostics
    DataDiagnostics  — frozen, JSON-serializable diagnostics record
    DomainExecutionResult — frozen result envelope
    domain_determinism_check — module-level determinism helper

Test sections
─────────────
     1. DomainRegistry — construction, caching, eviction
     2. Routing logic  — is_routable, not_routable abort
     3. Schema validation integration
     4. Permission enforcement
     5. Successful execution path
     6. Sandbox enforcement
     7. Rollback / snapshot creation
     8. MetaDiagnostics injection (DataDiagnostics record)
     9. Determinism check
    10. Schema error handled safely (no state mutation)
    11. DomainExecutionResult contract
    12. DataDiagnostics contract
    13. build_default_registry helper
    14. Edge cases
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import pytest

from agent.domain.data_engine import DataEngine
from agent.domain.data_router import (
    DataDiagnostics,
    DataEngineRouter,
    DomainExecutionResult,
    build_default_registry,
    domain_determinism_check,
)
from agent.domain.domain_registry import DomainRegistry
from agent.domain.permissions import PermissionManager
from agent.domain.rollback import RollbackEngine
from agent.domain.sandbox import SandboxConfig, SandboxContext


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES / HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_CANONICAL = {"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
_SHORT     = {"values": [1.0, 2.0]}                  # schema error
_NO_VALUES = {"query": "hello"}                       # not routable
_FLAT      = {"values": [5.0] * 7}


def _make_router(
    *,
    grant: bool = True,
    timeout: float = 5.0,
    use_sandbox: bool = True,
    engine: Optional[DataEngine] = None,
) -> tuple[DataEngineRouter, DataEngine, PermissionManager, RollbackEngine]:
    """
    Return a fresh (router, engine, pm, rollback) quad every call.
    No shared state between callers.
    """
    eng = engine or DataEngine()
    pm  = PermissionManager()
    if grant:
        for perm in eng.required_permissions:
            pm.grant(eng.name, perm)

    registry = DomainRegistry()
    registry.register("data_backbone_engine", lambda e=eng: e)

    rb  = RollbackEngine(eng)
    cfg = SandboxConfig(timeout_seconds=timeout) if use_sandbox else None
    router = DataEngineRouter(
        registry=registry,
        permission_manager=pm,
        rollback_engine=rb,
        sandbox_config=cfg,
    )
    return router, eng, pm, rb


def _slow_engine(base: DataEngine, sleep: float = 5.0) -> DataEngine:
    """Patch engine.run() to sleep before returning."""
    orig = base.run

    def _slow(inp, sandbox_context=None):
        time.sleep(sleep)
        return orig(inp, sandbox_context=sandbox_context)

    base.run = _slow  # type: ignore[method-assign]
    return base


def _crash_engine(base: DataEngine) -> DataEngine:
    """Patch engine.run() to raise a RuntimeError."""
    base.run = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("simulated crash")
    )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1 — DOMAIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainRegistry:

    def test_register_and_get(self):
        reg = DomainRegistry()
        reg.register("my_engine", DataEngine)
        engine = reg.get("my_engine")
        assert isinstance(engine, DataEngine)

    def test_get_returns_cached_instance(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        e1 = reg.get("e")
        e2 = reg.get("e")
        assert e1 is e2

    def test_unknown_key_raises(self):
        reg = DomainRegistry()
        with pytest.raises(KeyError):
            reg.get("does_not_exist")

    def test_duplicate_register_raises(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        with pytest.raises(KeyError):
            reg.register("e", DataEngine)

    def test_register_or_replace_evicts_cached(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        e1 = reg.get("e")
        reg.register_or_replace("e", DataEngine)
        e2 = reg.get("e")
        assert e1 is not e2

    def test_has_registered(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        assert reg.has("e")
        assert not reg.has("other")

    def test_is_loaded_before_get(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        assert not reg.is_loaded("e")

    def test_is_loaded_after_get(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        reg.get("e")
        assert reg.is_loaded("e")

    def test_evict_resets_loaded(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        reg.get("e")
        assert reg.is_loaded("e")
        reg.evict("e")
        assert not reg.is_loaded("e")

    def test_evict_returns_false_when_not_loaded(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        assert reg.evict("e") is False

    def test_list_registered(self):
        reg = DomainRegistry()
        reg.register("z", DataEngine)
        reg.register("a", DataEngine)
        assert reg.list_registered() == ["a", "z"]

    def test_list_loaded_empty_before_get(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        assert reg.list_loaded() == []

    def test_unregister_removes_factory_and_instance(self):
        reg = DomainRegistry()
        reg.register("e", DataEngine)
        reg.get("e")
        reg.unregister("e")
        assert not reg.has("e")
        assert not reg.is_loaded("e")

    def test_clear_removes_everything(self):
        reg = DomainRegistry()
        reg.register("a", DataEngine)
        reg.register("b", DataEngine)
        reg.get("a")
        reg.clear()
        assert reg.list_registered() == []
        assert reg.list_loaded() == []

    def test_lazy_construction_not_triggered_at_register(self):
        """Factory must NOT be called at register() time."""
        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            return DataEngine()

        reg = DomainRegistry()
        reg.register("e", factory)
        assert call_count["n"] == 0   # not yet

    def test_factory_called_once_on_first_get(self):
        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            return DataEngine()

        reg = DomainRegistry()
        reg.register("e", factory)
        reg.get("e")
        reg.get("e")
        reg.get("e")
        assert call_count["n"] == 1   # exactly once


# ─────────────────────────────────────────────────────────────────────────────
# 2 — ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingLogic:

    def test_values_key_routes_to_engine(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert result.success

    def test_missing_values_key_not_routable(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_NO_VALUES)
        assert result.abort_reason == "not_routable"
        assert not result.success

    def test_empty_dict_not_routable(self):
        r, _, _, _ = _make_router()
        assert r.run_domain({}).abort_reason == "not_routable"

    def test_non_dict_input_not_routable(self):
        r, _, _, _ = _make_router()
        result = r.run_domain([1, 2, 3, 4, 5])  # type: ignore[arg-type]
        assert result.abort_reason == "not_routable"

    def test_not_routable_has_no_snapshot(self):
        r, _, _, rb = _make_router()
        r.run_domain(_NO_VALUES)
        assert rb.snapshot_count == 0

    def test_not_routable_engine_name_in_result(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_NO_VALUES)
        assert result.engine_name == "data_backbone_engine"

    def test_not_routable_errors_is_non_empty(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_NO_VALUES)
        assert len(result.errors) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3 — SCHEMA VALIDATION INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidationIntegration:

    def test_too_few_values_returns_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_SHORT)
        assert result.abort_reason == "schema_error"
        assert not result.success

    def test_schema_error_has_errors_list(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_SHORT)
        assert len(result.errors) >= 1
        assert all(isinstance(e, str) for e in result.errors)

    def test_nan_value_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1, 2, 3, 4, float("nan")]})
        assert result.abort_reason == "schema_error"

    def test_none_value_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1, 2, 3, 4, None]})
        assert result.abort_reason == "schema_error"

    def test_nested_value_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1, 2, 3, 4, [5]]})
        assert result.abort_reason == "schema_error"

    def test_schema_error_no_state_mutation(self):
        """Engine state must not change on a schema error."""
        r, eng, _, rb = _make_router()
        count_before = eng.run_count
        r.run_domain(_SHORT)
        assert eng.run_count == count_before

    def test_schema_error_no_snapshot(self):
        r, _, _, rb = _make_router()
        r.run_domain(_SHORT)
        assert rb.snapshot_count == 0

    def test_schema_error_output_dict_json_serializable(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_SHORT)
        json.dumps(result.to_dict())

    def test_valid_schema_not_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert result.abort_reason != "schema_error"


# ─────────────────────────────────────────────────────────────────────────────
# 4 — PERMISSION ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionEnforcement:

    def test_no_permissions_denied(self):
        r, _, _, _ = _make_router(grant=False)
        result = r.run_domain(_CANONICAL)
        assert result.abort_reason == "permission_denied"
        assert not result.success

    def test_denied_errors_mention_permission(self):
        r, _, _, _ = _make_router(grant=False)
        result = r.run_domain(_CANONICAL)
        assert any("filesystem_read" in err or "denied" in err.lower()
                   for err in result.errors)

    def test_denied_no_snapshot(self):
        r, _, _, rb = _make_router(grant=False)
        r.run_domain(_CANONICAL)
        assert rb.snapshot_count == 0

    def test_denied_no_engine_state_mutation(self):
        r, eng, _, _ = _make_router(grant=False)
        count_before = eng.run_count
        r.run_domain(_CANONICAL)
        assert eng.run_count == count_before

    def test_granted_permissions_allow_execution(self):
        r, _, _, _ = _make_router(grant=True)
        result = r.run_domain(_CANONICAL)
        assert result.success

    def test_audit_log_records_check(self):
        r, eng, pm, _ = _make_router(grant=False)
        r.run_domain(_CANONICAL)
        assert len(pm.audit_log.filter_denied()) >= 1

    def test_grant_after_denial_succeeds(self):
        r, eng, pm, _ = _make_router(grant=False)
        res1 = r.run_domain(_CANONICAL)
        assert not res1.success

        for perm in eng.required_permissions:
            pm.grant(eng.name, perm)

        res2 = r.run_domain(_CANONICAL)
        assert res2.success


# ─────────────────────────────────────────────────────────────────────────────
# 5 — SUCCESSFUL EXECUTION PATH
# ─────────────────────────────────────────────────────────────────────────────

class TestSuccessfulExecution:

    def test_success_true(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_CANONICAL).success

    def test_abort_reason_none(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_CANONICAL).abort_reason == "none"

    def test_output_status_ok(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_CANONICAL).output["status"] == "ok"

    def test_engine_name_in_result(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_CANONICAL).engine_name == "data_backbone_engine"

    def test_errors_empty_on_success(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_CANONICAL).errors == ()

    def test_engine_run_count_incremented(self):
        r, eng, _, _ = _make_router()
        r.run_domain(_CANONICAL)
        assert eng.run_count > 0

    def test_direct_execution_without_sandbox(self):
        """Router with sandbox_config=None runs engine directly."""
        r, _, _, _ = _make_router(use_sandbox=False)
        result = r.run_domain(_CANONICAL)
        assert result.success

    def test_flat_series_succeeds(self):
        r, _, _, _ = _make_router()
        assert r.run_domain(_FLAT).success

    def test_output_contains_required_keys(self):
        r, _, _, _ = _make_router()
        out = r.run_domain(_CANONICAL).output
        for key in ("status", "features", "signals", "drift_score",
                    "confidence", "engine", "version"):
            assert key in out


# ─────────────────────────────────────────────────────────────────────────────
# 6 — SANDBOX ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxEnforcement:

    def test_sandbox_timeout_detected(self):
        eng = _slow_engine(DataEngine(), sleep=5.0)
        r, _, _, _ = _make_router(engine=eng, timeout=0.1)
        result = r.run_domain(_CANONICAL)
        assert result.abort_reason == "sandbox_timeout"
        assert not result.success

    def test_sandbox_crash_detected(self):
        eng = _crash_engine(DataEngine())
        r, _, _, _ = _make_router(engine=eng)
        result = r.run_domain(_CANONICAL)
        assert result.abort_reason == "sandbox_failure"
        assert not result.success

    def test_sandbox_timeout_error_field_populated(self):
        eng = _slow_engine(DataEngine(), sleep=5.0)
        r, _, _, _ = _make_router(engine=eng, timeout=0.1)
        result = r.run_domain(_CANONICAL)
        assert len(result.errors) >= 1

    def test_sandbox_timeout_no_snapshot(self):
        eng = _slow_engine(DataEngine(), sleep=5.0)
        r, _, _, rb = _make_router(engine=eng, timeout=0.1)
        r.run_domain(_CANONICAL)
        assert rb.snapshot_count == 0

    def test_sandbox_context_forwarded(self):
        """sandbox_context passed to run_domain() must reach engine.run()."""
        received = []
        engine = DataEngine()
        orig_run = engine.run

        def spy_run(inp, sandbox_context=None):
            received.append(sandbox_context)
            return orig_run(inp, sandbox_context=sandbox_context)

        engine.run = spy_run  # type: ignore[method-assign]
        r, _, _, _ = _make_router(engine=engine, use_sandbox=False)
        ctx = SandboxContext(SandboxConfig())
        r.run_domain(_CANONICAL, sandbox_context=ctx)
        assert len(received) >= 1
        assert received[-1] is ctx

    def test_sandbox_context_none_accepted(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL, sandbox_context=None)
        assert result.success

    def test_sandbox_logs_action(self):
        """Engine should log filesystem_read into the sandbox context."""
        r, _, _, _ = _make_router(use_sandbox=False)
        ctx = SandboxContext(SandboxConfig())
        r.run_domain(_CANONICAL, sandbox_context=ctx)
        actions = [a.action for a in ctx.actions]
        assert "filesystem_read" in actions


# ─────────────────────────────────────────────────────────────────────────────
# 7 — ROLLBACK / SNAPSHOT CREATION
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackSnapshot:

    def test_snapshot_created_on_success(self):
        r, _, _, rb = _make_router()
        r.run_domain(_CANONICAL)
        assert rb.snapshot_count >= 1

    def test_snapshot_tag_in_result(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert result.snapshot_tag == "post_run"

    def test_snapshot_tag_none_on_error(self):
        r, _, _, _ = _make_router(grant=False)
        result = r.run_domain(_CANONICAL)
        assert result.snapshot_tag is None

    def test_snapshot_tag_none_on_schema_error(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_SHORT)
        assert result.snapshot_tag is None

    def test_snapshot_tag_none_on_timeout(self):
        eng = _slow_engine(DataEngine(), sleep=5.0)
        r, _, _, _ = _make_router(engine=eng, timeout=0.1)
        result = r.run_domain(_CANONICAL)
        assert result.snapshot_tag is None

    def test_snapshot_retrievable(self):
        r, _, _, rb = _make_router()
        r.run_domain(_CANONICAL)
        snap = rb.get_snapshot("post_run")
        assert snap is not None
        assert snap.tag == "post_run"

    def test_snapshot_contains_engine_name(self):
        r, _, _, rb = _make_router()
        r.run_domain(_CANONICAL)
        snap = rb.get_snapshot("post_run")
        assert snap.engine_name == "data_backbone_engine"

    def test_snapshot_state_json_serializable(self):
        r, _, _, rb = _make_router()
        r.run_domain(_CANONICAL)
        snap = rb.get_snapshot("post_run")
        json.dumps(snap.state)

    def test_multiple_runs_accumulate_snapshots(self):
        r, _, _, rb = _make_router()
        r.run_domain(_CANONICAL)
        r.run_domain(_CANONICAL)
        assert rb.snapshot_count >= 2

    def test_no_snapshot_on_permission_denied(self):
        r, _, _, rb = _make_router(grant=False)
        r.run_domain(_CANONICAL)
        assert rb.snapshot_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8 — META DIAGNOSTICS INJECTION (DataDiagnostics)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaDiagnosticsInjection:
    """
    "Inject drift_score, confidence, volatility into MetaDiagnostics" is
    implemented as a DataDiagnostics record on the DomainExecutionResult.
    These tests verify all three core metrics and their companions.
    """

    def test_diagnostics_present_on_success(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert result.diagnostics is not None

    def test_drift_score_present(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert hasattr(d, "drift_score")

    def test_confidence_present(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert hasattr(d, "confidence")

    def test_volatility_present(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert hasattr(d, "volatility")

    def test_drift_score_finite(self):
        r, _, _, _ = _make_router()
        assert math.isfinite(r.run_domain(_CANONICAL).diagnostics.drift_score)

    def test_confidence_finite(self):
        r, _, _, _ = _make_router()
        assert math.isfinite(r.run_domain(_CANONICAL).diagnostics.confidence)

    def test_volatility_finite(self):
        r, _, _, _ = _make_router()
        assert math.isfinite(r.run_domain(_CANONICAL).diagnostics.volatility)

    def test_drift_score_in_unit_interval(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert 0.0 <= d.drift_score <= 1.0

    def test_confidence_in_unit_interval(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert 0.0 <= d.confidence <= 1.0

    def test_volatility_non_negative(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert d.volatility >= 0.0

    def test_anomaly_score_in_range(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert 0.0 <= d.anomaly_score <= 1.0

    def test_momentum_score_in_range(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert -1.0 <= d.momentum_score <= 1.0

    def test_stability_score_in_range(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert 0.0 < d.stability_score <= 1.0

    def test_cold_start_drift_zero(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        assert d.drift_score == 0.0

    def test_diagnostics_to_dict_json_serializable(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        json.dumps(d.to_dict())

    def test_diagnostics_to_dict_has_all_keys(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics.to_dict()
        for key in ("drift_score", "confidence", "volatility",
                    "anomaly_score", "momentum_score", "stability_score"):
            assert key in d

    def test_flat_series_volatility_zero(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_FLAT).diagnostics
        assert d.volatility == 0.0

    def test_drift_increases_after_distribution_shift(self):
        """Second run on vastly different data must show positive drift."""
        r, _, _, _ = _make_router()
        r.run_domain(_CANONICAL)                  # prime baseline
        big = {"values": [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4]}
        d2  = r.run_domain(big).diagnostics
        assert d2.drift_score > 0.0

    def test_error_result_diagnostics_all_zero(self):
        """Aborted runs return a safe zeroed-out DataDiagnostics."""
        r, _, _, _ = _make_router(grant=False)
        d = r.run_domain(_CANONICAL).diagnostics
        assert d.drift_score     == 0.0
        assert d.confidence      == 0.0
        assert d.volatility      == 0.0
        assert d.anomaly_score   == 0.0
        assert d.momentum_score  == 0.0
        assert d.stability_score == 1.0   # healthy default


# ─────────────────────────────────────────────────────────────────────────────
# 9 — DETERMINISM CHECK
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminismCheck:

    def test_returns_true_for_data_engine(self):
        assert domain_determinism_check(DataEngine) is True

    def test_returns_bool(self):
        assert isinstance(domain_determinism_check(DataEngine), bool)

    def test_default_argument_is_data_engine(self):
        assert domain_determinism_check() is True

    def test_two_fresh_routers_same_output(self):
        """Two completely independent router stacks on the same input must agree."""
        r1, _, _, _ = _make_router()
        r2, _, _, _ = _make_router()
        out1 = r1.run_domain(_CANONICAL).output
        out2 = r2.run_domain(_CANONICAL).output
        for key in ("status", "features", "signals", "engine", "version"):
            assert out1[key] == out2[key], f"Mismatch on key {key!r}"

    def test_100_runs_consistent_features(self):
        """100 independent router instances must produce identical feature dicts."""
        results = [_make_router()[0].run_domain(_CANONICAL).output["features"]
                   for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_broken_factory_returns_false(self):
        def bad_factory():
            raise RuntimeError("boom")
        assert domain_determinism_check(bad_factory) is False

    def test_non_deterministic_factory_returns_false(self):
        """A factory whose run() returns different results each time → False."""
        import random

        class NonDetEngine(DataEngine):
            def run(self, inp, sandbox_context=None):
                out = super().run(inp, sandbox_context=sandbox_context)
                if out["status"] == "ok":
                    out["features"]["mean"] = random.random()
                return out

        assert domain_determinism_check(NonDetEngine) is False

    def test_router_run_produces_same_features_twice(self):
        r1, _, _, _ = _make_router()
        f1 = r1.run_domain(_CANONICAL).output["features"]
        r2, _, _, _ = _make_router()
        f2 = r2.run_domain(_CANONICAL).output["features"]
        assert f1 == f2


# ─────────────────────────────────────────────────────────────────────────────
# 10 — SCHEMA ERROR SAFE HANDLING (no state mutation)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaErrorSafeHandling:

    def test_engine_run_count_unchanged_after_schema_error(self):
        r, eng, _, _ = _make_router()
        before = eng.run_count
        r.run_domain(_SHORT)
        assert eng.run_count == before

    def test_engine_baseline_unchanged_after_schema_error(self):
        r, eng, _, _ = _make_router()
        r.run_domain(_CANONICAL)                          # prime baseline
        before = eng._baseline_mean
        r.run_domain(_SHORT)                              # schema error
        assert eng._baseline_mean == before               # baseline untouched

    def test_rollback_count_unchanged_after_schema_error(self):
        r, _, _, rb = _make_router()
        r.run_domain(_SHORT)
        assert rb.snapshot_count == 0

    def test_no_exception_on_schema_error(self):
        r, _, _, _ = _make_router()
        r.run_domain(_SHORT)                              # must not raise

    def test_schema_error_then_valid_input_succeeds(self):
        r, _, _, _ = _make_router()
        r.run_domain(_SHORT)                              # error
        result = r.run_domain(_CANONICAL)                 # must recover
        assert result.success

    def test_permission_error_then_valid_succeeds(self):
        r, eng, pm, _ = _make_router(grant=False)
        r.run_domain(_CANONICAL)                          # denied
        for perm in eng.required_permissions:
            pm.grant(eng.name, perm)
        result = r.run_domain(_CANONICAL)
        assert result.success

    def test_many_schema_errors_no_accumulating_state(self):
        r, eng, _, rb = _make_router()
        for _ in range(10):
            r.run_domain(_SHORT)
        assert rb.snapshot_count == 0
        assert eng.run_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 11 — DomainExecutionResult CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainExecutionResultContract:

    def _all_results(self) -> list[DomainExecutionResult]:
        results = []
        # success
        r, _, _, _ = _make_router()
        results.append(r.run_domain(_CANONICAL))
        # not routable
        r, _, _, _ = _make_router()
        results.append(r.run_domain(_NO_VALUES))
        # schema error
        r, _, _, _ = _make_router()
        results.append(r.run_domain(_SHORT))
        # permission denied
        r, _, _, _ = _make_router(grant=False)
        results.append(r.run_domain(_CANONICAL))
        # timeout
        eng = _slow_engine(DataEngine(), sleep=5.0)
        r, _, _, _ = _make_router(engine=eng, timeout=0.1)
        results.append(r.run_domain(_CANONICAL))
        return results

    def test_all_results_json_serializable(self):
        for result in self._all_results():
            json.dumps(result.to_dict())

    def test_all_results_have_required_keys(self):
        required = {"success", "output", "diagnostics", "snapshot_tag",
                    "errors", "engine_name", "abort_reason"}
        for result in self._all_results():
            assert required <= result.to_dict().keys()

    def test_success_field_is_bool(self):
        for result in self._all_results():
            assert isinstance(result.success, bool)

    def test_errors_is_tuple(self):
        for result in self._all_results():
            assert isinstance(result.errors, tuple)

    def test_abort_reason_valid_values(self):
        valid = {"none", "not_routable", "schema_error",
                 "permission_denied", "sandbox_failure", "sandbox_timeout"}
        for result in self._all_results():
            assert result.abort_reason in valid

    def test_abort_reason_none_iff_success(self):
        for result in self._all_results():
            if result.success:
                assert result.abort_reason == "none"
            else:
                assert result.abort_reason != "none"

    def test_snapshot_tag_none_iff_not_success(self):
        for result in self._all_results():
            if not result.success:
                assert result.snapshot_tag is None

    def test_result_is_frozen(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        with pytest.raises((AttributeError, TypeError)):
            result.success = False  # type: ignore[misc]

    def test_is_json_serializable_helper(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert result.is_json_serializable()

    def test_repr_contains_engine_name(self):
        r, _, _, _ = _make_router()
        result = r.run_domain(_CANONICAL)
        assert "data_backbone_engine" in repr(result)


# ─────────────────────────────────────────────────────────────────────────────
# 12 — DataDiagnostics CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

class TestDataDiagnosticsContract:

    def test_is_frozen(self):
        d = DataDiagnostics(0.0, 1.0, 0.5, 0.1, 0.2, 0.9)
        with pytest.raises((AttributeError, TypeError)):
            d.drift_score = 0.5  # type: ignore[misc]

    def test_to_dict_has_all_keys(self):
        d = DataDiagnostics(0.1, 0.9, 0.3, 0.2, 0.1, 0.8)
        keys = d.to_dict().keys()
        for k in ("drift_score", "confidence", "volatility",
                  "anomaly_score", "momentum_score", "stability_score"):
            assert k in keys

    def test_to_dict_json_serializable(self):
        d = DataDiagnostics(0.1, 0.9, 0.3, 0.2, 0.1, 0.8)
        json.dumps(d.to_dict())

    def test_is_healthy_low_drift(self):
        d = DataDiagnostics(
            drift_score=0.1, confidence=0.9, volatility=0.2,
            anomaly_score=0.1, momentum_score=0.0, stability_score=0.9,
        )
        assert d.is_healthy()

    def test_is_healthy_high_drift(self):
        d = DataDiagnostics(
            drift_score=0.9, confidence=0.1, volatility=0.8,
            anomaly_score=0.9, momentum_score=0.0, stability_score=0.2,
        )
        assert not d.is_healthy()

    def test_all_floats_finite_from_router(self):
        r, _, _, _ = _make_router()
        d = r.run_domain(_CANONICAL).diagnostics
        for field_name in ("drift_score", "confidence", "volatility",
                           "anomaly_score", "momentum_score", "stability_score"):
            val = getattr(d, field_name)
            assert math.isfinite(val), f"{field_name}={val} is not finite"


# ─────────────────────────────────────────────────────────────────────────────
# 13 — build_default_registry
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDefaultRegistry:

    def test_returns_domain_registry(self):
        assert isinstance(build_default_registry(), DomainRegistry)

    def test_data_backbone_engine_registered(self):
        reg = build_default_registry()
        assert reg.has("data_backbone_engine")

    def test_engine_not_yet_loaded(self):
        reg = build_default_registry()
        assert not reg.is_loaded("data_backbone_engine")

    def test_get_returns_data_engine(self):
        reg = build_default_registry()
        eng = reg.get("data_backbone_engine")
        assert isinstance(eng, DataEngine)

    def test_engine_loaded_after_get(self):
        reg = build_default_registry()
        reg.get("data_backbone_engine")
        assert reg.is_loaded("data_backbone_engine")

    def test_two_registries_independent(self):
        reg1 = build_default_registry()
        reg2 = build_default_registry()
        e1 = reg1.get("data_backbone_engine")
        e2 = reg2.get("data_backbone_engine")
        assert e1 is not e2   # fully isolated

    def test_usable_in_router(self):
        reg = build_default_registry()
        eng = reg.get("data_backbone_engine")
        pm  = PermissionManager()
        pm.grant(eng.name, "filesystem_read")
        rb  = RollbackEngine(eng)
        router = DataEngineRouter(
            registry=reg,
            permission_manager=pm,
            rollback_engine=rb,
        )
        result = router.run_domain(_CANONICAL)
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# 14 — EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_integer_values_routed_correctly(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1, 2, 3, 4, 5, 6, 7]})
        assert result.success

    def test_minimum_length_values(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert result.success

    def test_extra_keys_in_input_do_not_break_routing(self):
        r, _, _, _ = _make_router()
        result = r.run_domain({"values": [1, 2, 3, 4, 5, 6, 7], "extra": "ignored"})
        assert result.success

    def test_router_accessors(self):
        r, _, pm, rb = _make_router()
        assert r.permission_manager is pm
        assert r.rollback_engine is rb
        assert isinstance(r.registry, DomainRegistry)

    def test_run_domain_always_returns_result(self):
        """run_domain() must never raise — always returns DomainExecutionResult."""
        inputs = [
            _CANONICAL, _SHORT, _NO_VALUES, {}, None,  # type: ignore[list-item]
            {"values": []}, {"values": [float("nan")] * 5},
        ]
        r, _, _, _ = _make_router()
        for inp in inputs:
            result = r.run_domain(inp)  # type: ignore[arg-type]
            assert isinstance(result, DomainExecutionResult)

    def test_diagnostics_always_returned(self):
        """Even on error, diagnostics must be a valid DataDiagnostics."""
        inputs = [_SHORT, _NO_VALUES, {}]
        for inp in inputs:
            r, _, _, _ = _make_router()
            d = r.run_domain(inp).diagnostics
            assert isinstance(d, DataDiagnostics)
            assert math.isfinite(d.drift_score)

    def test_successive_successful_runs_accumulate_snapshots(self):
        r, _, _, rb = _make_router()
        for _ in range(5):
            r.run_domain(_CANONICAL)
        assert rb.snapshot_count >= 5

    def test_confidence_decreases_after_drift(self):
        r, _, _, _ = _make_router()
        d1 = r.run_domain(_CANONICAL).diagnostics          # cold start
        big = {"values": [1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5]}
        d2  = r.run_domain(big).diagnostics                # large shift
        assert d2.drift_score > 0
        assert d2.confidence <= d1.confidence + 1e-9