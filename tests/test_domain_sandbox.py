"""
Tests for agent/domain/sandbox.py

Coverage:
    SandboxConfig           — construction, validation
    SandboxContext          — permission checks, logging, modes
    SandboxRunner           — normal, read_only, dry_run, timeout, errors
    SandboxExecutionReport  — structure and serialization
    SandboxViolation        — raised on denied actions
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.domain.sandbox import (
    SandboxConfig,
    SandboxContext,
    SandboxMode,
    SandboxRunner,
    SandboxExecutionReport,
    SandboxActionLog,
    SandboxViolation,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ok_fn(input_data: dict, sandbox_context=None) -> dict:
    return {"status": "ok", "received": input_data}


def _write_fn(input_data: dict, sandbox_context=None) -> dict:
    if sandbox_context:
        sandbox_context.check_permission(
            SandboxContext.ACTION_FILESYSTEM_WRITE, "test write"
        )
    return {"status": "ok"}


def _network_fn(input_data: dict, sandbox_context=None) -> dict:
    if sandbox_context:
        sandbox_context.check_permission(
            SandboxContext.ACTION_NETWORK_CALL, "test network"
        )
    return {"status": "ok"}


def _slow_fn(input_data: dict, sandbox_context=None) -> dict:
    time.sleep(10)
    return {"status": "ok"}


def _error_fn(input_data: dict, sandbox_context=None) -> dict:
    raise ValueError("Intentional error for testing.")


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxConfig:

    def test_default_construction(self):
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 5.0
        assert cfg.allow_filesystem_write is False
        assert cfg.allow_network is False
        assert cfg.mode == SandboxMode.NORMAL
        assert cfg.max_memory_mb == 256

    def test_custom_values(self):
        cfg = SandboxConfig(
            timeout_seconds=10.0,
            allow_filesystem_write=True,
            allow_network=True,
            mode=SandboxMode.READ_ONLY,
            max_memory_mb=512,
        )
        assert cfg.timeout_seconds == 10.0
        assert cfg.allow_filesystem_write is True

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError):
            SandboxConfig(timeout_seconds=0.0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError):
            SandboxConfig(timeout_seconds=-1.0)

    def test_zero_memory_raises(self):
        with pytest.raises(ValueError):
            SandboxConfig(max_memory_mb=0)

    def test_frozen_immutable(self):
        cfg = SandboxConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.timeout_seconds = 99.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxContext:

    def test_filesystem_write_denied_by_default(self):
        ctx = SandboxContext(SandboxConfig())
        with pytest.raises(SandboxViolation):
            ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)

    def test_filesystem_write_permitted_when_allowed(self):
        cfg = SandboxConfig(allow_filesystem_write=True)
        ctx = SandboxContext(cfg)
        result = ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)
        assert result is True

    def test_network_denied_by_default(self):
        ctx = SandboxContext(SandboxConfig())
        with pytest.raises(SandboxViolation):
            ctx.check_permission(SandboxContext.ACTION_NETWORK_CALL)

    def test_network_permitted_when_allowed(self):
        cfg = SandboxConfig(allow_network=True)
        ctx = SandboxContext(cfg)
        result = ctx.check_permission(SandboxContext.ACTION_NETWORK_CALL)
        assert result is True

    def test_action_logged_on_denial(self):
        ctx = SandboxContext(SandboxConfig())
        try:
            ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)
        except SandboxViolation:
            pass
        assert len(ctx.actions) == 1
        assert ctx.actions[0].permitted is False

    def test_action_logged_on_grant(self):
        cfg = SandboxConfig(allow_filesystem_write=True)
        ctx = SandboxContext(cfg)
        ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)
        assert len(ctx.actions) == 1
        assert ctx.actions[0].permitted is True

    def test_log_action_always_permitted(self):
        ctx = SandboxContext(SandboxConfig())
        ctx.log_action("filesystem_read", "test")
        assert len(ctx.actions) == 1
        assert ctx.actions[0].permitted is True

    def test_read_only_blocks_filesystem_write(self):
        cfg = SandboxConfig(
            allow_filesystem_write=True,
            mode=SandboxMode.READ_ONLY,
        )
        ctx = SandboxContext(cfg)
        with pytest.raises(SandboxViolation):
            ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)

    def test_read_only_blocks_state_mutation(self):
        cfg = SandboxConfig(mode=SandboxMode.READ_ONLY)
        ctx = SandboxContext(cfg)
        with pytest.raises(SandboxViolation):
            ctx.check_permission(SandboxContext.ACTION_STATE_MUTATION)

    def test_actions_returns_immutable_tuple(self):
        ctx = SandboxContext(SandboxConfig())
        ctx.log_action("test_action")
        actions = ctx.actions
        assert isinstance(actions, tuple)

    def test_multiple_actions_accumulated(self):
        cfg = SandboxConfig(allow_filesystem_write=True, allow_network=True)
        ctx = SandboxContext(cfg)
        ctx.log_action("a")
        ctx.check_permission(SandboxContext.ACTION_FILESYSTEM_WRITE)
        ctx.check_permission(SandboxContext.ACTION_NETWORK_CALL)
        assert len(ctx.actions) == 3

    def test_action_log_entries_are_frozen(self):
        ctx = SandboxContext(SandboxConfig())
        ctx.log_action("test")
        entry = ctx.actions[0]
        with pytest.raises((AttributeError, TypeError)):
            entry.action = "hacked"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX RUNNER — normal execution
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxRunnerNormal:

    def test_successful_run_returns_report(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"x": 1})
        assert isinstance(report, SandboxExecutionReport)

    def test_successful_run_success_true(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"x": 1})
        assert report.success is True

    def test_successful_run_no_error(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"x": 1})
        assert report.error is None

    def test_successful_run_result_contains_status(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"x": 1})
        assert report.result is not None
        assert report.result.get("status") == "ok"

    def test_successful_run_elapsed_positive(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"x": 1})
        assert report.elapsed_seconds >= 0.0

    def test_error_fn_returns_failure_report(self):
        runner = SandboxRunner()
        report = runner.run(_error_fn, {})
        assert report.success is False
        assert report.error is not None

    def test_error_fn_result_is_none(self):
        runner = SandboxRunner()
        report = runner.run(_error_fn, {})
        assert report.result is None

    def test_violation_fn_returns_failure_report(self):
        runner = SandboxRunner(SandboxConfig(allow_filesystem_write=False))
        report = runner.run(_write_fn, {})
        assert report.success is False

    def test_permitted_write_succeeds(self):
        runner = SandboxRunner(SandboxConfig(allow_filesystem_write=True))
        report = runner.run(_write_fn, {})
        assert report.success is True

    def test_to_dict_json_serializable(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {"key": "value"})
        json.dumps(report.to_dict())  # must not raise

    def test_to_dict_contains_required_keys(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {})
        d = report.to_dict()
        for key in ("success", "result", "error", "timed_out",
                    "actions_log", "elapsed_seconds", "mode"):
            assert key in d

    def test_mode_recorded_in_report(self):
        runner = SandboxRunner(SandboxConfig(mode=SandboxMode.NORMAL))
        report = runner.run(_ok_fn, {})
        assert report.mode == "normal"


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX RUNNER — DRY_RUN mode
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxRunnerDryRun:

    def test_dry_run_does_not_invoke_callable(self):
        called = []

        def _spy(input_data, sandbox_context=None):
            called.append(True)
            return {"status": "ok"}

        runner = SandboxRunner(SandboxConfig(mode=SandboxMode.DRY_RUN))
        runner.run(_spy, {})
        assert len(called) == 0

    def test_dry_run_success_true(self):
        runner = SandboxRunner(SandboxConfig(mode=SandboxMode.DRY_RUN))
        report = runner.run(_ok_fn, {})
        assert report.success is True

    def test_dry_run_result_is_none(self):
        runner = SandboxRunner(SandboxConfig(mode=SandboxMode.DRY_RUN))
        report = runner.run(_ok_fn, {})
        assert report.result is None

    def test_dry_run_flag_true(self):
        runner = SandboxRunner(SandboxConfig(mode=SandboxMode.DRY_RUN))
        report = runner.run(_ok_fn, {})
        assert report.dry_run is True

    def test_normal_run_dry_run_flag_false(self):
        runner = SandboxRunner()
        report = runner.run(_ok_fn, {})
        assert report.dry_run is False


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX RUNNER — timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxRunnerTimeout:

    def test_timeout_report_timed_out_true(self):
        runner = SandboxRunner(SandboxConfig(timeout_seconds=0.1))
        report = runner.run(_slow_fn, {})
        assert report.timed_out is True

    def test_timeout_report_success_false(self):
        runner = SandboxRunner(SandboxConfig(timeout_seconds=0.1))
        report = runner.run(_slow_fn, {})
        assert report.success is False

    def test_timeout_error_message_present(self):
        runner = SandboxRunner(SandboxConfig(timeout_seconds=0.1))
        report = runner.run(_slow_fn, {})
        assert report.error is not None
        assert "timeout" in report.error.lower() or "exceeded" in report.error.lower()

    def test_fast_fn_does_not_timeout(self):
        runner = SandboxRunner(SandboxConfig(timeout_seconds=5.0))
        report = runner.run(_ok_fn, {})
        assert report.timed_out is False


# ─────────────────────────────────────────────────────────────────────────────
# ACTION LOG SERIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxActionLog:

    def test_to_dict_contains_required_keys(self):
        entry = SandboxActionLog(
            action="test_action",
            permitted=True,
            timestamp="2026-01-01T00:00:00+00:00",
            detail="test detail",
        )
        d = entry.to_dict()
        assert "action" in d
        assert "permitted" in d
        assert "timestamp" in d
        assert "detail" in d

    def test_to_dict_json_serializable(self):
        entry = SandboxActionLog(
            action="fs_write",
            permitted=False,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        json.dumps(entry.to_dict())

    def test_repr_contains_action(self):
        entry = SandboxActionLog(
            action="my_action",
            permitted=True,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert "my_action" in repr(entry)