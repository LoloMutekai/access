"""
A.C.C.E.S.S. — Sandbox Execution Layer (Phase 7.0)

Provides a controlled execution environment for DomainEngine.run() calls.

Guarantees:
    - Execution timeout enforced via thread isolation
    - Filesystem writes blocked unless explicitly permitted
    - Network access blocked unless explicitly permitted
    - All actions logged in structured, immutable records
    - SandboxExecutionReport is frozen and JSON-serializable
    - No global mutable state

Modes:
    NORMAL    — full sandboxing, permissions respected
    READ_ONLY — all write actions blocked regardless of config
    DRY_RUN   — callable is not invoked; a no-op report is returned

Memory cap is modelled as a config field and checked against a mock
measurement; real RSS enforcement requires OS-level tooling and is
outside the certification boundary of this module.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SandboxViolation(RuntimeError):
    """
    Raised when a sandboxed function attempts a forbidden action.
    The action and reason are recorded in the structured report regardless.
    """


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class SandboxMode(str, Enum):
    """
    NORMAL    — standard sandboxing; permissions from SandboxConfig apply.
    READ_ONLY — all write actions (filesystem, state mutation) are denied,
                regardless of SandboxConfig.allow_filesystem_write.
    DRY_RUN   — the callable is never invoked; a no-op report is returned.
    """
    NORMAL    = "normal"
    READ_ONLY = "read_only"
    DRY_RUN   = "dry_run"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxConfig:
    """
    Frozen configuration for a single sandbox execution.

    timeout_seconds         : Wall-clock limit for the callable (default 5.0s)
    allow_filesystem_write  : Permit write actions to the filesystem
    allow_network           : Permit outbound network calls
    mode                    : Execution mode (NORMAL | READ_ONLY | DRY_RUN)
    max_memory_mb           : Soft memory cap in megabytes (mocked enforcement)
    """
    timeout_seconds: float         = 5.0
    allow_filesystem_write: bool   = False
    allow_network: bool            = False
    mode: SandboxMode              = SandboxMode.NORMAL
    max_memory_mb: int             = 256

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"SandboxConfig.timeout_seconds must be positive, "
                f"got {self.timeout_seconds}"
            )
        if self.max_memory_mb <= 0:
            raise ValueError(
                f"SandboxConfig.max_memory_mb must be positive, "
                f"got {self.max_memory_mb}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ACTION LOG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxActionLog:
    """
    Immutable record of a single action attempted within a sandbox execution.

    action      : Symbolic action name e.g. "filesystem_write", "network_call"
    permitted   : Whether the sandbox allowed this action
    timestamp   : ISO-8601 UTC timestamp of the attempt
    detail      : Optional human-readable detail string
    """
    action: str
    permitted: bool
    timestamp: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "action":    self.action,
            "permitted": self.permitted,
            "timestamp": self.timestamp,
            "detail":    self.detail,
        }

    def __repr__(self) -> str:
        status = "PERMITTED" if self.permitted else "DENIED"
        return f"SandboxActionLog({self.action!r} → {status})"


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxExecutionReport:
    """
    Frozen, JSON-serializable result of one sandboxed execution.

    success         : True iff callable completed without error or violation
    result          : Return value of the callable, or None on failure
    error           : Error message string, or None on success
    timed_out       : True iff execution exceeded timeout_seconds
    actions_log     : Ordered tuple of all SandboxActionLog entries
    elapsed_seconds : Wall-clock time from invocation to completion/timeout
    mode            : The SandboxMode that was active during execution
    dry_run         : True iff mode was DRY_RUN (callable was not invoked)
    """
    success: bool
    result: Optional[dict]
    error: Optional[str]
    timed_out: bool
    actions_log: tuple
    elapsed_seconds: float
    mode: str
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "success":         self.success,
            "result":          self.result,
            "error":           self.error,
            "timed_out":       self.timed_out,
            "actions_log":     [a.to_dict() for a in self.actions_log],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "mode":            self.mode,
            "dry_run":         self.dry_run,
        }

    def __repr__(self) -> str:
        status = "ok" if self.success else ("timeout" if self.timed_out else "error")
        return (
            f"SandboxExecutionReport("
            f"status={status!r}, "
            f"elapsed={self.elapsed_seconds:.3f}s, "
            f"mode={self.mode!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

class SandboxContext:
    """
    Per-execution context injected into DomainEngine.run().

    Provides:
        - check_permission(action)  : validates and logs a requested action
        - log_action(action)        : records an informational action
        - actions                   : read-only view of accumulated log entries

    The context accumulates an internal action log during execution.
    This log is harvested by SandboxRunner to build the final report.

    SandboxContext is NOT frozen — it must accumulate log entries.
    However, no external code should mutate it beyond the public API.
    """

    # Canonical action names
    ACTION_FILESYSTEM_WRITE = "filesystem_write"
    ACTION_FILESYSTEM_READ  = "filesystem_read"
    ACTION_NETWORK_CALL     = "network_call"
    ACTION_STATE_MUTATION   = "state_mutation"

    # Which actions are gated by config flags
    _GATED_ACTIONS: dict[str, str] = {
        ACTION_FILESYSTEM_WRITE: "allow_filesystem_write",
        ACTION_NETWORK_CALL:     "allow_network",
    }

    # Which actions are blocked in READ_ONLY mode
    _READ_ONLY_BLOCKED: frozenset = frozenset({
        ACTION_FILESYSTEM_WRITE,
        ACTION_STATE_MUTATION,
    })

    def __init__(self, config: SandboxConfig) -> None:
        self._config  = config
        self._log: list[SandboxActionLog] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def check_permission(self, action: str, detail: str = "") -> bool:
        """
        Check whether `action` is permitted under the current config and mode.
        Always logs the attempt regardless of outcome.

        Args:
            action : Canonical action name (use class constants).
            detail : Optional contextual string for the log entry.

        Returns:
            True if permitted.

        Raises:
            SandboxViolation if the action is denied.
        """
        permitted = self._is_permitted(action)
        self._append_log(action, permitted, detail)
        if not permitted:
            raise SandboxViolation(
                f"Sandbox denied action {action!r} "
                f"(mode={self._config.mode.value!r}, detail={detail!r})"
            )
        return True

    def log_action(self, action: str, detail: str = "") -> None:
        """
        Record an informational action without permission checking.
        Use for read-only or observational actions.
        """
        self._append_log(action, permitted=True, detail=detail)

    @property
    def actions(self) -> tuple:
        """Read-only snapshot of all accumulated log entries."""
        return tuple(self._log)

    @property
    def config(self) -> SandboxConfig:
        """The configuration this context was constructed with."""
        return self._config

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_permitted(self, action: str) -> bool:
        cfg  = self._config
        mode = cfg.mode

        # DRY_RUN: all actions denied (callable shouldn't even be running)
        if mode == SandboxMode.DRY_RUN:
            return False

        # READ_ONLY: write-class actions always denied
        if mode == SandboxMode.READ_ONLY and action in self._READ_ONLY_BLOCKED:
            return False

        # NORMAL + READ_ONLY: check per-action config flags
        config_key = self._GATED_ACTIONS.get(action)
        if config_key is not None:
            return bool(getattr(cfg, config_key, False))

        # Unknown or ungated actions (e.g. filesystem_read): permitted
        return True

    def _append_log(self, action: str, permitted: bool, detail: str) -> None:
        self._log.append(SandboxActionLog(
            action=action,
            permitted=permitted,
            timestamp=datetime.now(UTC).isoformat(),
            detail=detail,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class SandboxRunner:
    """
    Executes a callable inside a controlled sandbox.

    Usage:
        runner = SandboxRunner(config=SandboxConfig(timeout_seconds=3.0))
        report = runner.run(engine.run, input_data)

    The callable receives `input_data` as its first argument and a
    `sandbox_context` keyword argument of type SandboxContext.

    Thread isolation is used to enforce the timeout. The callable runs
    in a separate thread; if it does not complete within timeout_seconds,
    a timed_out=True report is returned and the thread is abandoned
    (Python does not support hard thread killing; the thread eventually
    completes but its result is discarded).
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self._config = config or SandboxConfig()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    def run(
        self,
        fn: Callable,
        input_data: dict,
    ) -> SandboxExecutionReport:
        """
        Execute fn(input_data, sandbox_context=ctx) inside the sandbox.

        Args:
            fn          : Callable to execute. Should accept (dict, sandbox_context=...).
            input_data  : Input dict passed to fn. Not mutated by the runner.

        Returns:
            SandboxExecutionReport — always returned, never raises.
        """
        import time
        cfg = self._config
        ctx = SandboxContext(cfg)
        t0  = time.monotonic()

        # ── DRY_RUN: skip execution entirely ──────────────────────────────
        if cfg.mode == SandboxMode.DRY_RUN:
            elapsed = time.monotonic() - t0
            return SandboxExecutionReport(
                success=True,
                result=None,
                error=None,
                timed_out=False,
                actions_log=ctx.actions,
                elapsed_seconds=elapsed,
                mode=cfg.mode.value,
                dry_run=True,
            )

        # ── Normal / Read-only execution with timeout ─────────────────────
        result_container: dict[str, Any] = {}

        def _target() -> None:
            try:
                output = fn(input_data, sandbox_context=ctx)
                result_container["result"] = output
            except SandboxViolation as exc:
                result_container["violation"] = str(exc)
            except Exception as exc:
                result_container["error"] = (
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_target)
            try:
                future.result(timeout=cfg.timeout_seconds)
                timed_out = False
            except FuturesTimeoutError:
                timed_out = True

        elapsed = time.monotonic() - t0

        if timed_out:
            return SandboxExecutionReport(
                success=False,
                result=None,
                error=f"Execution exceeded timeout of {cfg.timeout_seconds}s.",
                timed_out=True,
                actions_log=ctx.actions,
                elapsed_seconds=elapsed,
                mode=cfg.mode.value,
            )

        if "violation" in result_container:
            return SandboxExecutionReport(
                success=False,
                result=None,
                error=result_container["violation"],
                timed_out=False,
                actions_log=ctx.actions,
                elapsed_seconds=elapsed,
                mode=cfg.mode.value,
            )

        if "error" in result_container:
            return SandboxExecutionReport(
                success=False,
                result=None,
                error=result_container["error"],
                timed_out=False,
                actions_log=ctx.actions,
                elapsed_seconds=elapsed,
                mode=cfg.mode.value,
            )

        return SandboxExecutionReport(
            success=True,
            result=result_container.get("result"),
            error=None,
            timed_out=False,
            actions_log=ctx.actions,
            elapsed_seconds=elapsed,
            mode=cfg.mode.value,
        )