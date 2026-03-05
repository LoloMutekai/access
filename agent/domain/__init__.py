"""
A.C.C.E.S.S. — Domain Infrastructure Layer (Phase 7.0)

This package provides the foundational infrastructure for all future
Domain Engines. It is deliberately decoupled from any specific domain.

Public surface:
    DomainEngine            Abstract base class for all domain engines
    DomainEngineMetadata    Frozen descriptor for engine identity
    SandboxConfig           Configuration for sandbox execution
    SandboxMode             Enum: NORMAL | READ_ONLY | DRY_RUN
    SandboxContext          Per-execution permission and logging context
    SandboxRunner           Executes a callable inside a controlled sandbox
    SandboxExecutionReport  Frozen result of a sandboxed execution
    SandboxViolation        Raised on permission or safety violations
    BenchmarkRunner         Runs and stores deterministic benchmark results
    BenchmarkResult         Frozen output of a benchmark run
    PermissionManager       Allowlist-based permission validation
    AuditLog                Append-only structured audit record
    RollbackEngine          Snapshot, restore, and failure-detection layer
    RollbackSnapshot        Frozen point-in-time state capture
    DummyDomainEngine       Reference implementation for testing
"""

from .base import DomainEngine, DomainEngineMetadata
from .sandbox import (
    SandboxConfig,
    SandboxMode,
    SandboxContext,
    SandboxRunner,
    SandboxExecutionReport,
    SandboxActionLog,
    SandboxViolation,
)
from .benchmark import BenchmarkRunner, BenchmarkResult
from .permissions import PermissionManager, AuditLog, AuditEntry, PermissionDeniedError
from .rollback import RollbackEngine, RollbackSnapshot, RollbackError
from .dummy import DummyDomainEngine

__all__ = [
    "DomainEngine",
    "DomainEngineMetadata",
    "SandboxConfig",
    "SandboxMode",
    "SandboxContext",
    "SandboxRunner",
    "SandboxExecutionReport",
    "SandboxActionLog",
    "SandboxViolation",
    "BenchmarkRunner",
    "BenchmarkResult",
    "PermissionManager",
    "AuditLog",
    "AuditEntry",
    "PermissionDeniedError",
    "RollbackEngine",
    "RollbackSnapshot",
    "RollbackError",
    "DummyDomainEngine",
]