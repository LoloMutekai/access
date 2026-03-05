"""
A.C.C.E.S.S. — DomainEngine Abstract Base Class (Phase 7.0)

Defines the contract every Domain Engine must fulfil.
No domain-specific logic. No external dependencies beyond stdlib.

Design rules:
    - run()                 Pure function of inputs unless sandbox_context supplied.
    - benchmark()           Must return reproducible, JSON-serializable metrics.
    - self_test()           Must complete in bounded time, return pass/fail per check.
    - deterministic_check() Must validate: identical input -> identical output.

Mathematical guarantees expected from all implementors:
    P1: run(x) == run(x)  for all valid x  (determinism)
    P2: No mutation of input_data dict
    P3: All float fields in return dicts are finite
    P4: All returned dicts are JSON-serializable
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainEngineMetadata:
    """
    Immutable identity descriptor for a DomainEngine.

    name                  : Unique engine identifier (snake_case)
    version               : Semantic version string e.g. "1.0.0"
    required_permissions  : Frozenset of permission keys the engine needs
    description           : Human-readable summary of engine purpose
    """
    name: str
    version: str
    required_permissions: frozenset
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("DomainEngineMetadata.name must be non-empty.")
        if not self.version or not self.version.strip():
            raise ValueError("DomainEngineMetadata.version must be non-empty.")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "required_permissions": sorted(self.required_permissions),
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"DomainEngineMetadata("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"permissions={sorted(self.required_permissions)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DomainEngine(ABC):
    """
    Abstract base class for all A.C.C.E.S.S. Domain Engines.

    Subclasses must implement:
        name                    : str property
        version                 : str property
        required_permissions    : frozenset property
        run()                   : core execution method
        benchmark()             : reproducible metrics method
        self_test()             : internal health check method
        deterministic_check()   : idempotency validator

    All methods are expected to be bounded, pure (unless sandbox_context
    is provided), and produce JSON-serializable return values.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique engine identifier. snake_case. Immutable after construction."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string. e.g. '1.0.0'."""

    @property
    @abstractmethod
    def required_permissions(self) -> frozenset:
        """
        Set of permission keys this engine requires to operate.
        Must be declared upfront; cannot be changed at runtime.
        """

    # ── Core contract ─────────────────────────────────────────────────────────

    @abstractmethod
    def run(
        self,
        input_data: dict,
        sandbox_context: Optional[Any] = None,
    ) -> dict:
        """
        Execute the engine's primary logic.

        Args:
            input_data      : Caller-supplied input. Must not be mutated.
            sandbox_context : Optional SandboxContext. If None, engine is
                              treated as running in pure / offline mode.

        Returns:
            dict — JSON-serializable result. Must contain at minimum:
                {
                    "status":  str,       # "ok" | "error" | "skipped"
                    "output":  Any,       # engine-specific payload
                    "engine":  str,       # self.name
                    "version": str,       # self.version
                }

        Constraints:
            - Must not mutate input_data.
            - Must not perform I/O unless sandbox_context explicitly permits.
            - Must terminate in bounded time.
            - Must be deterministic for identical inputs (same seed state).
        """

    @abstractmethod
    def benchmark(self) -> dict:
        """
        Run a self-contained, reproducible benchmark.

        Returns:
            dict — JSON-serializable. Must contain at minimum:
                {
                    "engine":   str,
                    "version":  str,
                    "score":    float,      # ∈ [0.0, 1.0]
                    "metrics":  dict,       # named float values
                    "passed":   bool,
                }

        Constraints:
            - Must use deterministic, pre-defined internal datasets.
            - Must not depend on external resources.
            - Repeated calls must produce identical scores (same inputs).
        """

    @abstractmethod
    def self_test(self) -> dict:
        """
        Run a suite of internal health checks.

        Returns:
            dict — JSON-serializable. Must contain at minimum:
                {
                    "engine":   str,
                    "version":  str,
                    "checks":   list[dict],  # [{"name": str, "passed": bool, "detail": str}]
                    "passed":   bool,         # True iff all checks passed
                }

        Constraints:
            - Each check must be named and individually reportable.
            - Must not depend on external resources.
            - Must complete in bounded time.
        """

    @abstractmethod
    def deterministic_check(self) -> bool:
        """
        Validate that run() produces identical output for identical input.

        Implementation must:
            1. Construct a fixed, canonical input_data dict.
            2. Call run() twice with that input.
            3. Compare outputs excluding any timestamp fields.
            4. Return True iff outputs are structurally identical.

        Returns:
            bool — True if determinism invariant holds.
        """

    # ── Derived helpers (non-abstract, may be overridden) ────────────────────

    def metadata(self) -> DomainEngineMetadata:
        """Return a frozen identity descriptor for this engine."""
        return DomainEngineMetadata(
            name=self.name,
            version=self.version,
            required_permissions=self.required_permissions,
        )

    def validate_input(self, input_data: dict) -> list[str]:
        """
        Optional hook for input validation.
        Returns list of error messages; empty list means valid.
        Subclasses may override. Default: accept any dict.
        """
        if not isinstance(input_data, dict):
            return [f"input_data must be dict, got {type(input_data).__name__}"]
        return []

    def _assert_json_serializable(self, obj: Any, context: str = "output") -> None:
        """
        Raise ValueError if obj is not JSON-serializable.
        Used by implementations to validate their own return values.
        """
        try:
            json.dumps(obj)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DomainEngine.{self.name}: {context} is not JSON-serializable: {exc}"
            ) from exc

    def _check_floats_finite(self, obj: Any) -> bool:
        """Recursively verify no float in obj is NaN or Inf."""
        if isinstance(obj, float):
            return math.isfinite(obj)
        if isinstance(obj, dict):
            return all(self._check_floats_finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(self._check_floats_finite(v) for v in obj)
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"