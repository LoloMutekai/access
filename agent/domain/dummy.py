"""
A.C.C.E.S.S. — DummyDomainEngine (Phase 7.0)

Reference implementation of DomainEngine for infrastructure validation.

Behaviour:
    - run()               : Echoes the input, appends a deterministic checksum
    - benchmark()         : Returns a fixed perfect score against a tiny dataset
    - self_test()         : Runs 5 named invariant checks
    - deterministic_check(): Verifies run() is idempotent
    - get_state()         : Returns serializable internal counters
    - set_state()         : Restores counters from a state dict

This engine does not implement any domain-specific logic.
It exists solely to validate the infrastructure layer.

Mathematical properties:
    P1: run(x) == run(x)  for all x  (determinism)
    P2: input_data is never mutated
    P3: all floats in output are finite
    P4: all outputs are JSON-serializable
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from .base import DomainEngine


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DummyDomainEngine(DomainEngine):
    """
    Minimal, deterministic reference implementation of DomainEngine.

    Intended use:
        - Infrastructure validation
        - Test harness baseline
        - Template for real domain engine authors

    Internal state:
        run_count    : Total number of run() calls (mutable for snapshot demo)
        last_input   : Shallow copy of the most recent input_data

    State is exposed via get_state() / set_state() for rollback testing.
    """

    _NAME    = "dummy_domain_engine"
    _VERSION = "1.0.0"
    _REQUIRED_PERMISSIONS: frozenset = frozenset({
        "filesystem_read",   # demonstrates a typical read-only permission
    })

    # Internal benchmark datasets (frozen, deterministic)
    _BENCHMARK_DATASETS: tuple = (
        {"input": {"value": 0},   "expected_status": "ok"},
        {"input": {"value": 1},   "expected_status": "ok"},
        {"input": {"value": 42},  "expected_status": "ok"},
        {"input": {"value": -1},  "expected_status": "ok"},
        {"input": {"text": "hi"}, "expected_status": "ok"},
    )

    def __init__(self) -> None:
        self.run_count: int  = 0
        self.last_input: dict = {}

    # ── DomainEngine abstract properties ─────────────────────────────────────

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def required_permissions(self) -> frozenset:
        return self._REQUIRED_PERMISSIONS

    # ── Core contract ─────────────────────────────────────────────────────────

    def run(
        self,
        input_data: dict,
        sandbox_context: Optional[Any] = None,
    ) -> dict:
        """
        Echo input and append a deterministic SHA-256 checksum.

        Side effects:
            - Increments self.run_count (mutable counter, used by rollback demo)
            - Stores shallow copy of input in self.last_input

        No mutation of input_data itself.
        """
        errors = self.validate_input(input_data)
        if errors:
            return {
                "status":  "error",
                "output":  None,
                "engine":  self.name,
                "version": self.version,
                "errors":  errors,
            }

        # Deterministic checksum of the serialised input
        checksum = _stable_checksum(input_data)

        # Update internal counters (mutable, captured by get_state)
        self.run_count  += 1
        self.last_input  = dict(input_data)   # shallow copy; no mutation of original

        result = {
            "status":       "ok",
            "output":       dict(input_data),  # echo, not a reference
            "checksum":     checksum,
            "run_count":    self.run_count,
            "engine":       self.name,
            "version":      self.version,
        }

        # If a sandbox_context is provided, log the read action
        if sandbox_context is not None:
            try:
                sandbox_context.log_action(
                    "filesystem_read",
                    detail="DummyDomainEngine.run — no actual I/O",
                )
            except Exception:
                pass

        return result

    def benchmark(self) -> dict:
        """
        Run against the internal fixed benchmark dataset.
        Returns a deterministic perfect score (all inputs are valid).
        """
        passed = 0
        total  = len(self._BENCHMARK_DATASETS)
        per_dataset: list[dict] = []

        for ds in self._BENCHMARK_DATASETS:
            output = self.run(ds["input"])
            ok = output.get("status") == ds["expected_status"]
            per_dataset.append({
                "input":   ds["input"],
                "passed":  ok,
                "status":  output.get("status"),
            })
            if ok:
                passed += 1

        score = passed / total if total > 0 else 0.0

        return {
            "engine":   self.name,
            "version":  self.version,
            "score":    round(score, 6),
            "metrics":  {"per_dataset": per_dataset},
            "passed":   passed == total,
        }

    def self_test(self) -> dict:
        """
        Run 5 named internal checks.

        Checks:
            1. run() returns a dict
            2. run() contains required keys
            3. run() is deterministic
            4. input_data is not mutated by run()
            5. benchmark() returns a finite score
        """
        checks: list[dict] = []

        # Check 1: run() returns a dict
        _run_ok = self._check(
            name="run_returns_dict",
            fn=lambda: isinstance(self.run({"x": 1}), dict),
        )
        checks.append(_run_ok)

        # Check 2: run() contains required keys
        def _has_required_keys() -> bool:
            out = self.run({"x": 1})
            return all(k in out for k in ("status", "output", "engine", "version"))

        checks.append(self._check("run_has_required_keys", _has_required_keys))

        # Check 3: deterministic output
        def _is_deterministic() -> bool:
            inp = {"value": 99, "label": "test"}
            o1 = self.run(inp)
            o2 = self.run(inp)
            return o1["checksum"] == o2["checksum"] and o1["output"] == o2["output"]

        checks.append(self._check("run_is_deterministic", _is_deterministic))

        # Check 4: input not mutated
        def _no_mutation() -> bool:
            original = {"key": "value", "num": 42}
            snapshot = dict(original)
            self.run(original)
            return original == snapshot

        checks.append(self._check("run_no_input_mutation", _no_mutation))

        # Check 5: benchmark score is finite and in [0,1]
        def _benchmark_valid() -> bool:
            import math
            result = self.benchmark()
            score = result.get("score", -1)
            return isinstance(score, float) and math.isfinite(score) and 0.0 <= score <= 1.0

        checks.append(self._check("benchmark_score_valid", _benchmark_valid))

        all_passed = all(c["passed"] for c in checks)

        return {
            "engine":  self.name,
            "version": self.version,
            "checks":  checks,
            "passed":  all_passed,
        }

    def deterministic_check(self) -> bool:
        """
        Validate run() idempotency on a canonical fixed input.

        Compares outputs excluding run_count (which is intentionally mutable).
        """
        canonical = {"determinism_probe": True, "value": 7}

        o1 = self.run(canonical)
        o2 = self.run(canonical)

        # run_count differs between calls by design; exclude it
        o1_cmp = {k: v for k, v in o1.items() if k != "run_count"}
        o2_cmp = {k: v for k, v in o2.items() if k != "run_count"}

        return o1_cmp == o2_cmp

    # ── State for rollback support ────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of mutable internal state."""
        return {
            "run_count":  self.run_count,
            "last_input": dict(self.last_input),
        }

    def set_state(self, state: dict) -> None:
        """Restore mutable internal state from a previously captured snapshot."""
        self.run_count  = int(state.get("run_count",  0))
        self.last_input = dict(state.get("last_input", {}))

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _check(name: str, fn) -> dict:
        """Run a single named check; catch exceptions as failures."""
        try:
            passed = bool(fn())
            detail = "pass" if passed else "assertion returned False"
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        return {"name": name, "passed": passed, "detail": detail}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stable_checksum(obj: dict) -> str:
    """
    Produce a deterministic SHA-256 hex digest of a JSON-serializable dict.
    Keys are sorted to ensure key-order independence.
    """
    serialised = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()