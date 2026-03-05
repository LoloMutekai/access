"""
A.C.C.E.S.S. — Rollback Engine (Phase 7.0)

Provides snapshot-based state capture and restore for DomainEngine instances.

Design:
    - RollbackSnapshot is a frozen, deep-copied point-in-time state record
    - RollbackEngine maintains an ordered version stack
    - Restore replaces the engine's mutable state with a snapshot
    - Failure detection hook triggers conservative mode
    - Conservative mode reduces the engine's operational aggressiveness
      (exact semantics delegated to the engine via a callback hook)

Constraints:
    - No file I/O (snapshots are in-memory only)
    - State copying uses json round-trip for serialization safety
    - No global state; each RollbackEngine is fully isolated
    - All outputs are JSON-serializable
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class RollbackError(RuntimeError):
    """Raised when a rollback operation cannot be completed."""


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RollbackSnapshot:
    """
    Immutable point-in-time capture of an engine's serializable state.

    tag             : Human-readable label for this snapshot
    version_id      : Monotonically increasing integer within a RollbackEngine
    engine_name     : Name of the engine this snapshot belongs to
    engine_version  : Version string of the engine at snapshot time
    state           : Deep copy of the engine's serializable state dict
    captured_at     : ISO-8601 UTC timestamp
    metadata        : Optional arbitrary metadata dict (must be JSON-serializable)
    """
    tag: str
    version_id: int
    engine_name: str
    engine_version: str
    state: dict
    captured_at: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tag":            self.tag,
            "version_id":     self.version_id,
            "engine_name":    self.engine_name,
            "engine_version": self.engine_version,
            "state":          self.state,
            "captured_at":    self.captured_at,
            "metadata":       self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"RollbackSnapshot("
            f"tag={self.tag!r}, "
            f"v={self.version_id}, "
            f"engine={self.engine_name!r}, "
            f"captured_at={self.captured_at!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RollbackEngine:
    """
    Snapshot, restore, and failure-detection layer for DomainEngine instances.

    State is captured by calling engine.get_state() (must return a
    JSON-serializable dict). State is restored by calling
    engine.set_state(state_dict).

    If the engine does not implement get_state / set_state, the RollbackEngine
    falls back to a best-effort dict copy of engine.__dict__.

    Usage:
        rollback = RollbackEngine(engine)
        rollback.snapshot("before_update")

        # ... apply changes to engine ...

        if something_went_wrong:
            rollback.restore("before_update")

    Conservative mode:
        After a restore, conservative_mode is set to True.
        If a failure_hook is registered, it is called with the snapshot.
        The engine's operational aggressiveness is expected to be reduced
        by the hook or by external code checking rollback.conservative_mode.
    """

    def __init__(
        self,
        engine: Any,
        failure_hook: Optional[Callable[[RollbackSnapshot], None]] = None,
        max_snapshots: int = 50,
    ) -> None:
        """
        Args:
            engine          : DomainEngine instance to manage.
            failure_hook    : Optional callable invoked after every restore.
                              Receives the snapshot that was restored to.
            max_snapshots   : Maximum number of snapshots to retain.
                              Oldest are evicted when the limit is exceeded.
        """
        if max_snapshots <= 0:
            raise ValueError(
                f"max_snapshots must be positive, got {max_snapshots}"
            )
        self._engine          = engine
        self._failure_hook    = failure_hook
        self._max_snapshots   = max_snapshots
        self._snapshots: list[RollbackSnapshot] = []
        self._version_counter = 0
        self._conservative_mode = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def conservative_mode(self) -> bool:
        """
        True after any restore operation.
        External code should check this flag before applying
        aggressive engine operations.
        """
        return self._conservative_mode

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots currently stored."""
        return len(self._snapshots)

    @property
    def engine(self) -> Any:
        """Reference to the managed engine."""
        return self._engine

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self, tag: str, metadata: Optional[dict] = None) -> RollbackSnapshot:
        """
        Capture the current engine state as an immutable snapshot.

        Args:
            tag      : Human-readable label. Must be non-empty.
            metadata : Optional JSON-serializable metadata dict.

        Returns:
            RollbackSnapshot — frozen, stored in the internal stack.

        Raises:
            ValueError if tag is empty.
            RollbackError if state cannot be captured or serialized.
        """
        if not tag or not tag.strip():
            raise ValueError("RollbackEngine.snapshot() requires a non-empty tag.")

        state = self._capture_state()
        self._version_counter += 1

        snap = RollbackSnapshot(
            tag=tag,
            version_id=self._version_counter,
            engine_name=getattr(self._engine, "name",    "unknown"),
            engine_version=getattr(self._engine, "version", "unknown"),
            state=state,
            captured_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )

        self._snapshots.append(snap)
        self._evict_if_needed()

        return snap

    # ── Restore ───────────────────────────────────────────────────────────────

    def restore(self, tag: str) -> RollbackSnapshot:
        """
        Restore the engine to the most recent snapshot with the given tag.

        Sets conservative_mode = True and invokes failure_hook if registered.

        Args:
            tag : Tag of the snapshot to restore to.

        Returns:
            The RollbackSnapshot that was restored.

        Raises:
            RollbackError if no snapshot with that tag exists.
        """
        snap = self._find_latest(tag)
        if snap is None:
            available = [s.tag for s in self._snapshots]
            raise RollbackError(
                f"No snapshot found with tag {tag!r}. "
                f"Available tags: {available}"
            )

        self._apply_state(snap.state)
        self._conservative_mode = True

        if self._failure_hook is not None:
            try:
                self._failure_hook(snap)
            except Exception:
                # Failure hook errors must not block the restore
                pass

        return snap

    def restore_to_version(self, version_id: int) -> RollbackSnapshot:
        """
        Restore the engine to the snapshot with the given version_id.

        Args:
            version_id : The exact version_id of the snapshot to restore.

        Returns:
            The RollbackSnapshot that was restored.

        Raises:
            RollbackError if no snapshot with that version_id exists.
        """
        snap = next(
            (s for s in self._snapshots if s.version_id == version_id), None
        )
        if snap is None:
            raise RollbackError(
                f"No snapshot found with version_id={version_id}."
            )

        self._apply_state(snap.state)
        self._conservative_mode = True

        if self._failure_hook is not None:
            try:
                self._failure_hook(snap)
            except Exception:
                pass

        return snap

    # ── Query ─────────────────────────────────────────────────────────────────

    def list_snapshots(self) -> list[dict]:
        """Return a list of snapshot summaries (tag, version_id, captured_at)."""
        return [
            {
                "tag":        s.tag,
                "version_id": s.version_id,
                "captured_at": s.captured_at,
                "engine_name": s.engine_name,
            }
            for s in self._snapshots
        ]

    def get_snapshot(self, tag: str) -> Optional[RollbackSnapshot]:
        """Return the most recent snapshot with the given tag, or None."""
        return self._find_latest(tag)

    def delete_snapshot(self, tag: str) -> bool:
        """
        Delete all snapshots with the given tag.
        Returns True if any were deleted.
        """
        before = len(self._snapshots)
        self._snapshots = [s for s in self._snapshots if s.tag != tag]
        return len(self._snapshots) < before

    def clear_snapshots(self) -> None:
        """Remove all stored snapshots. conservative_mode is unaffected."""
        self._snapshots.clear()

    def reset_conservative_mode(self) -> None:
        """
        Clear the conservative_mode flag.
        Should only be called after a human reviewer confirms the engine
        is stable post-restore.
        """
        self._conservative_mode = False

    def to_dict(self) -> dict:
        return {
            "engine_name":        getattr(self._engine, "name",    "unknown"),
            "engine_version":     getattr(self._engine, "version", "unknown"),
            "snapshot_count":     self.snapshot_count,
            "conservative_mode":  self._conservative_mode,
            "version_counter":    self._version_counter,
            "snapshots":          self.list_snapshots(),
        }

    # ── Failure detection hook ────────────────────────────────────────────────

    def register_failure_hook(
        self,
        hook: Callable[[RollbackSnapshot], None],
    ) -> None:
        """
        Register or replace the failure hook.
        Hook is called with the snapshot after every restore operation.
        """
        self._failure_hook = hook

    def detect_failure_and_rollback(
        self,
        check_fn: Callable[[], bool],
        tag: str,
    ) -> bool:
        """
        Run check_fn(); if it returns False (failure), restore to tag.

        Args:
            check_fn : Zero-argument callable returning True = healthy.
            tag      : Snapshot tag to roll back to on failure.

        Returns:
            True if check passed (no rollback).
            False if check failed (rollback performed).

        Raises:
            RollbackError if failure detected but no snapshot with tag exists.
        """
        try:
            healthy = check_fn()
        except Exception:
            healthy = False

        if not healthy:
            self.restore(tag)
            return False

        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _capture_state(self) -> dict:
        """
        Capture engine state as a deep-copied, JSON-safe dict.

        Priority:
            1. engine.get_state() if it exists and returns a dict.
            2. Shallow copy of engine.__dict__ filtered to JSON-serializable keys.
        """
        if hasattr(self._engine, "get_state"):
            try:
                raw = self._engine.get_state()
                if isinstance(raw, dict):
                    return self._json_round_trip(raw)
            except Exception as exc:
                raise RollbackError(
                    f"engine.get_state() failed: {exc}"
                ) from exc

        # Fallback: best-effort __dict__ copy
        try:
            raw = {
                k: v for k, v in vars(self._engine).items()
                if not k.startswith("_")
            }
            return self._json_round_trip(raw)
        except Exception as exc:
            raise RollbackError(
                f"Failed to capture engine state via __dict__: {exc}"
            ) from exc

    def _apply_state(self, state: dict) -> None:
        """
        Restore engine state from a captured dict.

        Priority:
            1. engine.set_state(state) if it exists.
            2. Update engine.__dict__ directly for non-private keys.
        """
        restored = copy.deepcopy(state)

        if hasattr(self._engine, "set_state"):
            try:
                self._engine.set_state(restored)
                return
            except Exception as exc:
                raise RollbackError(
                    f"engine.set_state() failed: {exc}"
                ) from exc

        # Fallback: direct __dict__ update
        try:
            for k, v in restored.items():
                if not k.startswith("_"):
                    setattr(self._engine, k, v)
        except Exception as exc:
            raise RollbackError(
                f"Failed to restore engine state via setattr: {exc}"
            ) from exc

    @staticmethod
    def _json_round_trip(obj: dict) -> dict:
        """
        Serialize and deserialize via JSON to ensure the state copy
        contains only JSON-safe primitives. Raises RollbackError on failure.
        """
        try:
            return json.loads(json.dumps(obj))
        except (TypeError, ValueError) as exc:
            raise RollbackError(
                f"Engine state is not JSON-serializable: {exc}"
            ) from exc

    def _find_latest(self, tag: str) -> Optional[RollbackSnapshot]:
        """Return the most recent snapshot matching the given tag."""
        for snap in reversed(self._snapshots):
            if snap.tag == tag:
                return snap
        return None

    def _evict_if_needed(self) -> None:
        """Evict the oldest snapshot if the stack exceeds max_snapshots."""
        while len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def __repr__(self) -> str:
        return (
            f"RollbackEngine("
            f"engine={getattr(self._engine, 'name', '?')!r}, "
            f"snapshots={self.snapshot_count}, "
            f"conservative={self._conservative_mode})"
        )