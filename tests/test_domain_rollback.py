"""
Tests for agent/domain/rollback.py

Coverage:
    RollbackSnapshot — structure, immutability, serialization
    RollbackEngine   — snapshot, restore, failure hook, conservative mode,
                       version management, eviction
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.domain.rollback import RollbackEngine, RollbackSnapshot, RollbackError
from agent.domain.dummy import DummyDomainEngine


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_stateful_engine() -> DummyDomainEngine:
    engine = DummyDomainEngine()
    engine.run({"setup": True})  # advances run_count to 1
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK SNAPSHOT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackSnapshot:

    def _make(self) -> RollbackSnapshot:
        return RollbackSnapshot(
            tag="v1",
            version_id=1,
            engine_name="test_engine",
            engine_version="1.0.0",
            state={"counter": 42},
            captured_at="2026-01-01T00:00:00+00:00",
            metadata={"note": "test"},
        )

    def test_construction(self):
        snap = self._make()
        assert snap.tag == "v1"
        assert snap.version_id == 1
        assert snap.state == {"counter": 42}

    def test_frozen_immutable(self):
        snap = self._make()
        with pytest.raises((AttributeError, TypeError)):
            snap.tag = "hacked"  # type: ignore[misc]

    def test_to_dict_contains_required_keys(self):
        d = self._make().to_dict()
        for key in ("tag", "version_id", "engine_name", "state", "captured_at"):
            assert key in d

    def test_to_dict_json_serializable(self):
        json.dumps(self._make().to_dict())

    def test_repr_contains_tag(self):
        assert "v1" in repr(self._make())


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK ENGINE TESTS — snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackEngineSnapshot:

    def test_snapshot_returns_rollback_snapshot(self):
        engine   = _make_stateful_engine()
        rollback = RollbackEngine(engine)
        snap = rollback.snapshot("v1")
        assert isinstance(snap, RollbackSnapshot)

    def test_snapshot_tag_recorded(self):
        engine   = _make_stateful_engine()
        rollback = RollbackEngine(engine)
        snap = rollback.snapshot("my_tag")
        assert snap.tag == "my_tag"

    def test_snapshot_version_id_increments(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        s1 = rollback.snapshot("a")
        s2 = rollback.snapshot("b")
        assert s2.version_id == s1.version_id + 1

    def test_snapshot_captures_engine_name(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        snap = rollback.snapshot("v")
        assert snap.engine_name == engine.name

    def test_snapshot_state_is_copy(self):
        engine = DummyDomainEngine()
        engine.run_count = 5
        rollback = RollbackEngine(engine)
        snap = rollback.snapshot("v1")
        engine.run_count = 999
        # snapshot must be unaffected
        assert snap.state["run_count"] == 5

    def test_empty_tag_raises(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        with pytest.raises(ValueError):
            rollback.snapshot("")

    def test_whitespace_tag_raises(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        with pytest.raises(ValueError):
            rollback.snapshot("   ")

    def test_snapshot_count_increments(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("a")
        rollback.snapshot("b")
        assert rollback.snapshot_count == 2

    def test_list_snapshots_contains_tag(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("my_tag")
        tags = [s["tag"] for s in rollback.list_snapshots()]
        assert "my_tag" in tags

    def test_get_snapshot_returns_correct(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("target")
        snap = rollback.get_snapshot("target")
        assert snap is not None
        assert snap.tag == "target"

    def test_get_snapshot_returns_none_for_missing(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        assert rollback.get_snapshot("nonexistent") is None

    def test_max_snapshots_evicts_oldest(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine, max_snapshots=3)
        for i in range(5):
            rollback.snapshot(f"v{i}")
        assert rollback.snapshot_count == 3
        # earliest tags v0, v1 should be gone
        assert rollback.get_snapshot("v0") is None
        assert rollback.get_snapshot("v1") is None


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK ENGINE TESTS — restore
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackEngineRestore:

    def test_restore_reverts_state(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)

        engine.run_count = 5
        rollback.snapshot("before")

        engine.run_count = 100
        rollback.restore("before")

        assert engine.run_count == 5

    def test_restore_sets_conservative_mode(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("v1")
        assert rollback.conservative_mode is False
        rollback.restore("v1")
        assert rollback.conservative_mode is True

    def test_restore_unknown_tag_raises(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        with pytest.raises(RollbackError):
            rollback.restore("nonexistent")

    def test_restore_invokes_failure_hook(self):
        hook_calls = []

        def hook(snap: RollbackSnapshot) -> None:
            hook_calls.append(snap.tag)

        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine, failure_hook=hook)
        rollback.snapshot("v1")
        rollback.restore("v1")
        assert hook_calls == ["v1"]

    def test_restore_hook_exception_does_not_block(self):
        def bad_hook(snap):
            raise RuntimeError("Hook failed.")

        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine, failure_hook=bad_hook)
        rollback.snapshot("v1")
        rollback.restore("v1")  # must not raise
        assert rollback.conservative_mode is True

    def test_restore_to_version_id(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)

        engine.run_count = 10
        snap = rollback.snapshot("v1")
        engine.run_count = 999

        rollback.restore_to_version(snap.version_id)
        assert engine.run_count == 10

    def test_restore_to_nonexistent_version_raises(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        with pytest.raises(RollbackError):
            rollback.restore_to_version(9999)

    def test_reset_conservative_mode(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("v1")
        rollback.restore("v1")
        assert rollback.conservative_mode is True
        rollback.reset_conservative_mode()
        assert rollback.conservative_mode is False


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK ENGINE TESTS — failure detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackEngineFailureDetection:

    def test_healthy_check_returns_true(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("v1")
        result = rollback.detect_failure_and_rollback(lambda: True, "v1")
        assert result is True

    def test_failing_check_triggers_rollback(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)

        engine.run_count = 5
        rollback.snapshot("v1")
        engine.run_count = 999

        result = rollback.detect_failure_and_rollback(lambda: False, "v1")
        assert result is False
        assert engine.run_count == 5

    def test_failing_check_sets_conservative_mode(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("v1")
        rollback.detect_failure_and_rollback(lambda: False, "v1")
        assert rollback.conservative_mode is True

    def test_check_raises_treated_as_failure(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        engine.run_count = 3
        rollback.snapshot("v1")
        engine.run_count = 99

        def bad_check():
            raise RuntimeError("check exploded")

        result = rollback.detect_failure_and_rollback(bad_check, "v1")
        assert result is False
        assert engine.run_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# ROLLBACK ENGINE TESTS — management
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackEngineManagement:

    def test_delete_snapshot_removes_by_tag(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("target")
        rollback.snapshot("other")
        deleted = rollback.delete_snapshot("target")
        assert deleted is True
        assert rollback.get_snapshot("target") is None
        assert rollback.get_snapshot("other") is not None

    def test_delete_nonexistent_returns_false(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        result = rollback.delete_snapshot("nonexistent")
        assert result is False

    def test_clear_snapshots_empties_stack(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("a")
        rollback.snapshot("b")
        rollback.clear_snapshots()
        assert rollback.snapshot_count == 0

    def test_register_failure_hook_replaces_old(self):
        calls = []
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.register_failure_hook(lambda s: calls.append("first"))
        rollback.register_failure_hook(lambda s: calls.append("second"))
        rollback.snapshot("v1")
        rollback.restore("v1")
        assert calls == ["second"]

    def test_to_dict_json_serializable(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        rollback.snapshot("v1")
        json.dumps(rollback.to_dict())

    def test_to_dict_contains_required_keys(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        d = rollback.to_dict()
        for key in ("engine_name", "snapshot_count", "conservative_mode", "snapshots"):
            assert key in d

    def test_max_snapshots_zero_raises(self):
        engine = DummyDomainEngine()
        with pytest.raises(ValueError):
            RollbackEngine(engine, max_snapshots=0)

    def test_repr_contains_engine_name(self):
        engine   = DummyDomainEngine()
        rollback = RollbackEngine(engine)
        assert "dummy_domain_engine" in repr(rollback)