"""
Tests for agent/domain/permissions.py

Coverage:
    AuditEntry       — structure, serialization, immutability
    AuditLog         — append, query, filter
    PermissionManager — grant, revoke, check, wildcard, audit
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.domain.permissions import (
    AuditEntry,
    AuditLog,
    PermissionDeniedError,
    PermissionManager,
)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT ENTRY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditEntry:

    def _make(self, granted: bool = True) -> AuditEntry:
        return AuditEntry(
            subject="engine_a",
            action="filesystem_read",
            granted=granted,
            timestamp="2026-01-01T00:00:00+00:00",
            context="test",
        )

    def test_construction(self):
        e = self._make()
        assert e.subject == "engine_a"
        assert e.action == "filesystem_read"
        assert e.granted is True

    def test_frozen_immutable(self):
        e = self._make()
        with pytest.raises((AttributeError, TypeError)):
            e.granted = False  # type: ignore[misc]

    def test_to_dict_contains_keys(self):
        d = self._make().to_dict()
        for key in ("subject", "action", "granted", "timestamp", "context"):
            assert key in d

    def test_to_dict_json_serializable(self):
        json.dumps(self._make().to_dict())

    def test_repr_contains_action(self):
        assert "filesystem_read" in repr(self._make())

    def test_repr_shows_granted(self):
        assert "GRANTED" in repr(self._make(granted=True))

    def test_repr_shows_denied(self):
        assert "DENIED" in repr(self._make(granted=False))


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLog:

    def _entry(self, subject: str = "s", action: str = "a",
               granted: bool = True) -> AuditEntry:
        return AuditEntry(
            subject=subject, action=action,
            granted=granted, timestamp="2026-01-01T00:00:00+00:00",
        )

    def test_empty_log_has_zero_entries(self):
        log = AuditLog()
        assert len(log) == 0

    def test_append_increments_length(self):
        log = AuditLog()
        log.append(self._entry())
        assert len(log) == 1

    def test_entries_returns_tuple(self):
        log = AuditLog()
        log.append(self._entry())
        assert isinstance(log.entries, tuple)

    def test_append_non_entry_raises(self):
        log = AuditLog()
        with pytest.raises(TypeError):
            log.append("not an entry")  # type: ignore[arg-type]

    def test_filter_by_subject(self):
        log = AuditLog()
        log.append(self._entry(subject="alpha"))
        log.append(self._entry(subject="beta"))
        log.append(self._entry(subject="alpha"))
        result = log.filter_by_subject("alpha")
        assert len(result) == 2
        assert all(e.subject == "alpha" for e in result)

    def test_filter_by_action(self):
        log = AuditLog()
        log.append(self._entry(action="read"))
        log.append(self._entry(action="write"))
        result = log.filter_by_action("read")
        assert len(result) == 1
        assert result[0].action == "read"

    def test_filter_denied(self):
        log = AuditLog()
        log.append(self._entry(granted=True))
        log.append(self._entry(granted=False))
        log.append(self._entry(granted=False))
        denied = log.filter_denied()
        assert len(denied) == 2
        assert all(not e.granted for e in denied)

    def test_to_dict_json_serializable(self):
        log = AuditLog()
        log.append(self._entry())
        json.dumps(log.to_dict())

    def test_to_dict_entry_count_correct(self):
        log = AuditLog()
        for _ in range(4):
            log.append(self._entry())
        assert log.to_dict()["entry_count"] == 4

    def test_repr_contains_counts(self):
        log = AuditLog()
        log.append(self._entry(granted=True))
        log.append(self._entry(granted=False))
        r = repr(log)
        assert "total=2" in r
        assert "denied=1" in r


# ─────────────────────────────────────────────────────────────────────────────
# PERMISSION MANAGER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionManager:

    def test_fresh_manager_denies_everything(self):
        pm = PermissionManager()
        assert pm.is_permitted("engine_a", "filesystem_read") is False

    def test_grant_allows_check(self):
        pm = PermissionManager()
        pm.grant("engine_a", "filesystem_read")
        assert pm.is_permitted("engine_a", "filesystem_read") is True

    def test_grant_idempotent(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.grant("engine_a", "read")
        assert pm.is_permitted("engine_a", "read") is True

    def test_revoke_removes_permission(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.revoke("engine_a", "read")
        assert pm.is_permitted("engine_a", "read") is False

    def test_revoke_nonexistent_is_silent(self):
        pm = PermissionManager()
        pm.revoke("engine_a", "nonexistent")  # must not raise

    def test_revoke_all_removes_subject_permissions(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.grant("engine_a", "write")
        pm.grant("engine_b", "read")
        pm.revoke_all("engine_a")
        assert pm.is_permitted("engine_a", "read") is False
        assert pm.is_permitted("engine_a", "write") is False
        assert pm.is_permitted("engine_b", "read") is True

    def test_grant_all_grants_multiple_actions(self):
        pm = PermissionManager()
        pm.grant_all("engine_a", ["read", "write", "network"])
        assert pm.is_permitted("engine_a", "read") is True
        assert pm.is_permitted("engine_a", "write") is True
        assert pm.is_permitted("engine_a", "network") is True

    def test_check_raises_on_denial(self):
        pm = PermissionManager()
        with pytest.raises(PermissionDeniedError):
            pm.check("engine_a", "write")

    def test_check_returns_true_on_grant(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        result = pm.check("engine_a", "read")
        assert result is True

    def test_check_no_raise_returns_false(self):
        pm = PermissionManager()
        result = pm.check("engine_a", "write", raise_on_denial=False)
        assert result is False

    def test_check_writes_to_audit_log(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.check("engine_a", "read")
        assert len(pm.audit_log) == 1

    def test_denied_check_logged(self):
        pm = PermissionManager()
        try:
            pm.check("engine_a", "write")
        except PermissionDeniedError:
            pass
        denied = pm.audit_log.filter_denied()
        assert len(denied) == 1

    def test_wildcard_subject_grants_all_subjects(self):
        pm = PermissionManager()
        pm.grant(PermissionManager.WILDCARD, "public_read")
        assert pm.is_permitted("engine_a", "public_read") is True
        assert pm.is_permitted("engine_b", "public_read") is True
        assert pm.is_permitted("anyone",   "public_read") is True

    def test_wildcard_action_grants_all_actions(self):
        pm = PermissionManager()
        pm.grant("superuser", PermissionManager.WILDCARD)
        assert pm.is_permitted("superuser", "read")          is True
        assert pm.is_permitted("superuser", "write")         is True
        assert pm.is_permitted("superuser", "anything_else") is True

    def test_permitted_actions_returns_frozenset(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.grant("engine_a", "write")
        actions = pm.permitted_actions("engine_a")
        assert isinstance(actions, frozenset)
        assert "read" in actions
        assert "write" in actions

    def test_permitted_actions_empty_for_unknown_subject(self):
        pm = PermissionManager()
        assert pm.permitted_actions("unknown_subject") == frozenset()

    def test_audit_summary_structure(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        pm.check("engine_a", "read")
        try:
            pm.check("engine_a", "write")
        except PermissionDeniedError:
            pass
        summary = pm.audit_summary()
        assert summary["total_checks"] == 2
        assert summary["granted_count"] == 1
        assert summary["denied_count"] == 1

    def test_snapshot_allowlist_is_frozenset(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        snap = pm.snapshot_allowlist()
        assert isinstance(snap, frozenset)
        assert ("engine_a", "read") in snap

    def test_to_dict_json_serializable(self):
        pm = PermissionManager()
        pm.grant("engine_a", "read")
        json.dumps(pm.to_dict())

    def test_repr_contains_allowlist_size(self):
        pm = PermissionManager()
        pm.grant("e", "a")
        assert "allowlist_size=1" in repr(pm)