"""
A.C.C.E.S.S. — Permission System (Phase 7.0)

Provides allowlist-based action-level permission validation with a
structured, append-only audit log.

Design:
    - PermissionManager holds an explicit allowlist of (subject, action) pairs
    - Each check produces an AuditEntry regardless of outcome
    - The AuditLog is append-only; entries are frozen and immutable
    - No global mutable state; each PermissionManager is fully isolated
    - All structures are JSON-serializable

Terminology:
    subject : The entity requesting permission (e.g. engine name)
    action  : The permission key being checked (e.g. "filesystem_write")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class PermissionDeniedError(PermissionError):
    """Raised when a subject attempts an action they are not permitted."""


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEntry:
    """
    Immutable record of a single permission check.

    subject     : Entity that requested the action
    action      : Permission key that was checked
    granted     : True iff permission was granted
    timestamp   : ISO-8601 UTC timestamp
    context     : Optional free-form context string
    """
    subject: str
    action: str
    granted: bool
    timestamp: str
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "subject":   self.subject,
            "action":    self.action,
            "granted":   self.granted,
            "timestamp": self.timestamp,
            "context":   self.context,
        }

    def __repr__(self) -> str:
        verdict = "GRANTED" if self.granted else "DENIED"
        return (
            f"AuditEntry("
            f"subject={self.subject!r}, "
            f"action={self.action!r}, "
            f"{verdict})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog:
    """
    Append-only log of AuditEntry records.

    Once an entry is appended it is immutable.
    The log itself can be queried but not modified except by appending.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(self, entry: AuditEntry) -> None:
        """Append an immutable AuditEntry."""
        if not isinstance(entry, AuditEntry):
            raise TypeError(
                f"AuditLog only accepts AuditEntry instances, got {type(entry).__name__}"
            )
        self._entries.append(entry)

    @property
    def entries(self) -> tuple:
        """Snapshot of all entries as an immutable tuple."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def filter_by_subject(self, subject: str) -> list[AuditEntry]:
        """Return all entries for a given subject (oldest first)."""
        return [e for e in self._entries if e.subject == subject]

    def filter_by_action(self, action: str) -> list[AuditEntry]:
        """Return all entries for a given action (oldest first)."""
        return [e for e in self._entries if e.action == action]

    def filter_denied(self) -> list[AuditEntry]:
        """Return all entries where permission was denied."""
        return [e for e in self._entries if not e.granted]

    def to_dict(self) -> dict:
        return {
            "entry_count": len(self._entries),
            "entries":     [e.to_dict() for e in self._entries],
        }

    def __repr__(self) -> str:
        denied = sum(1 for e in self._entries if not e.granted)
        return f"AuditLog(total={len(self._entries)}, denied={denied})"


# ─────────────────────────────────────────────────────────────────────────────
# PERMISSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class PermissionManager:
    """
    Allowlist-based permission manager.

    Permissions are stored as (subject, action) pairs.
    A subject may be a DomainEngine name or any other string identifier.
    An action may be any permission key string.

    Wildcard support:
        subject="*" grants the action to ALL subjects.
        action="*"  grants ALL actions to the given subject.

    Usage:
        pm = PermissionManager()
        pm.grant("my_engine", "filesystem_read")
        pm.grant("my_engine", "network_call")

        pm.check("my_engine", "filesystem_read")   # returns True
        pm.check("my_engine", "filesystem_write")  # raises PermissionDeniedError

    Audit log:
        pm.audit_log.entries   → tuple of all AuditEntry records
        pm.audit_log.filter_denied()  → denied entries only
    """

    WILDCARD = "*"

    def __init__(self) -> None:
        # Set of (subject, action) tuples that are explicitly permitted
        self._allowlist: set[tuple[str, str]] = set()
        self._audit_log = AuditLog()

    # ── Grant / Revoke ────────────────────────────────────────────────────────

    def grant(self, subject: str, action: str) -> None:
        """
        Add (subject, action) to the allowlist.
        Idempotent — granting an already-granted pair has no effect.
        """
        self._allowlist.add((subject, action))

    def revoke(self, subject: str, action: str) -> None:
        """
        Remove (subject, action) from the allowlist.
        Silent if the pair was not present.
        """
        self._allowlist.discard((subject, action))

    def revoke_all(self, subject: str) -> None:
        """Remove all permissions for a given subject."""
        to_remove = {pair for pair in self._allowlist if pair[0] == subject}
        self._allowlist -= to_remove

    def grant_all(self, subject: str, actions: list[str]) -> None:
        """Grant multiple actions to a subject in one call."""
        for action in actions:
            self.grant(subject, action)

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_permitted(self, subject: str, action: str) -> bool:
        """
        Return True iff (subject, action) is in the allowlist.
        Does NOT write to the audit log.
        """
        return self._is_allowed(subject, action)

    def check(
        self,
        subject: str,
        action: str,
        context: str = "",
        raise_on_denial: bool = True,
    ) -> bool:
        """
        Check permission and record in the audit log.

        Args:
            subject          : Identity of the requesting entity.
            action           : Permission key to check.
            context          : Optional context string for the audit record.
            raise_on_denial  : If True (default), raises PermissionDeniedError
                               on denial. If False, returns False instead.

        Returns:
            True if permitted.

        Raises:
            PermissionDeniedError if denied and raise_on_denial is True.
        """
        granted = self._is_allowed(subject, action)
        self._audit_log.append(AuditEntry(
            subject=subject,
            action=action,
            granted=granted,
            timestamp=datetime.now(UTC).isoformat(),
            context=context,
        ))
        if not granted and raise_on_denial:
            raise PermissionDeniedError(
                f"Subject {subject!r} is not permitted to perform "
                f"action {action!r}."
            )
        return granted

    def permitted_actions(self, subject: str) -> frozenset:
        """
        Return all explicitly granted actions for a subject.
        Does not include wildcard-derived grants.
        """
        return frozenset(
            action for (subj, action) in self._allowlist
            if subj == subject
        )

    # ── Audit ─────────────────────────────────────────────────────────────────

    @property
    def audit_log(self) -> AuditLog:
        """Read-only reference to the audit log."""
        return self._audit_log

    def audit_summary(self) -> dict:
        """Return a JSON-serializable summary of the audit log."""
        log = self._audit_log
        total   = len(log)
        denied  = len(log.filter_denied())
        granted = total - denied
        return {
            "total_checks":  total,
            "granted_count": granted,
            "denied_count":  denied,
            "denial_rate":   round(denied / total, 4) if total > 0 else 0.0,
        }

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot_allowlist(self) -> frozenset:
        """Return an immutable snapshot of the current allowlist."""
        return frozenset(self._allowlist)

    def to_dict(self) -> dict:
        return {
            "allowlist": [
                {"subject": s, "action": a}
                for (s, a) in sorted(self._allowlist)
            ],
            "audit_summary": self.audit_summary(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_allowed(self, subject: str, action: str) -> bool:
        """
        Core allowlist lookup with wildcard support.
        Subject wildcard OR action wildcard OR exact match → permitted.
        """
        if (self.WILDCARD, action) in self._allowlist:
            return True
        if (subject, self.WILDCARD) in self._allowlist:
            return True
        return (subject, action) in self._allowlist

    def __repr__(self) -> str:
        return (
            f"PermissionManager("
            f"allowlist_size={len(self._allowlist)}, "
            f"audit_entries={len(self._audit_log)})"
        )