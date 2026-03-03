"""
A.C.C.E.S.S. — Structured Logger (Phase 3)

Provides pure JSON event logging without side effects.

Design:
    - All events are pure dicts (no custom types)
    - Events accumulated in-memory — not printed automatically
    - Retrieved via get_logs() or get_logs_by_type()
    - Optional external sink callback for real-time forwarding (file, socket, etc.)
    - Thread-safe for single-process use (list.append is GIL-protected)

Event structure:
    {
        "event_type":  "<category>",
        "timestamp":   "<ISO 8601 UTC>",
        "session_id":  "<session or null>",
        "turn_index":  <int or null>,
        "payload":     { ... }
    }

Event types used by AgentCore:
    "turn_completed"         — blocking turn finished
    "stream_finalized"       — streaming turn finalized
    "tool_used"              — tool dispatched
    "reflection_done"        — reflection computed
    "memory_write"           — memory written
    "memory_maintenance"     — maintenance run completed
    "error"                  — non-fatal error
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    In-memory structured event logger.

    Usage:
        structured_logger = StructuredLogger()

        structured_logger.log_event(
            event_type="turn_completed",
            payload=response.to_log_dict(),
            session_id="session_1",
            turn_index=3,
        )

        all_logs = structured_logger.get_logs()
        turn_logs = structured_logger.get_logs_by_type("turn_completed")

    With external sink:
        def sink(event: dict) -> None:
            with open("agent.jsonl", "a") as f:
                import json; f.write(json.dumps(event) + "\\n")

        structured_logger = StructuredLogger(sink=sink)
    """

    def __init__(
        self,
        sink: Optional[Callable[[dict], None]] = None,
        max_events: int = 10_000,
    ):
        """
        Args:
            sink:       Optional callable. Called with each event dict immediately
                        after it is recorded. Must not raise (errors are suppressed).
            max_events: Maximum events to keep in memory. Oldest are dropped when exceeded.
        """
        self._events: list[dict] = []
        self._sink = sink
        self._max_events = max_events

    def log_event(
        self,
        event_type: str,
        payload: dict,
        session_id: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> None:
        """
        Record a structured event.

        Args:
            event_type:  Category string (e.g., "turn_completed", "tool_used").
            payload:     Dict of event-specific data. Must be JSON-serializable.
            session_id:  Optional session identifier for filtering.
            turn_index:  Optional turn counter for chronological filtering.
        """
        event = {
            "event_type": event_type,
            "timestamp":  datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "turn_index": turn_index,
            "payload":    payload,
        }

        self._events.append(event)

        # Enforce memory limit (drop oldest) AFTER append
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Forward to external sink — never crash
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception as exc:
                logger.warning(f"StructuredLogger sink error (suppressed): {exc}")

    def get_logs(self) -> list[dict]:
        """Return all recorded events (newest last). Returns a copy."""
        return list(self._events)

    def get_logs_by_type(self, event_type: str) -> list[dict]:
        """Return only events matching the given event_type."""
        return [e for e in self._events if e["event_type"] == event_type]

    def get_logs_by_session(self, session_id: str) -> list[dict]:
        """Return only events matching the given session_id."""
        return [e for e in self._events if e["session_id"] == session_id]

    def get_recent(self, n: int) -> list[dict]:
        """Return the N most recent events."""
        return list(self._events[-n:])

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events = []

    @property
    def event_count(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"StructuredLogger(events={self.event_count}, max={self._max_events})"