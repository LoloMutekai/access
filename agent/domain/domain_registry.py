"""
A.C.C.E.S.S. — Domain Registry (Phase 7.1)
agent/domain/domain_registry.py

Provides a lazy, isolated registry for DomainEngine instances.
Each engine is constructed on first access and cached per-registry instance.

Design rules
────────────
    - No global mutable state. Every DomainRegistry is fully isolated.
    - Engines are constructed lazily — factory callable is invoked only once,
      on the first call to get(). Subsequent calls return the cached instance.
    - Factories are registered by string key (typically engine.name).
    - The registry never imports engine modules at module load time, preventing
      circular imports and keeping startup cost near zero.
    - All public methods are thread-safe via a simple per-instance lock
      (standard threading.Lock — no external dependencies required).

Usage
─────
    registry = DomainRegistry()
    registry.register("data_backbone_engine", DataEngine)

    engine = registry.get("data_backbone_engine")   # constructed here
    engine = registry.get("data_backbone_engine")   # cached — same object

    # Check without constructing
    registry.has("data_backbone_engine")   # True
    registry.is_loaded("data_backbone_engine")  # True (already built)

    # Evict cached instance (next get() re-constructs)
    registry.evict("data_backbone_engine")
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class DomainRegistry:
    """
    Lazy, isolated registry for DomainEngine instances.

    Accepts either a class (called with no arguments) or any zero-argument
    callable as the factory for each engine key.

    Thread safety
    ─────────────
    get() uses a per-instance lock to prevent double-construction when the
    first call is made concurrently.  All other methods (register, evict,
    list_registered, list_loaded) are also guarded by the same lock.
    """

    def __init__(self) -> None:
        self._factories:  dict[str, Callable] = {}
        self._instances:  dict[str, object]   = {}
        self._lock = threading.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, key: str, factory: Callable) -> None:
        """
        Register an engine factory under ``key``.

        Args:
            key     : Unique string identifier (e.g. ``engine.name``).
            factory : Zero-argument callable that returns a DomainEngine.
                      A class qualifies as a zero-argument callable.

        Raises:
            ValueError  : if key is empty or factory is not callable.
            KeyError    : if key is already registered (use ``evict`` first).
        """
        if not key or not key.strip():
            raise ValueError("DomainRegistry.register(): key must be non-empty.")
        if not callable(factory):
            raise ValueError(
                f"DomainRegistry.register(): factory for {key!r} must be callable."
            )
        with self._lock:
            if key in self._factories:
                raise KeyError(
                    f"DomainRegistry.register(): key {key!r} already registered. "
                    f"Call evict({key!r}) first to replace it."
                )
            self._factories[key] = factory

    def register_or_replace(self, key: str, factory: Callable) -> None:
        """
        Register a factory, replacing any existing registration.

        Evicts any cached instance for ``key`` before installing the new
        factory, so the next get() call builds a fresh engine.
        """
        if not key or not key.strip():
            raise ValueError("DomainRegistry.register_or_replace(): key must be non-empty.")
        if not callable(factory):
            raise ValueError(
                f"DomainRegistry.register_or_replace(): factory for {key!r} must be callable."
            )
        with self._lock:
            self._factories[key]  = factory
            self._instances.pop(key, None)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> object:
        """
        Return the engine for ``key``, constructing it on first access.

        Args:
            key : Key used during register().

        Returns:
            The DomainEngine instance (cached after first construction).

        Raises:
            KeyError    : if ``key`` was never registered.
            RuntimeError: if the factory raises during construction.
        """
        with self._lock:
            if key not in self._factories:
                raise KeyError(
                    f"DomainRegistry.get(): no engine registered for key {key!r}. "
                    f"Available: {sorted(self._factories)}"
                )
            if key not in self._instances:
                try:
                    self._instances[key] = self._factories[key]()
                except Exception as exc:
                    raise RuntimeError(
                        f"DomainRegistry: factory for {key!r} raised during construction."
                    ) from exc
            return self._instances[key]

    def get_or_none(self, key: str) -> Optional[object]:
        """Return the cached/constructed engine, or None if key is unknown."""
        try:
            return self.get(key)
        except (KeyError, RuntimeError):
            return None

    # ── Inspection ────────────────────────────────────────────────────────────

    def has(self, key: str) -> bool:
        """True iff a factory is registered for ``key`` (loaded or not)."""
        with self._lock:
            return key in self._factories

    def is_loaded(self, key: str) -> bool:
        """True iff the engine for ``key`` has been constructed and cached."""
        with self._lock:
            return key in self._instances

    def list_registered(self) -> list[str]:
        """Return sorted list of all registered keys."""
        with self._lock:
            return sorted(self._factories)

    def list_loaded(self) -> list[str]:
        """Return sorted list of keys whose engines are already constructed."""
        with self._lock:
            return sorted(self._instances)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def evict(self, key: str) -> bool:
        """
        Discard the cached engine instance for ``key``.

        The factory registration is preserved — the next get() call will
        re-construct the engine from scratch.

        Returns True if an instance was evicted, False if none was cached.
        """
        with self._lock:
            if key in self._instances:
                del self._instances[key]
                return True
            return False

    def unregister(self, key: str) -> bool:
        """
        Remove both the factory and any cached instance for ``key``.

        Returns True if the key existed, False otherwise.
        """
        with self._lock:
            existed = key in self._factories
            self._factories.pop(key, None)
            self._instances.pop(key, None)
            return existed

    def clear(self) -> None:
        """Remove all registrations and cached instances."""
        with self._lock:
            self._factories.clear()
            self._instances.clear()

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        with self._lock:
            n_reg    = len(self._factories)
            n_loaded = len(self._instances)
        return f"DomainRegistry(registered={n_reg}, loaded={n_loaded})"