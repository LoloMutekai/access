"""
A.C.C.E.S.S. — Persistent Identity Store (Phase 4)

JSON-based persistence for all Phase 4 cognitive identity states.

Design:
    - Atomic writes via temp file + os.replace() (no partial writes)
    - Versioned schema (schema_version field in every file)
    - Each state type has its own file (no monolithic dump)
    - Load returns defaults on missing/corrupt files (resilient boot)
    - Optional backup on save

File layout under data_dir/:
    identity/
        relationship_state.json
        personality_state.json
        self_model.json
        goal_queue.json
        identity_meta.json          ← schema version + timestamps

Future: migrate to SQLite when state grows beyond what JSON handles well.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from .relationship_state import RelationshipState
from .personality_state import PersonalityTraits
from .self_model import SelfModel
from .goal_queue import GoalQueue, GoalQueueConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "4.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersistenceConfig:
    data_dir: Path = Path("data/identity")
    create_backups: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# STORE
# ─────────────────────────────────────────────────────────────────────────────

class IdentityStore:
    """
    Persistence layer for Phase 4 cognitive identity.
    All operations are safe — never crash, return defaults on failure.
    """

    _FILES = {
        "relationship": "relationship_state.json",
        "personality": "personality_state.json",
        "self_model": "self_model.json",
        "goal_queue": "goal_queue.json",
        "meta": "identity_meta.json",
        "meta_history": "meta_history.json",
        "adaptive_meta": "adaptive_meta_state.json",
    }

    def __init__(self, config: Optional[PersistenceConfig] = None):
        self._cfg = config or PersistenceConfig()
        self._dir = Path(self._cfg.data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_meta()

    def _path(self, key: str) -> Path:
        return self._dir / self._FILES[key]

    # ── RELATIONSHIP STATE ────────────────────────────────────────────────

    def save_relationship_state(self, state: RelationshipState) -> bool:
        return self._save("relationship", state.to_dict())

    def load_relationship_state(self) -> RelationshipState:
        data = self._load("relationship")
        if data is None:
            return RelationshipState()
        try:
            return RelationshipState.from_dict(data)
        except Exception as exc:
            logger.warning(f"Corrupt relationship_state.json, using defaults: {exc}")
            return RelationshipState()

    # ── PERSONALITY STATE ─────────────────────────────────────────────────

    def save_personality_state(self, traits: PersonalityTraits) -> bool:
        return self._save("personality", traits.to_dict())

    def load_personality_state(self) -> PersonalityTraits:
        data = self._load("personality")
        if data is None:
            return PersonalityTraits()
        try:
            return PersonalityTraits.from_dict(data)
        except Exception as exc:
            logger.warning(f"Corrupt personality_state.json, using defaults: {exc}")
            return PersonalityTraits()

    # ── SELF MODEL ────────────────────────────────────────────────────────

    def save_self_model(self, model: SelfModel) -> bool:
        return self._save("self_model", model.to_dict())

    def load_self_model(self) -> SelfModel:
        data = self._load("self_model")
        if data is None:
            return SelfModel()
        try:
            return SelfModel.from_dict(data)
        except Exception as exc:
            logger.warning(f"Corrupt self_model.json, using defaults: {exc}")
            return SelfModel()

    # ── GOAL QUEUE ────────────────────────────────────────────────────────

    def save_goal_queue(self, queue: GoalQueue) -> bool:
        return self._save("goal_queue", queue.to_dict())

    def load_goal_queue(self, config: Optional[GoalQueueConfig] = None) -> GoalQueue:
        data = self._load("goal_queue")
        if data is None:
            return GoalQueue(config=config)
        try:
            return GoalQueue.from_dict(data, config=config)
        except Exception as exc:
            logger.warning(f"Corrupt goal_queue.json, using defaults: {exc}")
            return GoalQueue(config=config)

    # ── META HISTORY (Phase 4.6) ──────────────────────────────────────────

    def save_meta_history(self, snapshots: list, window_size: int = 20) -> bool:
        """
        Save meta-cognitive history snapshots to disk.

        Args:
            snapshots: list of IdentitySnapshot (duck-typed — uses .to_dict())
            window_size: only persist the last N snapshots
        """
        trimmed = snapshots[-window_size:] if len(snapshots) > window_size else snapshots
        data = {
            "window_size": window_size,
            "count": len(trimmed),
            "snapshots": [s.to_dict() for s in trimmed],
        }
        return self._save("meta_history", data)

    def load_meta_history(self) -> list:
        """
        Load meta-cognitive history snapshots from disk.

        Returns list of IdentitySnapshot instances.
        Returns empty list on missing/corrupt file.
        """
        data = self._load("meta_history")
        if data is None:
            return []
        try:
            from .meta_diagnostics import IdentitySnapshot
            raw_snapshots = data.get("snapshots", [])
            return [IdentitySnapshot.from_dict(s) for s in raw_snapshots]
        except Exception as exc:
            logger.warning(f"Corrupt meta_history.json, using empty: {exc}")
            return []

    # ── ADAPTIVE META STATE (Phase 5) ─────────────────────────────────────

    def save_adaptive_meta(self, state) -> bool:
        """
        Save adaptive meta-controller state to disk.

        Args:
            state: AdaptiveMetaState (duck-typed — uses .to_dict())
        """
        return self._save("adaptive_meta", state.to_dict())

    def load_adaptive_meta(self):
        """
        Load adaptive meta-controller state from disk.

        Returns AdaptiveMetaState instance.
        Returns fresh default state on missing/corrupt file.
        """
        data = self._load("adaptive_meta")
        if data is None:
            from .adaptive_meta import AdaptiveMetaState
            return AdaptiveMetaState()
        try:
            from .adaptive_meta import AdaptiveMetaState
            return AdaptiveMetaState.from_dict(data)
        except Exception as exc:
            logger.warning(f"Corrupt adaptive_meta_state.json, using defaults: {exc}")
            from .adaptive_meta import AdaptiveMetaState
            return AdaptiveMetaState()

    # ── SAVE ALL / LOAD ALL ───────────────────────────────────────────────

    def save_all(
        self,
        relationship: RelationshipState,
        personality: PersonalityTraits,
        self_model: SelfModel,
        goal_queue: GoalQueue,
    ) -> bool:
        """Save all identity states atomically (best-effort)."""
        ok = True
        ok &= self.save_relationship_state(relationship)
        ok &= self.save_personality_state(personality)
        ok &= self.save_self_model(self_model)
        ok &= self.save_goal_queue(goal_queue)
        self._update_meta()
        return ok

    # ── INTERNALS ─────────────────────────────────────────────────────────

    def _save(self, key: str, data: dict) -> bool:
        """Atomic JSON write: temp file → os.replace()."""
        path = self._path(key)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(UTC).isoformat(),
            "data": data,
        }
        try:
            # Write to temp file in same directory (same filesystem → atomic replace)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._dir), suffix=".tmp", prefix=f"{key}_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(envelope, f, indent=2, default=str)
                # Backup existing file
                if self._cfg.create_backups and path.exists():
                    backup = path.with_suffix(".json.bak")
                    try:
                        os.replace(str(path), str(backup))
                    except OSError:
                        pass
                # Atomic replace
                os.replace(tmp_path, str(path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.debug(f"Saved {key} → {path}")
            return True

        except Exception as exc:
            logger.error(f"Failed to save {key}: {exc}", exc_info=True)
            return False

    def _load(self, key: str) -> Optional[dict]:
        """Load JSON data. Returns None on missing/corrupt files."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
            # Version check
            version = envelope.get("schema_version", "0.0.0")
            if version != SCHEMA_VERSION:
                logger.warning(
                    f"{key}: schema version mismatch "
                    f"(file={version}, expected={SCHEMA_VERSION}). "
                    f"Loading anyway — may need migration."
                )
            return envelope.get("data", {})
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"Corrupt {key} file: {exc}")
            return None

    def _ensure_meta(self) -> None:
        """Create meta file if it doesn't exist."""
        meta_path = self._path("meta")
        if not meta_path.exists():
            self._update_meta()

    def _update_meta(self) -> None:
        """Update identity metadata."""
        self._save("meta", {
            "schema_version": SCHEMA_VERSION,
            "last_save_at": datetime.now(UTC).isoformat(),
            "files": list(self._FILES.values()),
        })

    @property
    def data_dir(self) -> Path:
        return self._dir

    def exists(self, key: str) -> bool:
        """Check if a specific state file exists."""
        return self._path(key).exists() if key in self._FILES else False