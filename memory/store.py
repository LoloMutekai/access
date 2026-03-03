import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from .config import MemoryConfig
from .models import MemoryRecord

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    summary           TEXT NOT NULL DEFAULT '',
    memory_type       TEXT NOT NULL DEFAULT 'episodic',
    tags              TEXT NOT NULL DEFAULT '',        -- comma-separated
    importance_score  REAL NOT NULL DEFAULT 0.5,
    created_at        TEXT NOT NULL,
    last_accessed_at  TEXT,
    access_count      INTEGER NOT NULL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'interaction',
    session_id        TEXT
);

-- Fast lookups
CREATE INDEX IF NOT EXISTS idx_memory_type     ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_importance      ON memories(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_created_at      ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_id      ON memories(session_id);

-- Version tracking for future migrations
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');
INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_model', 'all-MiniLM-L6-v2');
INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_dim', '384');
"""


class MemoryStore:
    """
    Handles all SQLite operations for memory metadata.
    Does NOT know about embeddings — that's the VectorIndex's job.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"MemoryStore initialized at {self.config.db_path}")

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.config.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent write perf
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────

    def insert(self, record: MemoryRecord) -> None:
        row = record.to_db_dict()
        sql = """
            INSERT INTO memories (
                id, content, summary, memory_type, tags,
                importance_score, created_at, last_accessed_at,
                access_count, source, session_id
            ) VALUES (
                :id, :content, :summary, :memory_type, :tags,
                :importance_score, :created_at, :last_accessed_at,
                :access_count, :source, :session_id
            )
        """
        with self._get_conn() as conn:
            conn.execute(sql, row)
        logger.debug(f"Inserted memory {record.id[:8]}...")

    def update_importance(self, memory_id: str, new_score: float) -> None:
        score = max(self.config.importance_min, min(self.config.importance_max, new_score))
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE memories SET importance_score = ? WHERE id = ?",
                (score, memory_id)
            )

    def mark_accessed(self, memory_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE memories 
                   SET last_accessed_at = ?, access_count = access_count + 1
                   WHERE id = ?""",
                (datetime.utcnow().isoformat(), memory_id)
            )

    def delete(self, memory_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    # ─────────────────────────────────────────
    # READ
    # ─────────────────────────────────────────

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return MemoryRecord.from_db_row(dict(row)) if row else None

    def get_by_ids(self, memory_ids: list[str]) -> list[MemoryRecord]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                memory_ids
            ).fetchall()
        id_to_record = {r["id"]: MemoryRecord.from_db_row(dict(r)) for r in rows}
        # Preserve input order
        return [id_to_record[mid] for mid in memory_ids if mid in id_to_record]

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get_schema_meta(self) -> dict:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_all_ids(self) -> list[str]:
        """Return all memory IDs — used for consistency checks."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT id FROM memories").fetchall()
        return [r["id"] for r in rows]

    def get_all_for_decay(self) -> list[dict]:
        """
        Return lightweight records for decay computation.
        Excludes content/summary to minimize memory usage.
        """
        sql = """
            SELECT id, memory_type, importance_score,
                   created_at, last_accessed_at, access_count
            FROM memories
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "memory_type": r["memory_type"],
                "importance_score": r["importance_score"],
                "created_at": datetime.fromisoformat(r["created_at"]),
                "last_accessed_at": datetime.fromisoformat(r["last_accessed_at"]) if r["last_accessed_at"] else None,
                "access_count": r["access_count"],
            })
        return results

    def bulk_update_importance(self, updates: list[tuple[float, str]]) -> None:
        """
        Batch update importance scores.
        Args: list of (new_score, memory_id) tuples
        """
        with self._get_conn() as conn:
            conn.executemany(
                "UPDATE memories SET importance_score = ? WHERE id = ?",
                updates
            )

    def get_purgeable_candidates(
        self,
        exclude_types: tuple,
        max_importance: float,
    ) -> list[dict]:
        """
        Return memories eligible for purge:
        - Not in excluded types
        - Below max_importance threshold
        Sorted by importance ASC (cheapest to delete first).
        """
        placeholders = ",".join("?" * len(exclude_types))
        sql = f"""
            SELECT id, memory_type, importance_score,
                   created_at, last_accessed_at, access_count
            FROM memories
            WHERE memory_type NOT IN ({placeholders})
              AND importance_score < ?
            ORDER BY importance_score ASC
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql, (*exclude_types, max_importance)).fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "memory_type": r["memory_type"],
                "importance_score": r["importance_score"],
                "created_at": datetime.fromisoformat(r["created_at"]),
                "last_accessed_at": datetime.fromisoformat(r["last_accessed_at"]) if r["last_accessed_at"] else None,
                "access_count": r["access_count"],
            })
        return results
    def get_recent_memories(
        self,
        n: int = 10,
        min_importance: float = 0.0,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """
        Return most recently created memories, newest first.
        Used by EmotionEngine for protection tagging.
        """
        conditions = ["importance_score >= ?"]
        params: list = [min_importance]
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        where = " AND ".join(conditions)
        params.append(n)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ?"
        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [MemoryRecord.from_db_row(dict(r)) for r in rows]

    def add_tags(self, memory_id: str, new_tags: list[str]) -> None:
        """
        Append new_tags to a memory's existing tags (no duplicates).
        No-op if memory_id does not exist.
        """
        record = self.get_by_id(memory_id)
        if record is None:
            logger.warning(f"add_tags: memory not found: {memory_id[:8]}...")
            return
        existing = set(record.tags)
        merged = sorted(existing | set(new_tags))  # deterministic order
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE memories SET tags = ? WHERE id = ?",
                (",".join(merged), memory_id)
            )
        logger.debug(f"Tags updated for {memory_id[:8]}...: {merged}")