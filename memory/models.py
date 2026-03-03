from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class MemoryRecord:
    """
    Represents a single episodic memory entry.
    
    Separation of concerns:
    - `id`         → primary key in SQLite AND identifier in FAISS (via id_map)
    - `embedding`  → stored ONLY in FAISS, never in SQLite
    - everything else → stored ONLY in SQLite
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Content
    content: str = ""           # raw text of the memory
    summary: str = ""           # compressed version (for RAG injection)
    
    # Classification
    memory_type: str = "episodic"   # episodic | semantic | emotional | performance
    tags: list[str] = field(default_factory=list)
    
    # Scoring
    importance_score: float = 0.5   # 0.0 → forgettable | 1.0 → critical
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    
    # Source context
    source: str = "interaction"     # interaction | autonomous | os_event | reflection
    session_id: Optional[str] = None

    # Embedding (transient — not persisted in SQLite, only in FAISS)
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_db_dict(self) -> dict:
        """Serialize for SQLite (excludes embedding)."""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type,
            "tags": ",".join(self.tags),
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "source": self.source,
            "session_id": self.session_id,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "MemoryRecord":
        """Deserialize from SQLite row (embedding will be None)."""
        return cls(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            memory_type=row["memory_type"],
            tags=row["tags"].split(",") if row["tags"] else [],
            importance_score=row["importance_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=row["access_count"],
            source=row["source"],
            session_id=row["session_id"],
        )


@dataclass
class RetrievedMemory:
    """A memory record augmented with its retrieval similarity score."""
    record: MemoryRecord
    similarity: float   # cosine similarity [0.0, 1.0]
    relevance: float    # combined score: similarity * importance_score

    def __repr__(self):
        return (
            f"RetrievedMemory(id={self.record.id[:8]}..., "
            f"similarity={self.similarity:.3f}, "
            f"relevance={self.relevance:.3f}, "
            f"summary='{self.record.summary[:60]}...')"
        )