from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryConfig:
    # Paths
    data_dir: Path = Path("data/memory")
    db_path: Path = None          # auto-resolved below
    faiss_index_path: Path = None # auto-resolved below

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval
    default_top_k: int = 5
    min_similarity_score: float = 0.25  # below this → ignored

    # Importance
    default_importance: float = 0.5
    importance_min: float = 0.0
    importance_max: float = 1.0

    # Memory limits
    max_episodic_entries: int = 10_000  # purge policy trigger

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path is None:
            self.db_path = self.data_dir / "memory.db"
        if self.faiss_index_path is None:
            self.faiss_index_path = self.data_dir / "faiss.index"