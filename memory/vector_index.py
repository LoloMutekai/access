import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from .config import MemoryConfig

logger = logging.getLogger(__name__)

# FAISS index save paths
_INDEX_FILE = "faiss.index"
_IDMAP_FILE = "faiss_idmap.pkl"   # FAISS uses int IDs; this maps int ↔ str UUID


class VectorIndex:
    """
    Manages the FAISS index for semantic similarity search.
    
    Knows nothing about SQLite — only stores (int_id → str_uuid) mappings
    and float32 embedding vectors.
    
    Design decision: IndexFlatIP (inner product on normalized vectors = cosine similarity).
    This is the cleanest approach for Phase 1. Can swap to IVF for scale later.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.dim = config.embedding_dim

        self._index_path = config.data_dir / _INDEX_FILE
        self._idmap_path = config.data_dir / _IDMAP_FILE

        # Bidirectional mapping: FAISS int ↔ str UUID
        self._int_to_uuid: dict[int, str] = {}
        self._uuid_to_int: dict[str, int] = {}
        self._next_id: int = 0

        self._index = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        if self._index_path.exists() and self._idmap_path.exists():
            index = faiss.read_index(str(self._index_path))
            with open(self._idmap_path, "rb") as f:
                saved = pickle.load(f)
                self._int_to_uuid = saved["int_to_uuid"]
                self._uuid_to_int = saved["uuid_to_int"]
                self._next_id = saved["next_id"]
            logger.info(f"VectorIndex loaded — {index.ntotal} vectors")
            return index
        else:
            # IndexFlatIP: exact cosine similarity (after L2 normalization)
            index = faiss.IndexFlatIP(self.dim)
            logger.info(f"VectorIndex created — dim={self.dim}")
            return index

    def save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._idmap_path, "wb") as f:
            pickle.dump({
                "int_to_uuid": self._int_to_uuid,
                "uuid_to_int": self._uuid_to_int,
                "next_id": self._next_id,
            }, f)
        logger.debug(f"VectorIndex saved — {self._index.ntotal} vectors")

    # ─────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────

    def add(self, memory_id: str, embedding: list[float]) -> None:
        vec = self._normalize(np.array(embedding, dtype=np.float32))
        int_id = self._next_id

        self._index.add(vec.reshape(1, -1))
        self._int_to_uuid[int_id] = memory_id
        self._uuid_to_int[memory_id] = int_id
        self._next_id += 1

        self.save()
        logger.debug(f"Added vector for {memory_id[:8]}... (int_id={int_id})")

    def remove(self, memory_id: str) -> None:
        """
        Note: IndexFlatIP doesn't support deletion natively.
        We mark as removed and rebuild index if needed.
        For Phase 1 scale (<10k entries), this is fine.
        """
        if memory_id not in self._uuid_to_int:
            logger.warning(f"Tried to remove unknown memory_id: {memory_id}")
            return

        int_id = self._uuid_to_int.pop(memory_id)
        self._int_to_uuid.pop(int_id)
        # Full rebuild (acceptable at MVP scale)
        self._rebuild_index()
        self.save()

    def _rebuild_index(self) -> None:
        """Reconstruct FAISS index from scratch (used after deletions)."""
        new_index = faiss.IndexFlatIP(self.dim)
        if self._index.ntotal > 0:
            all_vecs = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self.dim
            ).reshape(self._index.ntotal, self.dim)
            valid_int_ids = set(self._int_to_uuid.keys())
            valid_vecs = np.array(
                [all_vecs[i] for i in range(self._index.ntotal) if i in valid_int_ids],
                dtype=np.float32
            )
            if len(valid_vecs) > 0:
                new_index.add(valid_vecs)
        self._index = new_index

    # ─────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """
        Returns list of (memory_uuid, cosine_similarity) sorted by similarity desc.
        Filters out scores below config.min_similarity_score.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        vec = self._normalize(np.array(query_embedding, dtype=np.float32))

        scores, int_ids = self._index.search(vec.reshape(1, -1), k)
        scores = scores[0]
        int_ids = int_ids[0]

        results = []
        for score, int_id in zip(scores, int_ids):
            if int_id == -1:  # FAISS padding
                continue
            if score < self.config.min_similarity_score:
                continue
            uuid = self._int_to_uuid.get(int(int_id))
            if uuid:
                results.append((uuid, float(score)))

        return results  # already sorted by FAISS (desc)

    # ─────────────────────────────────────────
    # UTILS
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalization → IndexFlatIP becomes cosine similarity."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def get_all_uuids(self) -> list[str]:
        """Return all UUIDs currently tracked in the FAISS id map."""
        return list(self._uuid_to_int.keys())