from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FAISSIndex:
    """Cosine-similarity index implemented as inner-product search
    over L2-normalized vectors.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {embeddings.shape}"
            )
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, k: int = 10):
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        scores, ids = self.index.search(query_embeddings, k)
        return scores, ids

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FAISSIndex":
        index = faiss.read_index(str(path))
        obj = cls(index.d)
        obj.index = index
        return obj
