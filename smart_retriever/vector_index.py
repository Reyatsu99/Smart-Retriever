from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> "VectorIndex":
        if vectors.ndim != 2:
            raise ValueError("Expected a 2D array of vectors.")
        instance = cls(vectors.shape[1])
        if len(vectors):
            instance.index.add(np.asarray(vectors, dtype="float32"))
        return instance

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        index = faiss.read_index(str(path))
        instance = cls(index.d)
        instance.index = index
        return instance

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = np.asarray(query_vector, dtype="float32").reshape(1, self.dimension)
        scores, indices = self.index.search(query, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
