from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from smart_retriever import settings
from smart_retriever.text_utils import chunk_text


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingBackend:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    @property
    def embedding_dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        if not batch:
            return np.empty((0, self.embedding_dim), dtype="float32")
        vectors = self._model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_document(self, text: str) -> np.ndarray:
        chunks = [chunk for chunk in chunk_text(text) if chunk.strip()]
        if not chunks:
            chunks = [""]
        vectors = self.encode(chunks)
        if len(vectors) == 1:
            return vectors[0]
        mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean)
        if norm:
            mean = mean / norm
        return mean.astype("float32")
