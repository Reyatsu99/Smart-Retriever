from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from smart_retriever_v2 import settings


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, expand_semantics: bool = True) -> list[str]:
    tokens = normalize_text(text).split()
    if not expand_semantics:
        return tokens

    expanded: list[str] = []
    alias_map: dict[str, set[str]] = {}
    for key, aliases in settings.SEMANTIC_ALIASES.items():
        alias_map.setdefault(key, set()).update(aliases)
        for alias in aliases:
            alias_map.setdefault(alias, set()).add(key)
            alias_map[alias].update(aliases)

    for token in tokens:
        expanded.append(token)
        expanded.extend(sorted(alias for alias in alias_map.get(token, set()) if alias != token))
    return expanded


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    if not text.strip():
        return [""]

    chunk_size = chunk_size or settings.MAX_CHUNK_CHARS
    overlap = overlap or settings.CHUNK_OVERLAP_CHARS
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append("\n".join(current))
            joined = "\n".join(current)
            tail = joined[-overlap:] if overlap else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_len = len("\n".join(current))
            continue
        current.append(paragraph)
        current_len += extra

    if current:
        chunks.append("\n".join(current))
    return chunks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    slug = normalize_text(value).replace(" ", "_")
    return slug or "general"


def mean_vector(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros(settings.EMBEDDING_DIM, dtype="float32")
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    if norm:
        mean = mean / norm
    return mean.astype("float32")
