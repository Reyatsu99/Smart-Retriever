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
    
    # Separation priority: paragraphs, lines, sentences, spaces
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(txt: str, seps: list[str]) -> list[str]:
        if len(txt) <= chunk_size:
            return [txt]
        
        if not seps:
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - overlap)]
            
        sep = seps[0]
        parts = txt.split(sep)
        
        final_parts = []
        current_part = ""
        
        for p in parts:
            if current_part and len(current_part) + len(sep) + len(p) > chunk_size:
                final_parts.append(current_part)
                # Keep overlap from previous part
                current_part = current_part[-overlap:] + sep + p if overlap else p
            else:
                current_part = (current_part + sep + p) if current_part else p
        
        if current_part:
            # If the resulting part is still too big, go to next separator
            if len(current_part) > chunk_size:
                final_parts.extend(split_text(current_part, seps[1:]))
            else:
                final_parts.append(current_part)
                
        return final_parts

    return [c.strip() for c in split_text(text, separators) if c.strip()]


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
