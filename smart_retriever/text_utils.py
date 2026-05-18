from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from smart_retriever import settings


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
    
    # Separation priority: paragraphs, lines, sentences (including ? and !), and spaces
    # Each entry is (regex_pattern, join_character)
    separators = [
        (r"\n\n", "\n\n"),
        (r"\n", "\n"),
        (r"(?<=[.!?])\s+", " "),
        (r" ", " ")
    ]
    
    def get_clean_overlap(txt: str) -> str:
        if not overlap or len(txt) <= overlap:
            return txt
        # Take a slightly larger slice to find a good boundary
        raw_overlap = txt[-overlap:]
        # Find the first space to avoid starting with a partial word
        first_space = raw_overlap.find(" ")
        # If space is found early in the overlap, cut there for a clean start
        if 0 <= first_space < (overlap // 2):
            return raw_overlap[first_space:].lstrip()
        return raw_overlap

    def split_text(txt: str, seps: list[tuple[str, str]]) -> list[str]:
        if len(txt) <= chunk_size:
            return [txt]
        
        if not seps:
            # Fallback: hard character split if no more logical separators exist
            step = max(1, chunk_size - overlap)
            return [txt[i : i + chunk_size] for i in range(0, len(txt), step)]
            
        sep_pattern, sep_join = seps[0]
        parts = [p for p in re.split(sep_pattern, txt) if p]
        
        final_parts = []
        current_part = ""
        
        for p in parts:
            # Test if adding this part exceeds the limit
            potential_part = (current_part + sep_join + p) if current_part else p
            
            if len(potential_part) > chunk_size:
                if current_part:
                    final_parts.append(current_part)
                    # Start next chunk with a 'clean' overlap from the previous one
                    overlap_txt = get_clean_overlap(current_part)
                    current_part = (overlap_txt + sep_join + p) if overlap_txt else p
                else:
                    # Single part is already too big (e.g. huge paragraph), recurse to next separator
                    final_parts.extend(split_text(p, seps[1:]))
                    current_part = ""
            else:
                current_part = potential_part
        
        if current_part:
            # Final check for the last remaining piece
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
