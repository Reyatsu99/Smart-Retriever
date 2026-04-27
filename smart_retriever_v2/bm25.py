from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, index: BM25Okapi) -> None:
        self.index = index

    @classmethod
    def build(cls, corpus: Sequence[Sequence[str]]) -> "BM25Store":
        return cls(BM25Okapi(list(corpus)))

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        with path.open("rb") as handle:
            return cls(pickle.load(handle))

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self.index, handle)

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        return [float(score) for score in self.index.get_scores(list(query_tokens))]
