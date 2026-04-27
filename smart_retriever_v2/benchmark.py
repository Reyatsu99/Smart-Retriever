from __future__ import annotations

import json
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from smart_retriever_v2 import settings
from smart_retriever_v2.embeddings import EmbeddingBackend
from smart_retriever_v2.indexer import build_index
from smart_retriever_v2.text_utils import tokenize

QUERY_TEMPLATES = {
    "finance": "budget report for {subject}",
    "hr": "offer letter for {subject}",
    "legal": "contract details for {subject}",
    "operations": "runbook procedure for {subject}",
    "projects": "project proposal for {subject}",
    "research": "research analysis for {subject}",
}


class StubEmbedder:
    model_name = "benchmark-stub-embeddings"
    embedding_dim = 128

    def encode_document(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype="float32")
        for token in tokenize(text, expand_semantics=False):
            ordinal_sum = sum(ord(char) for char in token)
            slot_a = ordinal_sum % self.embedding_dim
            slot_b = (ordinal_sum * 7 + len(token)) % self.embedding_dim
            vector[slot_a] += 1.0
            vector[slot_b] += 0.5
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector


def run_benchmark(
    files: int = 600,
    queries: int = 60,
    modified_files: int = 30,
    top_k: int = 5,
    mode: str = "stub",
    work_dir: Path | str = Path(".benchmarks/v2/latest"),
    keep_workspace: bool = True,
) -> dict[str, Any]:
    if files < 1:
        raise ValueError("files must be positive")
    if queries < 1:
        raise ValueError("queries must be positive")

    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    data_dir = work_dir / "data"
    index_dir = work_dir / "index"
    data_dir.mkdir(parents=True, exist_ok=True)

    embedder = _select_backends(mode)
    corpus = _generate_corpus(data_dir, files)

    started = time.perf_counter()
    manifest = build_index(data_dir, index_dir, embedder=embedder)
    index_seconds = time.perf_counter() - started

    from smart_retriever_v2.search import SearchEngine

    search_engine = SearchEngine(index_dir, embedder=embedder)
    query_set = corpus[: min(queries, len(corpus))]
    latencies_ms: list[float] = []
    top1_hits = 0
    top3_hits = 0

    for item in query_set:
        started = time.perf_counter()
        results = search_engine.search(item["query"], top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        ranked_paths = [result["relative_path"] for result in results]
        if ranked_paths[:1] == [item["relative_path"]]:
            top1_hits += 1
        if item["relative_path"] in ranked_paths[:3]:
            top3_hits += 1

    changed = _modify_files(data_dir, corpus, modified_files)
    started = time.perf_counter()
    incremental_manifest = build_index(data_dir, index_dir, embedder=embedder)
    incremental_seconds = time.perf_counter() - started

    # Calculate Shard Balance
    if manifest["shards"]:
        shard_sizes = [s["file_count"] for s in manifest["shards"].values()]
        shard_balance_ratio = round(max(shard_sizes) / min(shard_sizes), 3)
    else:
        shard_balance_ratio = 1.0

    # Calculate Storage Ratio
    total_data_size = sum(item["size"] for item in manifest["files"])
    def get_dir_size(p):
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    total_index_size = get_dir_size(index_dir)
    storage_ratio = round(total_index_size / total_data_size, 3) if total_data_size > 0 else 0

    # Calculate Routing Accuracy (how often the correct shard is in the top candidate shards)
    routing_hits = 0
    for item in query_set:
        try:
            # We peek into the SearchEngine's internal _candidate_shards logic
            query_vector = embedder.encode_document(item["query"])
            candidates = search_engine._candidate_shards(query_vector)
            # Find which shard the document actually belongs to
            actual_shard = next(f["shard"] for f in manifest["files"] if f["relative_path"] == item["relative_path"])
            if actual_shard in candidates:
                routing_hits += 1
        except Exception:
            pass
    routing_recall = round(routing_hits / len(query_set), 3) if query_set else 0

    report = {
        "mode": mode,
        "workspace": str(work_dir.resolve()),
        "files_indexed": manifest["indexed_file_count"],
        "queries_run": len(query_set),
        "modified_files": changed,
        "performance": {
            "full_index_time_sec": round(index_seconds, 4),
            "incremental_index_time_sec": round(incremental_seconds, 4),
            "indexing_throughput_fps": round(manifest["indexed_file_count"] / index_seconds, 2) if index_seconds > 0 else 0,
            "indexing_speedup_factor": round(index_seconds / incremental_seconds, 2) if incremental_seconds > 0 else 0,
            "search_avg_ms": round(statistics.mean(latencies_ms), 3),
            "search_p50_ms": round(_percentile(latencies_ms, 50), 3),
            "search_p95_ms": round(_percentile(latencies_ms, 95), 3),
            "queries_per_second": round(1000.0 / statistics.mean(latencies_ms), 3),
        },
        "accuracy": {
            "top1_hit_rate": round(top1_hits / len(query_set), 3),
            "top3_hit_rate": round(top3_hits / len(query_set), 3),
            "shard_routing_recall": routing_recall,
        },
        "architecture": {
            "shard_count": len(manifest["shards"]),
            "shard_balance_ratio": shard_balance_ratio,
            "storage_expansion_ratio": storage_ratio,
            "total_index_size_kb": round(total_index_size / 1024, 2),
        },
        "changes": incremental_manifest["changes"],
        "routing_model": manifest["routing_model"],
    }

    report_path = work_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    if not keep_workspace:
        shutil.rmtree(work_dir)
    return report


def _select_backends(mode: str) -> Any:
    if mode == "real":
        return EmbeddingBackend()
    if mode == "stub":
        return StubEmbedder()
    raise ValueError("mode must be 'stub' or 'real'")


def _generate_corpus(data_dir: Path, files: int) -> list[dict[str, str]]:
    topics = list(QUERY_TEMPLATES)
    randomizer = random.Random(42)
    corpus: list[dict[str, str]] = []

    for index in range(files):
        topic = topics[index % len(topics)]
        subject = f"subject_{index:05d}"
        file_name = f"{topic}_{subject}.txt"
        relative_path = Path(topic) / file_name
        target = data_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        hints = sorted(tokenize(QUERY_TEMPLATES[topic], expand_semantics=True))
        randomizer.shuffle(hints)
        selected_hints = hints[:4]
        content = "\n".join(
            [
                f"Document owner: {subject}",
                f"Topic: {topic}",
                f"Search key: {subject} {subject}",
                f"Keywords: {' '.join(selected_hints)}",
                f"This file contains {topic} material for {subject}.",
            ]
        )
        target.write_text(content, encoding="utf-8")
        corpus.append(
            {
                "topic": topic,
                "relative_path": str(relative_path),
                "query": QUERY_TEMPLATES[topic].format(subject=subject),
            }
        )
    return corpus


def _modify_files(data_dir: Path, corpus: list[dict[str, str]], modified_files: int) -> int:
    changed = 0
    for item in corpus[: min(modified_files, len(corpus))]:
        target = data_dir / item["relative_path"]
        target.write_text(
            target.read_text(encoding="utf-8") + "\nIncremental update marker.",
            encoding="utf-8",
        )
        changed += 1
    return changed


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[index]
