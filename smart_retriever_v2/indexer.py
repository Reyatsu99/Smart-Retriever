from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from smart_retriever_v2 import settings
from smart_retriever_v2.bm25 import BM25Store
from smart_retriever_v2.embeddings import EmbeddingBackend
from smart_retriever_v2.parsers import extract_text
from smart_retriever_v2.text_utils import mean_vector, sha256_file, tokenize
from smart_retriever_v2.vector_index import VectorIndex

LOGGER = logging.getLogger(__name__)


def build_index(
    data_dir: Path | str = settings.DATA_DIR,
    index_dir: Path | str = settings.INDEX_DIR,
    force: bool = False,
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    embedder = embedder or EmbeddingBackend()
    routing_model = f"centroid-router:{embedder.model_name}"

    previous_manifest = _load_json(index_dir / "manifest.json", default={"files": []})
    previous_records = _load_json(index_dir / "file_metadata.json", default=[])
    previous_by_path = {record["relative_path"]: record for record in previous_records}
    previous_fingerprints = {record["relative_path"]: record for record in previous_manifest.get("files", [])}
    can_reuse_cached = (
        not force
        and previous_manifest.get("embedding_model") == embedder.model_name
        and previous_manifest.get("routing_model") == routing_model
    )

    changes = {"added": [], "modified": [], "deleted": [], "unchanged": 0}
    current_records: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for path in _discover_files(data_dir):
        stat = path.stat()
        relative_path = str(path.relative_to(data_dir))
        seen_paths.add(relative_path)
        fingerprint = {
            "relative_path": relative_path,
            "sha256": sha256_file(path),
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
        }
        cached_fingerprint = previous_fingerprints.get(relative_path)
        cached_record = previous_by_path.get(relative_path)
        if can_reuse_cached and cached_fingerprint and cached_record and _same_fingerprint(cached_fingerprint, fingerprint):
            reused = dict(cached_record)
            reused["path"] = str(path.resolve())
            current_records.append(reused)
            changes["unchanged"] += 1
            continue

        if cached_fingerprint:
            changes["modified"].append(relative_path)
        else:
            changes["added"].append(relative_path)

        try:
            raw_text = extract_text(path)
            content = f"{path.name}\n{raw_text}".strip()
            vector = embedder.encode_document(content).tolist()
            lexical_tokens = tokenize(f"{path.name}\n{relative_path}\n{raw_text}", expand_semantics=True)
            current_records.append(
                {
                    "relative_path": relative_path,
                    "file_name": path.name,
                    "path": str(path.resolve()),
                    "sha256": fingerprint["sha256"],
                    "mtime": fingerprint["mtime"],
                    "size": fingerprint["size"],
                    "shard": "",
                    "vector": vector,
                    "tokens": lexical_tokens,
                }
            )
        except Exception as exc:
            LOGGER.exception("Skipping %s due to indexing failure", relative_path)
            skipped_files.append({"relative_path": relative_path, "error": str(exc)})

    for relative_path in sorted(previous_fingerprints):
        if relative_path not in seen_paths:
            changes["deleted"].append(relative_path)

    shard_groups = _assign_shards(current_records)
    staging_dir = _prepare_staging_dir(index_dir)
    shard_metadata: dict[str, dict[str, Any]] = {}
    shard_vectors: list[np.ndarray] = []
    shard_items: list[str] = []

    for shard_name, records in sorted(shard_groups.items()):
        vectors = np.asarray([record["vector"] for record in records], dtype="float32")
        relative_paths = [record["relative_path"] for record in records]
        for record in records:
            record["shard"] = shard_name

        vector_index = VectorIndex.from_vectors(vectors)
        vector_path = staging_dir / f"files_{shard_name}.faiss"
        bm25_path = staging_dir / f"bm25_{shard_name}.pkl"
        meta_path = staging_dir / f"files_{shard_name}.meta.json"
        vector_index.save(vector_path)
        BM25Store.build([record["tokens"] for record in records]).save(bm25_path)
        meta_path.write_text(
            json.dumps({"artifact": vector_path.name, "items": relative_paths}, indent=2),
            encoding="utf-8",
        )

        shard_metadata[shard_name] = {
            "file_count": len(records),
            "artifact": vector_path.name,
            "bm25_file": bm25_path.name,
            "meta_file": meta_path.name,
        }
        shard_vectors.append(mean_vector(vectors))
        shard_items.append(shard_name)

    shard_index_path = staging_dir / "shard_index.faiss"
    shard_matrix = _stack_shard_vectors(shard_vectors, current_records, embedder)
    VectorIndex.from_vectors(shard_matrix).save(shard_index_path)
    (staging_dir / "shard_metadata.json").write_text(
        json.dumps(
            {
                "artifact": shard_index_path.name,
                "items": shard_items,
                "shards": shard_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    current_records = sorted(current_records, key=lambda item: item["relative_path"])
    (staging_dir / "file_metadata.json").write_text(json.dumps(current_records, indent=2), encoding="utf-8")

    manifest = {
        "version": settings.MANIFEST_VERSION,
        "data_dir": str(data_dir),
        "embedding_model": embedder.model_name,
        "routing_model": routing_model,
        "indexed_file_count": len(current_records),
        "skipped_file_count": len(skipped_files),
        "shards": {
            name: {"file_count": meta["file_count"]}
            for name, meta in shard_metadata.items()
        },
        "files": [
            {
                "relative_path": record["relative_path"],
                "file_name": record["file_name"],
                "path": record["path"],
                "sha256": record["sha256"],
                "mtime": record["mtime"],
                "size": record["size"],
                "shard": record["shard"],
            }
            for record in current_records
        ],
        "changes": changes,
        "errors": skipped_files,
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _swap_index_dir(staging_dir, index_dir)
    return manifest


def _discover_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in settings.SUPPORTED_EXTENSIONS
        and not (path.parent == data_dir and path.name.lower() == "readme.md")
    ]


def _same_fingerprint(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("sha256") == right.get("sha256")
        and int(left.get("mtime", 0)) == int(right.get("mtime", 0))
        and int(left.get("size", 0)) == int(right.get("size", 0))
    )


def _assign_shards(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not records:
        return {}
    if len(records) <= settings.MAX_FILES_PER_SHARD:
        return {"shard_0000": sorted(records, key=lambda item: item["relative_path"])}

    cluster_count = math.ceil(len(records) / settings.MAX_FILES_PER_SHARD)
    vectors = np.asarray([record["vector"] for record in records], dtype="float32")
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=settings.CLUSTER_SEED,
        n_init=10,
    ).fit_predict(vectors)

    buckets: dict[int, list[dict[str, Any]]] = {}
    for label, record in zip(labels, records):
        buckets.setdefault(int(label), []).append(record)

    shards: dict[str, list[dict[str, Any]]] = {}
    for idx, partition in enumerate(
        sorted(buckets.values(), key=lambda items: min(item["relative_path"] for item in items))
    ):
        shard_name = f"shard_{idx:04d}"
        shards[shard_name] = sorted(partition, key=lambda item: item["relative_path"])
    return shards


def _prepare_staging_dir(index_dir: Path) -> Path:
    staging_dir = index_dir.parent / settings.TMP_INDEX_DIR.name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir


def _stack_shard_vectors(
    shard_vectors: list[np.ndarray],
    records: list[dict[str, Any]],
    embedder: EmbeddingBackend,
) -> np.ndarray:
    if shard_vectors:
        return np.asarray(shard_vectors, dtype="float32")
    dimension = getattr(embedder, "embedding_dim", None)
    if dimension is None:
        dimension = len(records[0]["vector"]) if records else settings.EMBEDDING_DIM
    return np.zeros((0, int(dimension)), dtype="float32")


def _swap_index_dir(staging_dir: Path, index_dir: Path) -> None:
    backup_dir = index_dir.parent / settings.BACKUP_INDEX_DIR.name
    manifest_backup = index_dir.parent / settings.MANIFEST_BACKUP_NAME
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if index_dir.exists():
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            shutil.copy2(manifest_path, manifest_backup)
        index_dir.rename(backup_dir)
    staging_dir.rename(index_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
