import logging
import gc
from pathlib import Path
from typing import Any

from smart_retriever_v2 import settings
from smart_retriever_v2.db import get_db, get_or_create_table
from smart_retriever_v2.embeddings import EmbeddingBackend
from smart_retriever_v2.parsers import extract_text
from smart_retriever_v2.text_utils import sha256_file, chunk_text

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
    db = get_db(index_dir.parent / "lancedb_store")

    if force and "document_chunks" in db.table_names():
        db.drop_table("document_chunks")

    table = get_or_create_table(db, "document_chunks")

    changes = {"added": [], "modified": [], "deleted": [], "unchanged": 0}
    skipped_files: list[dict[str, str]] = []
    
    # Load existing fingerprints to simulate incremental cache
    # LanceDB doesn't natively do purely distinct without querying, so we load tiny metadata map
    existing_fingerprints = {}
    if table.count_rows() > 0:
        # Use Arrow directly to avoid pandas dependency
        arrow_table = table.search().limit(100000).select(["relative_path", "sha256"]).to_arrow()
        for row in arrow_table.to_pylist():
            existing_fingerprints[row["relative_path"]] = row["sha256"]

    file_batch = []
    chunk_batch = []
    metadata_batch = []
    seen_paths = set()

    for path in _discover_files(data_dir):
        relative_path = str(path.relative_to(data_dir))
        seen_paths.add(relative_path)
        stat = path.stat()
        file_sha = sha256_file(path)

        # Incremental logic check
        if not force and relative_path in existing_fingerprints:
            if existing_fingerprints[relative_path] == file_sha:
                changes["unchanged"] += 1
                continue
            else:
                changes["modified"].append(relative_path)
                table.delete(f"relative_path = '{relative_path}'")
        else:
            changes["added"].append(relative_path)

        try:
            raw_text = extract_text(path)
            chunks = [c for c in chunk_text(raw_text) if c.strip()]
            
            for chunk_id, chunk in enumerate(chunks):
                chunk_batch.append(chunk)
                metadata_batch.append({
                    "relative_path": relative_path,
                    "chunk_id": chunk_id,
                    "mtime": float(stat.st_mtime),
                    "size": int(stat.st_size),
                    "sha256": file_sha,
                })
        except Exception as exc:
            LOGGER.exception("Skipping %s due to indexing failure", relative_path)
            skipped_files.append({"relative_path": relative_path, "error": str(exc)})

        # Process batches to keep memory stable
        # Increased batch size to 512 to utilize more available RAM (up to 8GB)
        if len(chunk_batch) >= 512:
            vectors = embedder.encode(chunk_batch)
            rows = []
            for vec, meta, txt in zip(vectors, metadata_batch, chunk_batch):
                meta["vector"] = vec.tolist()
                meta["text"] = txt
                rows.append(meta)
            table.add(rows)
            chunk_batch = []
            metadata_batch = []

    if chunk_batch:
        vectors = embedder.encode(chunk_batch)
        rows = []
        for vec, meta, txt in zip(vectors, metadata_batch, chunk_batch):
            meta["vector"] = vec.tolist()
            meta["text"] = txt
            rows.append(meta)
        table.add(rows)
        gc.collect()

    # Create Full-Text Search (FTS) index for Hybrid Retrieval
    # This requires tantivy (handled by lancedb)
    if table.count_rows() > 0:
        try:
            table.create_fts_index("text", replace=True)
        except Exception as exc:
            LOGGER.warning("Could not create FTS index (keyword search will be disabled): %s", exc)

    # Clean up deleted files from DB
    for relative_path in existing_fingerprints:
        if relative_path not in seen_paths:
            changes["deleted"].append(relative_path)
            table.delete(f"relative_path = '{relative_path}'")

    indexed_files = len(changes["added"]) + len(changes["modified"]) + changes["unchanged"]

    manifest = {
        "version": 4,  # Flagging V3 Architecture
        "data_dir": str(data_dir),
        "embedding_model": embedder.model_name,
        "indexed_file_count": indexed_files,
        "indexed_chunks": table.count_rows() if "document_chunks" in db.table_names() else 0,
        "skipped_file_count": len(skipped_files),
        "changes": changes,
        "errors": skipped_files,
    }

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
