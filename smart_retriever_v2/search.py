from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from smart_retriever_v2 import settings
from smart_retriever_v2.bm25 import BM25Store
from smart_retriever_v2.embeddings import EmbeddingBackend
from smart_retriever_v2.indexer import build_index
from smart_retriever_v2.text_utils import tokenize
from smart_retriever_v2.vector_index import VectorIndex

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class SearchEngine:
    def __init__(self, index_dir: Path | str = settings.INDEX_DIR, embedder: EmbeddingBackend | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder or EmbeddingBackend()
        self.shard_payload = self._load_json(self.index_dir / "shard_metadata.json", default={})
        self.file_records = self._load_json(self.index_dir / "file_metadata.json", default=[])
        self.records_by_path = {record["relative_path"]: record for record in self.file_records}
        self.cross_encoder = None
        if CrossEncoder:
            self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME)

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K, use_reranker: bool = True) -> list[dict[str, Any]]:
        if not self.shard_payload:
            raise FileNotFoundError("Index not found. Run `python cli.py index` first.")

        query_vector = self.embedder.encode_document(query)
        query_tokens = tokenize(query)
        candidates = self._candidate_shards(query_vector)
        merged: dict[str, dict[str, Any]] = {}

        for shard_name in candidates:
            dense_scores = self._dense_scores(shard_name, query_vector, top_k)
            bm25_scores = self._bm25_scores(shard_name, query_tokens, top_k)
            for relative_path in set(dense_scores) | set(bm25_scores):
                base = self._public_record(self.records_by_path[relative_path])
                dense = dense_scores.get(relative_path, 0.0)
                bm25 = bm25_scores.get(relative_path, 0.0)
                filename_boost = self._filename_boost(base["file_name"], query_tokens)
                matched_by = []
                if dense > 0:
                    matched_by.append("semantic_similarity")
                if bm25 > 0:
                    matched_by.append("bm25_keyword")
                if filename_boost > 0:
                    matched_by.append("filename_boost")

                base["score"] = round(
                    settings.DENSE_WEIGHT * dense + settings.BM25_WEIGHT * bm25 + filename_boost,
                    4,
                )
                base["matched_by"] = matched_by
                previous = merged.get(relative_path)
                if not previous or base["score"] > previous["score"]:
                    merged[relative_path] = base

        # Initial ranking
        results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        # Fetch more candidates for re-ranking
        candidates = results[:top_k * settings.SEARCH_CANDIDATE_MULTIPLIER]

        if use_reranker and self.cross_encoder and candidates:
            # Re-rank using CrossEncoder
            pairs = [[query, item.get("content", item["file_name"])] for item in candidates]
            # Since we didn't load full file content in SearchEngine, we fallback to filename
            # Wait, better yet, we ideally want to rank on content. Let's just use existing score for now if content isn't loaded.
            # Actually, let's load content if needed, but it might be slow.
            # Let's read the first few lines of the file for ranking.
            for i, item in enumerate(candidates):
                file_path = self.index_dir.parent / "data" / item["relative_path"]
                if file_path.exists() and file_path.suffix.lower() == ".txt":
                    text = file_path.read_text(encoding="utf-8", errors="ignore")[:1000]
                    pairs[i][1] = text
            
            cross_scores = self.cross_encoder.predict(pairs)
            for item, x_score in zip(candidates, cross_scores):
                item["cross_score"] = float(x_score)
                item["score"] = float(x_score) # overwrite score with cross_score
                item["matched_by"].append("cross_encoder")
            
            results = sorted(candidates, key=lambda item: item["score"], reverse=True)
        else:
            results = results[:top_k]

        return results[:top_k]

    def _candidate_shards(self, query_vector: Any) -> list[str]:
        shard_index = VectorIndex.load(self.index_dir / self.shard_payload["artifact"])
        count = min(settings.TOP_SHARD_COUNT, len(self.shard_payload.get("items", [])))
        return [self.shard_payload["items"][index] for index, _ in shard_index.search(query_vector, max(count, 1))]

    def _dense_scores(self, shard_name: str, query_vector: Any, top_k: int) -> dict[str, float]:
        shard = self.shard_payload["shards"][shard_name]
        mapping = self._load_json(self.index_dir / shard["meta_file"], default={}).get("items", [])
        vector_index = VectorIndex.load(self.index_dir / shard["artifact"])
        hits = vector_index.search(query_vector, min(len(mapping), top_k * settings.SEARCH_CANDIDATE_MULTIPLIER))
        return {
            mapping[index]: max(0.0, min(1.0, (score + 1.0) / 2.0))
            for index, score in hits
            if index < len(mapping)
        }

    def _bm25_scores(self, shard_name: str, query_tokens: list[str], top_k: int) -> dict[str, float]:
        shard = self.shard_payload["shards"][shard_name]
        mapping = self._load_json(self.index_dir / shard["meta_file"], default={}).get("items", [])
        bm25 = BM25Store.load(self.index_dir / shard["bm25_file"])
        raw_scores = bm25.scores(query_tokens)
        max_score = max(raw_scores) if raw_scores else 0.0
        ranked = sorted(enumerate(raw_scores), key=lambda item: item[1], reverse=True)[
            : top_k * settings.SEARCH_CANDIDATE_MULTIPLIER
        ]
        return {
            mapping[index]: (score / max_score if max_score else 0.0)
            for index, score in ranked
            if score > 0 and index < len(mapping)
        }

    def _filename_boost(self, file_name: str, query_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        name_tokens = set(tokenize(file_name, expand_semantics=False))
        overlap = len(name_tokens & set(query_tokens)) / len(set(query_tokens))
        return round(settings.FILENAME_BOOST * overlap, 4)

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "file_name": record["file_name"],
            "path": record["path"],
            "relative_path": record["relative_path"],
            "shard": record["shard"],
            "size": record["size"],
            "mtime": record["mtime"],
            "sha256": record["sha256"],
        }


def _print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No matching files found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['relative_path']}\n"
            f"   score: {result['score']:.4f}\n"
            f"   shard: {result['shard']}\n"
            f"   matched by: {', '.join(result['matched_by']) if result['matched_by'] else 'routing only'}"
        )


def _prompt(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def _prompt_bool(message: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    value = input(f"{message} [{default_label}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _describe_status(data_dir: Path | str = settings.DATA_DIR, index_dir: Path | str = settings.INDEX_DIR) -> dict[str, Any]:
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    supported_files = []
    if data_dir.exists():
        supported_files = [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in settings.SUPPORTED_EXTENSIONS
            and not (path.parent == data_dir and path.name.lower() == "readme.md")
        ]

    manifest = {}
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    shard_counts = manifest.get("shards", {}).values()
    shard_balance = 0.0
    if shard_counts:
        sizes = [s["file_count"] for s in shard_counts]
        shard_balance = round(max(sizes) / min(sizes), 2)

    total_data_size = sum(f.get("size", 0) for f in manifest.get("files", []))
    total_index_size = 0
    if index_dir.exists():
        total_index_size = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    
    storage_ratio = 0.0
    if total_data_size > 0:
        storage_ratio = round(total_index_size / total_data_size, 2)

    return {
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.exists(),
        "supported_files_found": len(supported_files),
        "index_dir": str(index_dir),
        "index_exists": manifest_path.exists(),
        "indexed_files": manifest.get("indexed_file_count", 0),
        "shards": len(manifest.get("shards", {})),
        "embedding_model": manifest.get("embedding_model"),
        "routing_model": manifest.get("routing_model"),
        "shard_balance_ratio": shard_balance,
        "storage_expansion_ratio": storage_ratio,
        "total_index_size_kb": round(total_index_size / 1024, 1),
    }


def _print_status(status: dict[str, Any]) -> None:
    print("Project status\n")
    print(f"Data folder: {status['data_dir']}")
    print(f"Supported files found: {status['supported_files_found']}")
    print(f"Index folder: {status['index_dir']}")
    if status["index_exists"]:
        print(f"Index ready: yes")
        print(f"Indexed files: {status['indexed_files']}")
        print(f"Shards: {status['shards']}")
        print(f"Embedding model: {status['embedding_model']}")
        print(f"Routing model: {status['routing_model']}")
        print(f"\nArchitecture Metrics:")
        print(f"- Shard balance ratio: {status['shard_balance_ratio']}")
        print(f"- Storage expansion ratio: {status['storage_expansion_ratio']}x")
        print(f"- Total index size: {status['total_index_size_kb']} KB")
    else:
        print("Index ready: no")
        print("Next step: add files to data/ and choose 'Build or update index'.")


def _write_sample_data(data_dir: Path | str = settings.DATA_DIR) -> list[str]:
    data_dir = Path(data_dir)
    samples = {
        "finance/budget_forecast_q3.txt": (
            "Quarterly budget forecast for Q3.\n"
            "Revenue outlook is stable. Expenses are rising in infrastructure and hiring.\n"
            "Action items: revise budget plan and confirm invoice schedule."
        ),
        "people/offer_letter_alex.md": (
            "Offer letter draft for Alex.\n"
            "Role: Data Analyst.\n"
            "Compensation, joining date, and onboarding checklist are included."
        ),
        "operations/vendor_runbook.txt": (
            "Vendor onboarding runbook.\n"
            "Follow the checklist, submit required forms, and confirm service access."
        ),
        "projects/mobile_search_launch.md": (
            "Project launch notes for mobile search.\n"
            "Milestones, owners, roadmap, and release checklist."
        ),
    }
    created = []
    for relative_path, content in samples.items():
        target = data_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
            created.append(relative_path)
    return created


def _run_menu() -> int:
    print("Smart File Retriever V2")
    print("Local-first search with shard routing plus hybrid semantic and keyword retrieval.\n")

    while True:
        print("Choose an option:")
        print("1. Check project status")
        print("2. Create sample files for a quick demo")
        print("3. Build or update index")
        print("4. Search files")
        print("5. Exit")
        choice = input("> ").strip()

        try:
            if choice == "1":
                status = _describe_status()
                _print_status(status)
                print()
            elif choice == "2":
                data_dir = _prompt("Data folder", str(settings.DATA_DIR))
                created = _write_sample_data(data_dir)
                if created:
                    print("\nCreated sample files:\n")
                    for relative_path in created:
                        print(f"- {relative_path}")
                    print("\nNext step: choose 'Build or update index'.\n")
                else:
                    print("\nSample files already exist. You can build the index now.\n")
            elif choice == "3":
                data_dir = _prompt("Data folder", str(settings.DATA_DIR))
                index_dir = _prompt("Index folder", str(settings.INDEX_DIR))
                force = _prompt_bool("Force full rebuild", default=False)
                manifest = build_index(data_dir, index_dir, force=force)
                print(
                    f"\nIndexed {manifest['indexed_file_count']} files "
                    f"across {len(manifest['shards'])} shards.\n"
                )
            elif choice == "4":
                query = _prompt("Search query")
                if not query:
                    print("Please enter a query.\n")
                    continue
                index_dir = _prompt("Index folder", str(settings.INDEX_DIR))
                top_k = int(_prompt("Top K results", str(settings.DEFAULT_TOP_K)))
                results = SearchEngine(index_dir=index_dir).search(query, top_k=top_k)
                print()
                _print_results(results)
                print()
            elif choice == "5":
                print("Goodbye.")
                return 0
            else:
                print("Please choose 1, 2, 3, 4, or 5.\n")
        except KeyboardInterrupt:
            print("\nOperation cancelled.\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        return _run_menu()

    parser = argparse.ArgumentParser(description="Smart File Retriever V2 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("menu", help="Launch the interactive menu")
    subparsers.add_parser("status", help="Show whether your data folder and index are ready")
    sample_parser = subparsers.add_parser("sample-data", help="Create a few demo files inside the data folder")
    sample_parser.add_argument("--data-dir", default=str(settings.DATA_DIR))

    index_parser = subparsers.add_parser("index", help="Build or update the V2 index")
    index_parser.add_argument("--data-dir", default=str(settings.DATA_DIR))
    index_parser.add_argument("--index-dir", default=str(settings.INDEX_DIR))
    index_parser.add_argument("--force", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search indexed files")
    search_parser.add_argument("query")
    search_parser.add_argument("--index-dir", default=str(settings.INDEX_DIR))
    search_parser.add_argument("--top-k", type=int, default=settings.DEFAULT_TOP_K)
    search_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "menu":
        return _run_menu()
    if args.command == "status":
        _print_status(_describe_status())
        return 0
    if args.command == "sample-data":
        created = _write_sample_data(args.data_dir)
        if created:
            print(json.dumps({"created": created, "data_dir": str(Path(args.data_dir))}, indent=2))
        else:
            print(json.dumps({"created": [], "message": "Sample files already exist.", "data_dir": str(Path(args.data_dir))}, indent=2))
        return 0

    if args.command == "index":
        manifest = build_index(
            args.data_dir,
            args.index_dir,
            force=args.force,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    results = SearchEngine(args.index_dir).search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    _print_results(results)
    return 0
