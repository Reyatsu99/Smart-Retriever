from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from smart_retriever import settings
from smart_retriever.db import get_db
from smart_retriever.embeddings import EmbeddingBackend
from smart_retriever.indexer import build_index


class SearchEngine:
    def __init__(self, index_dir: Path | str = settings.INDEX_DIR, embedder: EmbeddingBackend | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder or EmbeddingBackend()
        self.db = get_db(self.index_dir.parent / "lancedb_store")
        
        try:
            from sentence_transformers import CrossEncoder
            # We use a fast cross-encoder to prevent timeouts but boost precision significantly
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        except Exception:
            self.reranker = None

    def search(self, query: str, top_k: int = settings.DEFAULT_TOP_K) -> list[dict[str, Any]]:
        table_name = "document_chunks"
        if table_name not in self.db.table_names():
            raise FileNotFoundError("Index not found. Run `python cli.py index` first.")
            
        table = self.db.open_table(table_name)
        if table.count_rows() == 0:
            return []
            
        query_vector = self.embedder.encode_document(query).tolist()
        
        # Stage 1: Hybrid Retrieval (Vector + Keyword)
        pool_size = max(100, top_k * 10)
        
        # A. Vector Search
        vector_hits = table.search(query_vector).limit(pool_size).to_arrow().to_pylist()
        
        # B. Keyword Search (FTS)
        fts_hits = []
        try:
            fts_hits = table.search(query, query_type="fts").limit(pool_size).to_arrow().to_pylist()
        except Exception as exc:
            # FTS might not be initialized or query too short
            pass

        # C. Reciprocal Rank Fusion (RRF) to merge results
        rrf_scores: dict[str, float] = {}
        # k is the constant used in RRF formula (standard is 60)
        RRF_K = 60
        
        hits_by_id: dict[str, dict[str, Any]] = {}
        
        # Function to add scores from a ranked list
        def add_rrf_scores(ranked_list, source_name):
            for rank, hit in enumerate(ranked_list):
                # Use relative_path + chunk_id as unique key for chunks
                hit_id = f"{hit['relative_path']}_{hit['chunk_id']}"
                hits_by_id[hit_id] = hit
                rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + 1.0 / (RRF_K + rank + 1)
                hit.setdefault("matched_by", []).append(source_name)

        add_rrf_scores(vector_hits, "vector")
        add_rrf_scores(fts_hits, "keyword")
        
        # Get sorted hit IDs by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:pool_size]
        hits = [hits_by_id[hid] for hid in sorted_ids]
        
        if not hits:
            return []
            
        # Stage 2: Semantic Reranking (Cross-Encoder)
        if self.reranker:
            pairs = [[query, hit["text"]] for hit in hits]
            scores = self.reranker.predict(pairs)
            for hit, score in zip(hits, scores):
                hit["score"] = float(score)
                hit["matched_by"].append("semantic_reranker")
                hit["matched_by"] = sorted(list(set(hit["matched_by"])))
        else:
            # Fallback to normalized RRF score
            for hit_id, score in rrf_scores.items():
                hit = hits_by_id[hit_id]
                hit["score"] = score
                hit["matched_by"] = sorted(list(set(hit["matched_by"])))

        # Aggregate top chunks back to Document level
        merged: dict[str, dict[str, Any]] = {}
        for hit in hits:
            rp = hit["relative_path"]
            if rp not in merged or hit["score"] > merged[rp]["score"]:
                merged[rp] = {
                    "matched_by": hit["matched_by"],
                    "relative_path": rp,
                    "file_name": Path(rp).name,
                    "path": str((settings.DATA_DIR / rp).resolve()),
                    "score": hit["score"],
                    "text": hit["text"],
                    "shard": "v3_lancedb_chunk_" + str(hit["chunk_id"]),
                    "mtime": hit["mtime"],
                    "size": hit["size"],
                    "sha256": hit["sha256"]
                }
                
        final_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return final_results



def _resolve_result_path(result: dict[str, Any]) -> str:
    """Returns the absolute file path for a result, reconstructing it from
    relative_path + DATA_DIR if the 'path' field wasn't already populated."""
    if result.get("path"):
        return result["path"]
    return str((settings.DATA_DIR / result["relative_path"]).resolve())


def _file_uri(path_str: str) -> str:
    """Builds a proper file:// URI from an absolute path string."""
    try:
        return Path(path_str).resolve().as_uri()
    except (ValueError, OSError):
        return f"file://{path_str}"


def _hyperlink(uri: str, label: str) -> str:
    """Wraps label in an OSC 8 terminal hyperlink escape sequence so supporting
    terminals (iTerm2, GNOME Terminal, Windows Terminal, VS Code, etc.) render
    it as clickable. Falls back to plain label when stdout isn't a tty (piped
    output, redirected to a file) or NO_COLOR is set, so output stays clean."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return label
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"


def _format_file_link(path_str: str) -> tuple[str, str]:
    """Returns (plain_path, clickable_link) for displaying a file location."""
    uri = _file_uri(path_str)
    return path_str, _hyperlink(uri, uri)


def _print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No matching files found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for rank, result in enumerate(results, start=1):
        plain_path, link = _format_file_link(_resolve_result_path(result))
        print(
            f"{rank}. {result['relative_path']}\n"
            f"   score: {result['score']:.4f}\n"
            f"   shard: {result['shard']}\n"
            f"   matched by: {', '.join(result['matched_by']) if result['matched_by'] else 'routing only'}\n"
            f"   path: {plain_path}\n"
            f"   link: {link}"
        )


def _print_audit_reports(reports: list[dict[str, Any]], header: str) -> None:
    print(f"\n{header}\n" + "=" * 40)
    for r in reports:
        audit_data = r["audit"]
        is_authentic = audit_data.get("is_authentic", True)
        status_prefix = "[VERIFIED]" if is_authentic else "[!!! BOGUS / SPOOFED !!!]"
        plain_path, link = _format_file_link(_resolve_result_path(r))

        print(f"\nFILE: {r['file_name']} {status_prefix}")
        print(f"  path: {plain_path}")
        print(f"  link: {link}")
        if not is_authentic:
            print(f"  Warning: {audit_data.get('authenticity_reason')}")

        if "requirements" in audit_data:
            for req in audit_data["requirements"]:
                status = req.get("status", "UNKNOWN")
                print(f"- [{status}] {req.get('name')}")
                if req.get("reason"):
                    print(f"  Reason: {req['reason']}")
        print(f"Summary: {audit_data.get('overall_summary', 'N/A')}")
    print("=" * 40 + "\n")


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
        print("5. Audit documents (Manual Verification)")
        print("6. Auto-Verify documents (Autonomous)")
        print("7. Exit")
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
                query = _prompt("Search query for audit")
                req_str = _prompt("Enter requirements (comma-separated)")
                requirements = [r.strip() for r in req_str.split(",") if r.strip()]
                index_dir = _prompt("Index folder", str(settings.INDEX_DIR))
                
                engine = SearchEngine(index_dir=index_dir)
                from smart_retriever.auditor import DocumentAuditor
                auditor = DocumentAuditor(engine)
                reports = auditor.audit(query, requirements)
                _print_audit_reports(reports, "AUDIT RESULTS:")
            elif choice == "6":
                query = _prompt("Search query for autonomous verification")
                index_dir = _prompt("Index folder", str(settings.INDEX_DIR))
                
                engine = SearchEngine(index_dir=index_dir)
                from smart_retriever.auditor import DocumentAuditor
                auditor = DocumentAuditor(engine)
                reports = auditor.auto_verify(query)
                _print_audit_reports(reports, "AUTO-VERIFICATION RESULTS:")
            elif choice == "7":
                print("Goodbye.")
                return 0
            else:
                print("Please choose 1, 2, 3, 4, 5, 6, or 7.\n")
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

    audit_parser = subparsers.add_parser("audit", help="Audit documents against requirements")
    audit_parser.add_argument("query")
    audit_parser.add_argument("--reqs", help="Comma-separated list of requirements (optional, triggers auto-mode if omitted)")
    audit_parser.add_argument("--index-dir", default=str(settings.INDEX_DIR))
    audit_parser.add_argument("--top-k", type=int, default=3)

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

    if args.command == "search":
        results = SearchEngine(args.index_dir).search(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_results(results)
        return 0

    if args.command == "audit":
        engine = SearchEngine(args.index_dir)
        from smart_retriever.auditor import DocumentAuditor
        auditor = DocumentAuditor(engine)
        
        if args.reqs:
            requirements = [r.strip() for r in args.reqs.split(",") if r.strip()]
            reports = auditor.audit(args.query, requirements, top_k=args.top_k)
        else:
            reports = auditor.auto_verify(args.query, top_k=args.top_k)
            
        print(json.dumps(reports, indent=2))
        return 0

    return 0
