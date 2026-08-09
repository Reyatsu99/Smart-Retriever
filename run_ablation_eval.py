#!/usr/bin/env python3
import sys
import json
import time
import ast
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Fallback for Python 3.14+ where ast.Num / ast.Str were deprecated/removed
if not hasattr(ast, "Num"):
    class Num(ast.Constant):
        def __init__(self, n, **kwargs):
            super().__init__(value=n, **kwargs)
            self.n = n
        @property
        def n(self):
            return self.value
    ast.Num = Num

if not hasattr(ast, "Str"):
    class Str(ast.Constant):
        @property
        def s(self):
            return self.value
    ast.Str = Str

if not hasattr(ast, "NameConstant"):
    ast.NameConstant = ast.Constant
if not hasattr(ast, "Bytes"):
    ast.Bytes = ast.Constant

from smart_retriever.evaluation import run_public_evaluation

DATASETS = [
    ("NFCorpus (Medical)", "beir/nfcorpus", "full", 100),
    ("SciFact (Scientific)", "beir/scifact", "full", 100),
    ("FiQA (Finance)", "beir/fiqa", "full", 100),
]

MODES = [
    ("vector_only", "Vector-Only (Dense BGE-small)"),
    ("fts_only", "Sparse-Only (FTS / BM25)"),
    ("hybrid", "Hybrid Fusion (Vector + FTS via RRF)"),
    ("full", "Full Pipeline (Hybrid + Cross-Encoder Reranker)")
]

def main():
    parser = argparse.ArgumentParser(description="Run BEIR Ablation Evaluation Suite")
    parser.add_argument("--max-queries", type=int, default=30, help="Number of queries to evaluate per dataset")
    parser.add_argument("--force", action="store_true", help="Force rebuild of benchmark indices")
    args = parser.parse_args()

    print("==========================================================")
    print("      SMART RETRIEVER V3 — BEIR ABLATION EVALUATION       ")
    print("==========================================================")
    print(f"Max Queries per Dataset: {args.max_queries}")
    print(f"Ablation Modes: {[m[0] for m in MODES]}\n")

    results_matrix = {}

    for ds_label, ds_id, pool_mode, pool_depth in DATASETS:
        print(f"\n==========================================")
        print(f" Dataset: {ds_label} ({ds_id})")
        print(f"==========================================")
        results_matrix[ds_id] = {"label": ds_label, "modes": {}}

        # Make sure index is built once for the dataset
        force_flag = args.force
        
        for search_mode, mode_label in MODES:
            print(f"\n--- Testing Mode: {mode_label} ({search_mode}) ---")
            try:
                report = run_public_evaluation(
                    dataset_id=ds_id,
                    max_queries=args.max_queries,
                    force=force_flag,
                    work_dir=Path(f".benchmarks/eval_{ds_id.replace('/', '_')}"),
                    mode="real",
                    search_mode=search_mode,
                    pool=pool_mode,
                    pool_depth=pool_depth,
                    keep_workspace=True
                )
                
                # We don't force rebuild for subsequent modes on the same index workspace
                force_flag = False
                
                metrics = report.get("metrics", {})
                results_matrix[ds_id]["modes"][search_mode] = {
                    "label": mode_label,
                    "ndcg@10": metrics.get("ndcg@10", 0.0),
                    "mrr@10": metrics.get("mrr@10", 0.0),
                    "recall@10": metrics.get("recall@10", 0.0),
                    "precision@10": metrics.get("precision@10", 0.0),
                    "map": metrics.get("map", 0.0),
                    "success@10": metrics.get("success@10", 0.0),
                    "avg_latency_ms": metrics.get("avg_latency_ms", 0.0)
                }

                print(f"  > nDCG@10:    {metrics.get('ndcg@10')}")
                print(f"  > MRR@10:     {metrics.get('mrr@10')}")
                print(f"  > Recall@10:  {metrics.get('recall@10')}")
                print(f"  > MAP:        {metrics.get('map')}")
                print(f"  > Latency:    {metrics.get('avg_latency_ms')} ms")

            except Exception as e:
                print(f"  ❌ Error evaluating {search_mode} on {ds_id}: {e}")

    # Output Summary Table
    out_path = Path("ablation_results.json")
    out_path.write_text(json.dumps(results_matrix, indent=2), encoding="utf-8")
    print(f"\n✅ Ablation Study Complete! Saved matrix to {out_path.resolve()}")

if __name__ == "__main__":
    main()
