import sys
import traceback
import ast
import argparse
from pathlib import Path

# Add project root to path so we can import our package
sys.path.append(str(Path(__file__).parent))

# Fallback for Python 3.14+ where ast.Num was removed
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

# Map other missing names
if not hasattr(ast, "NameConstant"):
    ast.NameConstant = ast.Constant
if not hasattr(ast, "Bytes"):
    ast.Bytes = ast.Constant

try:
    from smart_retriever_v2.evaluation import run_public_evaluation
except ImportError as e:
    print(f"Error: Could not import evaluation modules. {e}")
    sys.exit(1)

def print_metrics(dataset_name, report):
    m = report["metrics"]
    print(f"\n--- Results for {dataset_name} ---")
    print(f"nDCG@10 (Ranking):    {m.get('ndcg@10', 'N/A')}")
    print(f"MRR@10  (Precision):  {m.get('mrr@10', 'N/A')}")
    print(f"Recall@10 (Recall):   {m.get('recall@10', 'N/A')}")
    print(f"Precision@10 (P@10):  {m.get('precision@10', 'N/A')}")
    print(f"MAP (Global Precision): {m.get('map', 'N/A')}")
    print(f"Success@10 (Hit Rate): {m.get('success@10', 'N/A')}")
    print(f"Avg Latency (Speed):   {m.get('avg_latency_ms', 'N/A')} ms/query")
    print(f"Docs Indexed: {report.get('documents_indexed', 0)}")
    print(f"Queries Tested: {report.get('queries_evaluated', 0)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-queries", type=int, default=50)
    args = parser.parse_args()

    datasets = [
        ("NFCorpus (Medical)", "beir/nfcorpus", "full", 100),
        ("SciFact (Scientific)", "beir/scifact", "full", 100),
        ("FiQA (Finance)", "beir/fiqa", "full", 100),
        ("DBPedia (General)", "beir/dbpedia-entity", "scoreddocs", 100),
        ("Quora (QA/Duplicates)", "beir/quora", "scoreddocs", 100)
    ]

    print("Smart File Retriever V3 - Multi-Domain 3D Benchmark Suite")
    print("Evaluating: Quality (nDCG/MAP), Speed (Latency), and Coverage (Recall)")
    
    for label, ds_id, pool_mode, pool_depth in datasets:
        print(f"\nSearching for {label} benchmarks...")
        try:
            report = run_public_evaluation(
                dataset_id=ds_id,
                max_queries=args.max_queries, 
                force=args.force,
                work_dir=Path(f".benchmarks/eval_{ds_id.replace('/', '_')}"),
                mode="real",
                pool=pool_mode,
                pool_depth=pool_depth,
                keep_workspace=True
            )
            print_metrics(label, report)
        except Exception as e:
            print(f"Failed to run {label}: {e}")
            # traceback.print_exc()

if __name__ == "__main__":
    main()
