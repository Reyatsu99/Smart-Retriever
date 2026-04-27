import sys
import json
import traceback
import ast
from pathlib import Path

# Add project root to path so we can import our package
sys.path.append(str(Path(__file__).parent))

# Fallback for Python 3.14+ where ast.Num was removed
if not hasattr(ast, "Num"):
    class Num(ast.Constant):
        def __init__(self, n, **kwargs):
            super().__init__(value=n, **kwargs)
            self.n = n
    ast.Num = Num
    ast.Str = ast.Constant
    ast.NameConstant = ast.Constant
    ast.Bytes = ast.Constant

try:
    from smart_retriever_v2.evaluation import run_public_evaluation
except ImportError as e:
    print(f"Error: Could not import evaluation modules. {e}")
    sys.exit(1)

def print_metrics(dataset_name, report):
    m = report["metrics"]
    print(f"\n--- Results for {dataset_name} ---")
    print(f"nDCG@10 (Ranking Quality): {m.get('ndcg@10', 'N/A')}")
    print(f"MRR@10  (Speed to First Hit): {m.get('mrr@10', 'N/A')}")
    print(f"Recall@10 (Completeness):    {m.get('recall@10', 'N/A')}")
    print(f"Docs Indexed: {report.get('documents_indexed', 0)}")
    print(f"Queries Tested: {report.get('queries_evaluated', 0)}")

def main():
    datasets = [
        ("NFCorpus (Medical/Technical)", "beir/nfcorpus"),
        ("SciFact (Scientific Verification)", "beir/scifact")
    ]

    print("Smart File Retriever V2 - Benchmark Suite")
    print("Evaluating metrics: nDCG (Ranking), MRR (Precision), Recall (Coverage)")
    
    for label, ds_id in datasets:
        print(f"\nSearching for {label} benchmarks...")
        try:
            # We limit to 50 queries for a quick test; remove max_queries for full evaluation
            report = run_public_evaluation(
                dataset_id=ds_id,
                max_queries=50, 
                work_dir=Path(f".benchmarks/eval_{ds_id}"),
                mode="real"
            )
            print_metrics(label, report)
        except Exception as e:
            print(f"Failed to run {label}: {e}")
            traceback.print_exc()
            if "ir_datasets" in str(e):
                print("\nSuggestion: install dependencies to run benchmarks:")
                print("pip install ir_datasets ir_measures docx python-docx openpyxl")

if __name__ == "__main__":
    main()
