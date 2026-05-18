
import sys
from pathlib import Path
import ast
import json

# Fallback for Python 3.14+
if not hasattr(ast, "Num"):
    class Num(ast.Constant):
        @property
        def n(self): return self.value
    ast.Num = Num
if not hasattr(ast, "Str"):
    class Str(ast.Constant):
        @property
        def s(self): return self.value
    ast.Str = Str

sys.path.append(".")
from smart_retriever.evaluation import run_public_evaluation

print("Starting Evaluation Test Run (5 queries)...")
report = run_public_evaluation(
    dataset_id="beir/nfcorpus",
    max_queries=5, 
    work_dir=Path(".benchmarks/eval_test"),
    mode="real"
)
print("Finished!")
print(json.dumps(report["metrics"], indent=2))
