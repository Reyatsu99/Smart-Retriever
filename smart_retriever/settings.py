from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = Path("index")

MANIFEST_VERSION = 4
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
DEFAULT_TOP_K = 5
DEFAULT_RERANK_DEPTH = 25

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}

SEMANTIC_ALIASES = {
    "salary": {"compensation", "payroll", "pay"},
    "revenue": {"sales", "income"},
    "contract": {"agreement"},
    "offer": {"joining", "employment"},
    "resume": {"cv", "candidate"},
    "budget": {"forecast", "plan"},
    "report": {"summary", "results"},
}
