from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = Path("index_v2")
TMP_INDEX_DIR = Path("index_v2_tmp")
BACKUP_INDEX_DIR = Path("index_v2_backup")
MANIFEST_BACKUP_NAME = "manifest_v2.backup.json"

MANIFEST_VERSION = 3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384

MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
MAX_FILES_PER_SHARD = 2000
TOP_SHARD_COUNT = 3
DEFAULT_TOP_K = 5
SEARCH_CANDIDATE_MULTIPLIER = 4
CLUSTER_SEED = 42

DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
FILENAME_BOOST = 0.15

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
