import lancedb
import pyarrow as pa
from pathlib import Path
from smart_retriever_v2 import settings

def get_db(db_path: Path | str | None = None) -> lancedb.DBConnection:
    """Connects to the local LanceDB instance."""
    if db_path is None:
        db_path = settings.INDEX_DIR.parent / "lancedb_store"
    
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))

def create_schema(dim: int = settings.EMBEDDING_DIM) -> pa.Schema:
    """Creates the PyArrow schema for the semantic document chunks."""
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("relative_path", pa.string()),
        pa.field("chunk_id", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("mtime", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("sha256", pa.string()),
    ])

def get_or_create_table(
    db: lancedb.DBConnection, 
    table_name: str = "document_chunks", 
    schema: pa.Schema | None = None
) -> lancedb.table.Table:
    """Gets an existing table or creates a new one defined by the schema."""
    if schema is None:
        schema = create_schema()
        
    if table_name in db.table_names():
        return db.open_table(table_name)
    return db.create_table(table_name, schema=schema)
