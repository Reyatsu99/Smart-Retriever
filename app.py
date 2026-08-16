import os
import time
import uvicorn
from pathlib import Path
from typing import List, Any, Optional, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from smart_retriever.search import SearchEngine
from smart_retriever.auditor import DocumentAuditor
from smart_retriever.indexer import build_index
from smart_retriever.db import get_db
from smart_retriever import settings

app = FastAPI(title="Smart Retriever V3 Enterprise Server", version="3.0.0")

class SearchQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    search_mode: str = "full"
    rerank_depth: int = 25

class AuditRequest(BaseModel):
    query: str
    requirements: Optional[List[str]] = None
    top_k: int = 3
    auto_infer: bool = True

class IndexRequest(BaseModel):
    data_dir: Optional[str] = None
    force_rebuild: bool = False

engine: Optional[SearchEngine] = None
auditor: Optional[DocumentAuditor] = None
indexing_in_progress: bool = False
last_index_time: Optional[float] = None

def get_search_engine() -> SearchEngine:
    global engine
    if engine is None:
        try:
            engine = SearchEngine()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search Engine initialization failed: {e}. Run indexing first.")
    return engine

def get_auditor() -> DocumentAuditor:
    global auditor
    se = get_search_engine()
    if auditor is None:
        auditor = DocumentAuditor(search_engine=se)
    return auditor

@app.on_event("startup")
async def startup_event():
    global engine, last_index_time
    try:
        engine = SearchEngine()
        last_index_time = time.time()
        print("✅ Smart Retriever V3 Search Engine Initialized!")
    except Exception as e:
        print(f"⚠️ Notice: SearchEngine deferred initialization until index is built ({e}).")

@app.get("/api/status")
async def status():
    global engine, indexing_in_progress, last_index_time
    table_count = 0
    chunk_count = 0
    if engine:
        try:
            if "document_chunks" in engine.db.table_names():
                tbl = engine.db.open_table("document_chunks")
                chunk_count = tbl.count_rows()
                table_count = len(engine.db.table_names())
        except Exception:
            pass

    return {
        "status": "ready" if engine else "uninitialized",
        "indexing": indexing_in_progress,
        "chunk_count": chunk_count,
        "table_count": table_count,
        "data_dir": str(settings.DATA_DIR.resolve()),
        "index_dir": str(settings.INDEX_DIR.resolve()),
        "manifest_version": settings.MANIFEST_VERSION,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "last_indexed": last_index_time
    }

@app.post("/api/search")
async def search(req: SearchQueryRequest):
    se = get_search_engine()
    start_time = time.perf_counter()
    try:
        results = se.search(
            query=req.query,
            top_k=req.top_k,
            search_mode=req.search_mode,
            rerank_depth=req.rerank_depth
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "query": req.query,
            "search_mode": req.search_mode,
            "latency_ms": elapsed_ms,
            "results_count": len(results),
            "results": results
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index table not found. Run indexing first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit")
async def audit_documents(req: AuditRequest):
    doc_auditor = get_auditor()
    start_time = time.perf_counter()
    try:
        if req.auto_infer or not req.requirements:
            reports = doc_auditor.auto_verify(req.query, top_k=req.top_k)
        else:
            reports = doc_auditor.audit(req.query, req.requirements, top_k=req.top_k)
        
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "query": req.query,
            "latency_ms": elapsed_ms,
            "reports": reports
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit failed: {e}")

@app.get("/api/documents")
async def list_documents():
    se = get_search_engine()
    try:
        if "document_chunks" not in se.db.table_names():
            return {"documents": [], "total_chunks": 0}
        
        tbl = se.db.open_table("document_chunks")
        
        # Optimize memory usage: don't load the vectors or text.
        # Use select to only retrieve necessary columns and set a large limit.
        arrow_table = tbl.search().limit(1000000).select(["relative_path", "size", "mtime"]).to_arrow()
        df = arrow_table.to_pydict()
        
        docs_map: Dict[str, Dict[str, Any]] = {}
        for rp, sz, mt in zip(
            df.get("relative_path", []),
            df.get("size", []),
            df.get("mtime", [])
        ):
            if rp not in docs_map:
                docs_map[rp] = {
                    "relative_path": rp,
                    "file_name": Path(rp).name,
                    "size_bytes": sz,
                    "mtime": mt,
                    "chunk_count": 0
                }
            docs_map[rp]["chunk_count"] += 1
            
        return {
            "documents": list(docs_map.values()),
            "total_documents": len(docs_map),
            "total_chunks": len(df.get("relative_path", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_indexing_task(target_dir: Path, force: bool):
    global engine, indexing_in_progress, last_index_time
    indexing_in_progress = True
    try:
        build_index(data_dir=target_dir, force=force)
        engine = SearchEngine()
        last_index_time = time.time()
        print("✅ Indexing completed successfully!")
    finally:
        indexing_in_progress = False

@app.post("/api/index")
async def trigger_index(req: IndexRequest, background_tasks: BackgroundTasks):
    global indexing_in_progress
    if indexing_in_progress:
        return {"status": "in_progress", "message": "Indexing is already running in background."}
    
    target = Path(req.data_dir) if req.data_dir else settings.DATA_DIR
    background_tasks.add_task(run_indexing_task, target, req.force_rebuild)
    return {"status": "started", "message": f"Indexing started for {target}"}

@app.get("/api/analytics")
async def analytics():
    results_path = Path("ablation_results.json")
    ablation_data = {}
    if results_path.exists():
        try:
            import json
            ablation_data = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    return {
        "benchmarks": ablation_data,
        "architecture": {
            "dense_model": settings.EMBEDDING_MODEL_NAME,
            "sparse_engine": "LanceDB Tantivy BM25",
            "reranker": "ms-marco-MiniLM-L-6-v2",
            "fusion": "Reciprocal Rank Fusion (k=60)"
        }
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = Path("frontend/index.html")
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Smart Retriever Server Running. Frontend not found.</h1>"

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
