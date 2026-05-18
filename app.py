import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any, Optional

from smart_retriever.search import SearchEngine
from smart_retriever import settings

app = FastAPI(title="Smart Retriever V3 Private API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    relative_path: str
    file_name: str
    score: float
    text: str
    matched_by: List[str]

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Warning: Could not initialize SearchEngine on startup: {e}")

@app.post("/api/search", response_model=List[SearchResult])
async def search(req: QueryRequest):
    global engine
    if engine is None:
        try:
            engine = SearchEngine()
        except Exception:
            raise HTTPException(status_code=500, detail="Search Engine not initialized. Run indexing first.")
            
    try:
        results = engine.search(req.query, top_k=req.top_k)
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found. Please index files first.")

@app.get("/api/status")
async def status():
    return {
        "status": "ready" if engine else "not_initialized",
        "data_dir": str(settings.DATA_DIR),
        "index_dir": str(settings.INDEX_DIR)
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
