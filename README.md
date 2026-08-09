# Smart File Retriever 🚀

Smart File Retriever is an advanced, production-grade local document retrieval, audit, and verification engine. It evolves traditional Information Retrieval (IR) into an autonomous, privacy-focused search and compliance platform operating 100% locally.

The system features a hybrid, multi-stage retrieval and audit architecture:
1. **Phase 1: Ingestion & Smart Chunking** – NLP-aware hierarchical chunking stored in LanceDB with vector and full-text search indices.
2. **Phase 2: Hybrid Search & RRF Fusion** – Dense vector search (BAAI/bge-small-en-v1.5) combined with sparse keyword search (LanceDB Tantivy FTS with semantic alias expansion), fused via Reciprocal Rank Fusion (RRF).
3. **Phase 3: Cross-Encoder Reranking** – Deep cross-attention reranker (`ms-marco-MiniLM-L-6-v2`) with configurable rerank depth pruning to optimize query latency.
4. **Phase 4: Autonomous Auditor & Authenticity Guard** – Local LLM integration (Ollama / `phi3`) to autonomously infer compliance requirements, verify document contents, and detect adversarial or spoofed files.
5. **Phase 5: Web Dashboard & Enterprise API** – Interactive modern web dashboard paired with a FastAPI REST server for live analytics, search mode execution, and index management.

---

## ⚡ Prerequisites

To use the **Phase 4 Autonomous Auditor**, run [Ollama](https://ollama.com/) locally:
```bash
# Start Ollama server in a separate terminal
ollama serve

# Pull the default model
ollama run phi3
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Web UI & FastAPI Server (Recommended)
Start the enterprise server and open the web dashboard:
```bash
python app.py
# Server running at http://127.0.0.1:8000
```
Open `http://127.0.0.1:8000` in your web browser to access:
- **Interactive Search Interface**: Toggle between `Full Pipeline`, `Hybrid RRF`, `Dense Vector Only`, and `Sparse FTS Only`.
- **Compliance Auditor**: Run zero-shot document verification with auto-requirement inference.
- **System Analytics**: View dataset size, chunk distribution, indexing status, and benchmark performance metrics.
- **Document Management**: Trigger index rebuilds and inspect loaded document chunks.

### 3. CLI Mode
Alternatively, use the interactive terminal interface:
```bash
python cli.py
```
Or run direct commands:
```bash
python cli.py status
python cli.py sample-data
python cli.py index
python cli.py search "budget forecast"
python cli.py audit "Find the offer letter for Alex with salary details"
```

---

## 📊 Ablation & Performance Evaluation

Evaluate retrieval performance (MRR, Hit@K, Latency) across different retrieval strategies using the evaluation harness:

```bash
python run_ablation_eval.py
```
This benchmarks `full`, `hybrid`, `vector_only`, and `fts_only` modes and exports the results to `ablation_results.json` for live dashboard visualization.

---

## 🔌 REST API Endpoints

The FastAPI server provides the following endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Serves the HTML5 Web Dashboard |
| `GET /api/status` | `GET` | System health, dataset statistics, and embedding status |
| `POST /api/search` | `POST` | Execute search (`query`, `top_k`, `search_mode`, `rerank_depth`) |
| `POST /api/audit` | `POST` | Perform LLM verification (`query`, `requirements`, `auto_infer`) |
| `GET /api/documents` | `GET` | List tracked files and chunk distribution |
| `POST /api/index` | `POST` | Trigger index update or forced rebuild task |
| `GET /api/analytics` | `GET` | Retrieve architectural overview and ablation benchmark data |

---

## 📁 Supported File Formats

Place documents inside the `data/` directory:
- Text & Docs: `.txt`, `.md`, `.pdf`, `.docx`
- Structured Data: `.csv`, `.xlsx`

---

## 🏗️ Project Structure

```text
smart_retriever/
  auditor.py          # Phase 4: Autonomous Auditor & Anti-Spoofing Guard
  db.py               # LanceDB database & index management
  embeddings.py       # Dense vector encoder (BGE-small-en)
  indexer.py          # Incremental NLP chunking & indexing pipeline
  llm.py              # Local LLM integration (Ollama / Phi-3)
  parsers.py          # Multi-format file parsers (PDF, DOCX, CSV, etc.)
  search.py           # Multi-mode search engine & reranker (Phases 2 & 3)
  settings.py         # Global configuration & default hyperparameters
  text_utils.py       # Semantic alias expansion & NLP tokenization
app.py                # FastAPI REST server & entrypoint
cli.py                # Terminal interactive & command interface
run_ablation_eval.py  # Ablation evaluation benchmark harness
adversarial_test.py    # Adversarial robustness test suite
frontend/
  index.html          # Modern Web Dashboard (CSS3 / JS / Responsive)
data/                 # Document storage directory
```

---

## 🛡️ Anti-Spoofing & Robustness Tests

Validate document authenticity verification against prompt injections or corrupted content:
```bash
python adversarial_test.py
```


