# Smart File Retriever V3

Smart File Retriever is an advanced, fully local document search and verification engine. It evolves traditional Information Retrieval (IR) into an autonomous logical agent, running 100% locally to ensure strict enterprise data privacy.

V3 introduces a powerful four-phase architecture:
1. **Phase 1: Ingestion & Smart Chunking** - NLP-aware hierarchical chunking using BGE embeddings and LanceDB.
2. **Phase 2: Hybrid Search & Fusion** - Simultaneous semantic vector and sparse keyword search merged via Reciprocal Rank Fusion (RRF).
3. **Phase 3: Cross-Encoder Reranking** - Deep cross-attention transformer model to refine the top results based on logical linguistic nuance.
4. **Phase 4: Autonomous Auditor & Authenticity Guard** - Local LLM integration (Ollama) to autonomously verify documents against query constraints and flag adversarial/spoofed files.

## Prerequisites

To use the **Phase 4 Autonomous Auditor**, you must have [Ollama](https://ollama.com/) installed and running locally with the `phi3` model (or update the model name in the code):
```bash
# Start Ollama server in a separate terminal
ollama serve

# Pull the default model
ollama run phi3
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python cli.py
```

The easiest first run is:
1. Run `python cli.py`
2. Choose `Check project status`
3. Choose `Create sample files for a quick demo` if your `data/` folder is empty
4. Choose `Build or update index`
5. Choose `Search files`
6. Choose `Auto-Verify documents (Autonomous)` to see the LLM auditor in action!

If you prefer commands instead of the menu:
```bash
python cli.py status
python cli.py sample-data
python cli.py index
python cli.py search "budget forecast"
```

## Advanced Commands (Auditor & Anti-Spoofing)

Use the CLI to run zero-shot compliance checks on your local documents:

```bash
# Autonomous extraction of requirements & verification
python cli.py audit "Find the offer letter for Alex with salary details"

# Manual requirement specification
python cli.py audit "Find Alex's offer" --reqs "Must be a policy document, Must be dated 2024, Must contain salary"
```

## Folder Setup

Put your documents inside the `data/` folder before indexing.

Supported file types:
- `.txt`, `.md`, `.csv`, `.pdf`, `.docx`, `.xlsx`

## Project Layout

```text
smart_retriever_v2/
  auditor.py       # Phase 4: Autonomous Auditor & Authenticity Guard
  llm.py           # Local LLM integration (Ollama)
  db.py            # LanceDB vector database management
  embeddings.py    # Dense vector encoding (BGE)
  indexer.py       # Incremental indexing pipeline
  parsers.py       # File parsing (PDF, Word, etc.)
  search.py        # Core search engine & CLI (Phase 2 & 3)
  settings.py
  text_utils.py    # Phase 1: Smart NLP-Aware Chunking
cli.py             # Main entrypoint
v3_adversarial_test.py # Test suite for anti-spoofing
data/
```

## Submission Bundle

To build the clean submission zip for V3:

```bash
python scripts/build_submission.py
```
*(Ensure `scripts/build_submission.py` is updated for V3 targets.)*

Author signature: Reyatsu99
