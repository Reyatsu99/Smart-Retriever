# Smart File Retriever V3

Smart File Retriever V3 is an advanced, fully local document search engine. It features a novel **Generative LLM-Augmented Ingestion** pipeline, running 100% locally to ensure strict enterprise data privacy while delivering state-of-the-art semantic search accuracy.

## 🚀 The Core Innovation: Forward-HyDE (LLM Ingestion Enrichment)

Traditional RAG systems perform **HyDE** (Hypothetical Document Embeddings) at *search time*, adding massive latency to every user query. 

Smart File Retriever V3 introduces **Forward-HyDE**—performing LLM enrichment at *ingestion time*:
1. **Dynamic Context Generation:** For every document ingested, the local LLM generates a concise semantic summary and 3-5 hypothetical search queries that the document is uniquely suited to answer.
2. **Pre-Chunking Enrichment:** This LLM-generated context is prepended to the document *before* chunking and vectorization.
3. **High-Fidelity Semantic Anchors:** Both the dense vector database (LanceDB) and the sparse search engine (Tantivy) index these synthetic questions, ensuring that plain-English queries match perfectly even if the original document is full of complex technical jargon—all with **zero added search-time latency**.

---

## 🛠️ Complete V3 Search Architecture

The search engine functions as a highly optimized pipeline:

1. **LLM Ingestion Enrichment:** Generates synthetic questions and summaries for each file using a local LLM via Ollama.
2. **NLP-Aware Hierarchical Chunking:** Splits the enriched text at natural structural boundaries (paragraphs, sentences) with adaptive, clean-cut word boundary overlaps to prevent context truncation.
3. **Hybrid Dense/Sparse Retrieval:** Concurrently performs a semantic vector search (LanceDB) and a full-text sparse search (BM25) on the corpus.
4. **Reciprocal Rank Fusion (RRF):** Exponentially merges both result lists based on their rank positions to yield the ultimate candidate list.
5. **Cross-Encoder Reranking:** Applies a deep cross-attention transformer (`ms-marco-MiniLM-L-6-v2`) to perform simultaneous query-document logic evaluation, delivering high-precision final rankings.

---

## ⚙️ Prerequisites

To run the LLM-augmented indexing, you must have [Ollama](https://ollama.com/) running locally with your model of choice (defaults to `phi3`):

```bash
# Start Ollama service in a separate terminal
ollama serve

# Pull the default model
ollama run phi3
```

---

## 💻 Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Menu-Driven CLI
```bash
python cli.py
```
From the interactive menu, you can:
* Check project status
* Create demo sample files
* Build/Update the LLM-augmented index
* Search your files

### 3. Run via Commands
```bash
# Generate mock demo data
python cli.py sample-data

# Build the LLM-enriched search index (forces full rebuild)
python cli.py index --force

# Search the index
python cli.py search "Alex offer letter compensation details"
```

---

## 📂 Project Layout

```text
smart_retriever_v2/
  llm.py           # Local LLM wrapper (Ollama)
  db.py            # LanceDB vector store initialization
  embeddings.py    # BGE dense vector embeddings
  indexer.py       # LLM-Augmented index builder (Forward-HyDE)
  parsers.py       # Rich parser support (.pdf, .docx, .xlsx, .txt)
  search.py        # Core search engine, CLI, & RRF
  settings.py      # App configurations
  text_utils.py    # Smart hierarchical chunking
cli.py             # CLI Entrypoint
data/              # Put your document corpus here
```

Author signature: Reyatsu99
