# Smart File Retriever V2

Smart File Retriever V2 is a local document search tool that indexes your files and lets you search them with a hybrid semantic plus keyword retriever.

This cleaned project is centered on V2:
- learned shard routing with no hand-written categories
- query-to-shard centroid search through a small FAISS routing index
- dense plus BM25 retrieval inside the top matching shards
- one simple CLI for indexing and search

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

If you prefer commands instead of the menu:

```bash
python cli.py status
python cli.py sample-data
python cli.py index
python cli.py search "budget forecast"
```

## Folder Setup

Put your documents inside the `data/` folder before indexing.

Supported file types:
- `.txt`
- `.md`
- `.csv`
- `.pdf`
- `.docx`
- `.xlsx`

If you just want to try the app quickly, run:

```bash
python cli.py sample-data
python cli.py index
python cli.py search "budget forecast"
```

## Main Commands

```bash
python cli.py
python cli.py status
python cli.py sample-data
python cli.py index
python cli.py search "offer letter"
python cli.py search "budget report" --top-k 10 --json
```

## What V2 Includes

- incremental indexing with file fingerprint reuse
- KMeans-based shard construction over the whole corpus
- shard-centroid routing through a compact FAISS index
- FAISS dense retrieval plus BM25 keyword scoring inside top shards
- a menu-driven CLI for first-time users
- demo data generation for quick testing

## Project Layout

```text
smart_retriever_v2/
  bm25.py
  embeddings.py
  indexer.py
  parsers.py
  search.py
  settings.py
  text_utils.py
  vector_index.py
cli.py
data/
scripts/
```

## Retrieval Design

V2 now uses a category-free two-level search pipeline:

1. Query vector to nearest shard centroids with a small FAISS routing index.
2. Hybrid dense plus BM25 retrieval inside the top shard candidates.

This keeps routing fast without relying on fixed domains like finance, HR, or legal.

## Submission Bundle

To build the clean submission zip for V2:

```bash
python scripts/build_submission.py
```

This creates:
- `dist/Smart_File_Retriever_V2_Submission.zip`

The generated zip is app-only and does not include benchmark or evaluation files.

Author signature: Reyatsu99
