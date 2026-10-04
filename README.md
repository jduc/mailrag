# RAG system for Mail
This project is a homelab test project on RAG. The goal is to have a fully working RAG framework built on all the company emails to quickly find information and answer question. 


## Install and set up

Quick start (details below):

```
uv run extract.py data/EMAILS.mbox   # 1. mbox -> data/processed/documents.jsonl
brew services start ollama           # 2. local models must be running (see below for first install)
uv run index.py                      # 3. chunk + embed + store in LanceDB
uv run ask.py "your question"        # 4. ask
```

Re-run steps 1 and 3 whenever the mbox changes; step 3 only re-embeds new or changed chunks.

### Prepare the emails data
Have the Archives mbox files in the data/ directory and run the extract.py script. The script will look for a EMAILS.mbox file. For thunderbird, you can concatenate the mbox files. The script will create a json line file which contains 1 email per row. Attachments are converted to markdown and therefore can be added into the RAG. 

```
uv run extract.py data/EMAILS.mbox
```

Quoted reply history, signatures, legal disclaimers and Teams/Zoom boilerplate are stripped from email bodies.

### Local models (Ollama)
Everything runs locally: nothing leaves the machine after the one-time model download.

```
brew install ollama && brew services start ollama
ollama pull bge-m3      # embedding model (multilingual, 1024 dims)
ollama pull qwen3:8b    # answering LLM (4-bit, ~5 GB)
```

### Chunk, embed and index

```
uv run index.py
```

Chunks the documents, embeds them with bge-m3 and writes a LanceDB table (`data/lancedb/`) with a full-text index. Vectors are cached by text hash (`data/processed/embeddings.npz`), so re-running only embeds new or changed chunks. A full embedding takes ~5 min on an M1; prefix long runs with `caffeinate -i` so the Mac doesn't sleep.

## Usage

```
uv run ask.py "When did we sign the contract with Example Corp?"
uv run ask.py "workplan" --from example.com --since 2023-06-01 --until 2023-07-31
uv run ask.py "INV-2023-042" --search-only --show-chunks
uv run ask.py "..." --think        # the model reasons first: better on multi-source questions, ~3x slower
```

Answers cite their sources as `[n]`, listed below the answer. Expect ~30-40 s per answer on an M1 (mostly reading the sources).

So that "I" and "we" in questions refer to you, set who owns the mailbox:

```
export MAILRAG_OWNER="Jane Doe <jane@example.com>, Example Corp"
```

## How it works

| File | Role |
|---|---|
| `extract.py` | mbox → `data/processed/documents.jsonl` (emails + attachment text) |
| `chunking.py` | documents → chunks with a Subject/From/To/Date header; dedup of re-sent attachments |
| `embeddings.py` | bge-m3 via Ollama, with a text-hash cache |
| `index.py` | chunks + vectors → LanceDB table + BM25 index |
| `rag.py` | `search()` (vector / keyword / hybrid with RRF, filters) and `generate()` / `answer()` |
| `ask.py` | command-line interface |
| `evaluate.py` | compares search modes (Hit@1, Hit@k, MRR) on questions with known answers |
| `config.py` | all tunable settings (chunk sizes, models, top-k, `NUM_CTX`…) |

`uv run chunking.py` prints chunk statistics, useful when tuning chunk sizes.

### Evaluation

Copy `eval.example.json` to `data/eval.json` and replace the examples with questions about your own mails whose answer you know, then run `uv run evaluate.py`. `data/` is git-ignored, so these stay private.


## Future
In the future, this could be a running instance where each new mail is sent to be embedded and added in the RAG. There could also be a web interface to bridge with the prompt directly from the web server. 