import os
from pathlib import Path

DOCUMENTS_PATH = Path("data/processed/documents.jsonl")

# Chunking, in characters (~4 chars per token)
EMAIL_CHUNK_CHARS = 1600
ATTACHMENT_CHUNK_CHARS = 2000
CHUNK_OVERLAP = 0.15
MIN_CHUNK_CHARS = 40

# Embeddings
EMBED_MODEL = "bge-m3"
EMBED_BATCH_SIZE = 32
EMBEDDINGS_PATH = Path("data/processed/embeddings.npz")

# Vector store
LANCEDB_PATH = Path("data/lancedb")
TABLE_NAME = "chunks"

# Retrieval
TOP_K = 8
CANDIDATES = 30
MAX_CHUNKS_PER_DOC = 2

# Generation
LLM_MODEL = "qwen3:8b"
NUM_CTX = 16384  # Ollama silently truncates prompts longer than this
TEMPERATURE = 0.2
THINK = False
# Who "I" / "we" refers to in questions, e.g. "Jane Doe <jane@example.com>, Example Corp"
OWNER = os.environ.get("MAILRAG_OWNER", "the mailbox owner")
