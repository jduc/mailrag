"""Chunk, embed and store the documents in LanceDB with a full-text index.

    uv run index.py

Only new or changed chunks are embedded; the rest come from the cache.
"""

import time

import lancedb
import pandas as pd
from lancedb.index import FTS

import config
from chunking import build_chunks, load_documents
from embeddings import embed_cached


def build_table() -> lancedb.table.Table:
    start = time.perf_counter()
    chunks, stats = build_chunks(load_documents())
    print(f"chunks: {len(chunks)} ({stats['duplicates']} duplicates removed)")

    vectors, new = embed_cached([c["embed_text"] for c in chunks], progress=True)
    print(f"vectors: {len(vectors)} ({new} newly embedded)")

    df = pd.DataFrame(chunks)
    # Some Date headers have no timezone; those are read as UTC
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    df["vector"] = list(vectors)

    table = lancedb.connect(config.LANCEDB_PATH).create_table(config.TABLE_NAME, df, mode="overwrite")
    table.create_index("embed_text", config=FTS(stem=True, ascii_folding=True), replace=True)
    # No ANN vector index: an exact scan is fast enough below ~100k rows

    print(f"table '{config.TABLE_NAME}' written to {config.LANCEDB_PATH} in {time.perf_counter() - start:.1f}s")
    return table


if __name__ == "__main__":
    build_table()
