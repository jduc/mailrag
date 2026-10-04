"""Retrieval and answer generation over the LanceDB table."""

from collections.abc import Iterator

import lancedb
import ollama
from lancedb.rerankers import RRFReranker

import config
from embeddings import embed

MODES = ("vector", "fts", "hybrid")
_table = None


def get_table():
    global _table
    if _table is None:
        _table = lancedb.connect(config.LANCEDB_PATH).open_table(config.TABLE_NAME)
    return _table


def sql_str(value: str) -> str:
    return value.replace("'", "''")


def build_filter(since: str | None = None, until: str | None = None, sender: str | None = None) -> str | None:
    """WHERE clause from optional filters: dates as YYYY-MM-DD, sender as a name/email fragment."""
    clauses = []
    if since:
        clauses.append(f"date >= timestamp '{sql_str(since)} 00:00:00'")
    if until:
        clauses.append(f"date <= timestamp '{sql_str(until)} 23:59:59'")
    if sender:
        s = sql_str(sender.lower())
        clauses.append(f"(lower(from_email) LIKE '%{s}%' OR lower(from_name) LIKE '%{s}%')")
    return " AND ".join(clauses) or None


def search(
    query: str,
    k: int = config.TOP_K,
    mode: str = "hybrid",
    since: str | None = None,
    until: str | None = None,
    sender: str | None = None,
) -> list[dict]:
    """Return the k best chunks, at most MAX_CHUNKS_PER_DOC per document. Each has a
    higher-is-better `score`: cosine similarity, BM25 or RRF depending on the mode."""
    table = get_table()
    if mode == "vector":
        q = table.search(embed([query])[0]).distance_type("cosine")
    elif mode == "fts":
        q = table.search(query, query_type="fts")
    elif mode == "hybrid":
        q = table.search(query_type="hybrid").vector(embed([query])[0]).text(query).rerank(RRFReranker())
    else:
        raise ValueError(f"mode must be one of {MODES}")

    where = build_filter(since, until, sender)
    if where:
        q = q.where(where, prefilter=True)
    rows = q.limit(config.CANDIDATES).to_list()

    results, per_doc = [], {}
    for row in rows:
        if per_doc.get(row["doc_id"], 0) >= config.MAX_CHUNKS_PER_DOC:
            continue
        per_doc[row["doc_id"]] = per_doc.get(row["doc_id"], 0) + 1
        row.pop("vector", None)
        if "_distance" in row:
            row["score"] = 1 - row["_distance"]
        else:
            row["score"] = row.get("_relevance_score", row.get("_score"))
        results.append(row)
        if len(results) == k:
            break
    return results


SYSTEM_PROMPT = f"""You answer questions about the email archive of {config.OWNER}.
"I", "me", "we" and "our" in questions refer to them.

Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite the sources supporting each statement inline, like [1] or [2][4].
- If the sources do not contain the answer, say you could not find it in the emails. Never guess.
- Mention dates and people when they matter.
- Answer in the language of the question. Be concise."""


def source_label(r: dict) -> str:
    label = f"{r['date']:%Y-%m-%d} | From: {r['from_name']} <{r['from_email']}> | Subject: {r['subject']}"
    if r["doc_type"] == "attachment":
        label += f" | Attachment: {r['filename']} (part {r['chunk_index'] + 1}/{r['n_chunks']})"
    return label


def build_messages(question: str, sources: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{i}] {source_label(r)}\n{r['text']}" for i, r in enumerate(sources, 1))
    # Question after the sources: models attend most to the end of the prompt
    user = f"Sources:\n\n{context}\n\n---\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def estimate_tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / 3.5)


def generate(
    question: str, sources: list[dict], num_ctx: int = config.NUM_CTX, think: bool = config.THINK
) -> Iterator:
    """Stream the answer as Ollama chunks; the last one (chunk.done) carries token counts."""
    return ollama.chat(
        model=config.LLM_MODEL,
        messages=build_messages(question, sources),
        stream=True,
        think=think,
        options={"num_ctx": num_ctx, "temperature": config.TEMPERATURE},
    )


def answer(question: str, **search_args) -> tuple[list[dict], Iterator]:
    sources = search(question, **search_args)
    return sources, generate(question, sources)
