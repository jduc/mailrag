"""Split documents into chunks. `uv run chunking.py` prints chunk statistics."""

import hashlib
import json
import re
import statistics
from collections import Counter

import config

SEPARATORS = ["\n\n", "\n", ". ", " "]  # preferred cut points, coarsest first
TABLE_SEPARATOR_ROW = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def split_recursive(text: str, max_chars: int, separators: list[str] = SEPARATORS) -> list[str]:
    """Cut text into pieces <= max_chars with the coarsest separator that works.
    Separators stay attached to the preceding piece, so "".join(pieces) == text."""
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    sep, finer = separators[0], separators[1:]
    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep
        pieces.extend(split_recursive(part, max_chars, finer))
    return pieces


def split_table(lines: list[str], max_chars: int) -> list[str]:
    """Split a markdown table between rows, repeating the header in every piece."""
    has_header = len(lines) > 1 and TABLE_SEPARATOR_ROW.match(lines[1])
    header = "\n".join(lines[:2]) + "\n" if has_header else ""
    rows = lines[2:] if has_header else lines
    pieces, current = [], header
    for row in rows:
        row = row[: max_chars - len(header) - 1]
        if len(current) + len(row) + 1 > max_chars and current != header:
            pieces.append(current)
            current = header
        current += row + "\n"
    if current != header:
        pieces.append(current)
    return pieces


def split_blocks(text: str, max_chars: int) -> list[str]:
    pieces, prose, table = [], [], []

    def flush_prose():
        if prose:
            pieces.extend(split_recursive("\n".join(prose) + "\n", max_chars))
            prose.clear()

    def flush_table():
        if table:
            pieces.extend(split_table(table, max_chars))
            table.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            flush_prose()
            table.append(line)
        else:
            flush_table()
            prose.append(line)
    flush_prose()
    flush_table()
    return pieces


def merge_pieces(pieces: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Pack pieces into chunks <= max_chars; the tail of each chunk is repeated at the start of the next."""
    chunks, current = [], []
    for piece in pieces:
        if current and sum(map(len, current)) + len(piece) > max_chars:
            chunks.append("".join(current))
            tail = []
            for prev in reversed(current):
                if sum(map(len, tail)) + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
            current = tail
            while current and sum(map(len, current)) + len(piece) > max_chars:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, max_chars: int) -> list[str]:
    pieces = split_blocks(text, max_chars)
    return merge_pieces(pieces, max_chars, int(max_chars * config.CHUNK_OVERLAP))


def format_people(people: list[dict], limit: int = 3) -> str:
    names = [p["name"] or p["email"] for p in people[:limit]]
    extra = f" +{len(people) - limit}" if len(people) > limit else ""
    return ", ".join(names) + extra


def context_header(doc: dict) -> str:
    """Prepended to each chunk before embedding, so the vector carries subject, people and date."""
    sender = doc.get("from") or {}
    lines = [
        f"Subject: {doc['subject']}",
        f"From: {sender.get('name', '')} <{sender.get('email', '')}>",
    ]
    if doc.get("to"):
        lines.append(f"To: {format_people(doc['to'])}")
    if doc.get("date"):
        lines.append(f"Date: {doc['date'][:10]}")
    if doc["doc_type"] == "attachment":
        lines.append(f"Attachment: {doc['filename']}")
    return "\n".join(lines)


def chunk_document(doc: dict) -> list[dict]:
    max_chars = config.EMAIL_CHUNK_CHARS if doc["doc_type"] == "email" else config.ATTACHMENT_CHUNK_CHARS
    texts = [t for t in split_text(doc["text"], max_chars) if len(t) >= config.MIN_CHUNK_CHARS]
    header = context_header(doc)
    sender = doc.get("from") or {}
    return [
        {
            "chunk_id": f"{doc['id']}-{i}",
            "doc_id": doc["id"],
            "doc_type": doc["doc_type"],
            "chunk_index": i,
            "n_chunks": len(texts),
            "message_id": doc["message_id"],
            "thread_id": doc["thread_id"],
            "date": doc["date"],
            "subject": doc["subject"],
            "from_name": sender.get("name", ""),
            "from_email": sender.get("email", ""),
            "to": ", ".join(p["email"] for p in doc.get("to", [])),
            "filename": doc.get("filename", ""),
            "text": text,
            "embed_text": f"{header}\n\n{text}",
        }
        for i, text in enumerate(texts)
    ]


def build_chunks(docs: list[dict]) -> tuple[list[dict], dict]:
    """Chunk all documents, oldest first. Identical chunks (e.g. re-sent attachments) are kept once,
    with the other message ids in `also_in`."""
    stats = Counter()
    by_hash: dict[str, dict] = {}
    for doc in sorted(docs, key=lambda d: d["date"] or ""):
        for chunk in chunk_document(doc):
            stats["chunks_total"] += 1
            key = hashlib.sha1(" ".join(chunk["text"].split()).encode()).hexdigest()
            if key in by_hash:
                stats["duplicates"] += 1
                by_hash[key].setdefault("also_in", []).append(chunk["message_id"])
                continue
            by_hash[key] = chunk
    chunks = list(by_hash.values())
    for chunk in chunks:
        chunk.setdefault("also_in", [])
    return chunks, dict(stats)


def load_documents() -> list[dict]:
    with config.DOCUMENTS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    docs = load_documents()
    chunks, stats = build_chunks(docs)
    print(f"documents: {len(docs)}")
    print(f"chunks: {len(chunks)} ({stats['duplicates']} duplicates removed)")
    for doc_type in ("email", "attachment"):
        subset = [c for c in chunks if c["doc_type"] == doc_type]
        if not subset:
            continue
        per_doc = Counter(c["doc_id"] for c in subset)
        print(
            f"  {doc_type:<10} chunks={len(subset):>5}  "
            f"median chars={statistics.median(len(c['text']) for c in subset):.0f}  "
            f"single-chunk docs={sum(n == 1 for n in per_doc.values())}/{len(per_doc)}  "
            f"max chunks/doc={max(per_doc.values())}"
        )
    print(f"documents without chunks (too short or duplicated): {len({d['id'] for d in docs} - {c['doc_id'] for c in chunks})}")


if __name__ == "__main__":
    main()
