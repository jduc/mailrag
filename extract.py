"""Extract an mbox file into JSONL documents: one per email body and one per attachment.

    uv run extract.py data/EMAILS.mbox -o data/processed/documents.jsonl
"""

import argparse
import email
import email.policy
import hashlib
import io
import json
import logging
import mailbox
import re
import sys
from email.message import EmailMessage
from email.utils import getaddresses
from pathlib import Path

from markitdown import MarkItDown, StreamInfo

log = logging.getLogger("extract")

CONVERTIBLE_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".csv", ".txt", ".md", ".html", ".htm"}
# Large data dumps (e.g. spreadsheets of raw numbers) are noise for retrieval
MAX_ATTACHMENT_CHARS = 50_000

SEPARATOR_LINE = re.compile(r"^(\\?_){10,}\s*$")  # Outlook separator, possibly markdown-escaped
REPLY_HEADER_PATTERNS = [
    re.compile(r"^-{2,}\s*(Original Message|Message d'origine|Ursprüngliche Nachricht)\s*-{2,}", re.I),
    SEPARATOR_LINE,
    re.compile(r"^On\b.{0,200}\bwrote:\s*$", re.I),
    re.compile(r"^Le\b.{0,200}\ba écrit\s*:\s*$", re.I),
    re.compile(r"^Am\b.{0,200}\bschrieb\b.{0,100}:\s*$", re.I),
]
# Outlook-style quote block: "From: X" followed shortly by "Sent: ..."
FROM_LINE = re.compile(r"^[*_\s]*(From|De|Von)[*_\s]*:", re.I)
SENT_LINE = re.compile(r"^[*_\s]*(Sent|Date|Envoyé|Gesendet)[*_\s]*:", re.I)

# Closing line, optionally followed by a name ("Kind regards, Anna")
SIGN_OFF = re.compile(
    r"^[*_\s]*(best|best regards|kind regards|warm regards|with best regards|regards|best wishes|all the best"
    r"|cheers|thanks|thank you|many thanks|thanks a lot|thanks very much|thank you very much|sincerely"
    r"|meilleures salutations|salutations|cordialement|bien à vous|bonne journée|merci)"
    r"([,.!]?\s+[\w.-]+)?[,.!]?[*_\s]*$",
    re.I,
)
SIGNATURE_MAX_LINES = 25
SIGNATURE_MAX_LINE_CHARS = 160
# Specific phrases only: plain "confidential" also appears in real contract content
DISCLAIMER = re.compile(
    r"intended (solely|only) for|intended recipient|not the intended|received (this|it) in error"
    r"|n'êtes pas le destinataire|before printing|^\W*disclaimer\W*$",
    re.I,
)
DISCLAIMER_MAX_CHARS = 1500
MEETING_LINE = re.compile(
    r"Microsoft Teams (meeting|Need help)|Join on your computer|Click here to join|Join the meeting now"
    r"|^\W*Meeting ID:|^\W*Passcode:|Download Teams|Join on the web|Meeting options"
    r"|Join with a video conferencing device|Video Conference ID|Alternate VTC|Or call in \(audio only\)"
    r"|Phone Conference ID|Find a local number|Reset PIN|Join Zoom Meeting|zoom\.us/j/|One tap mobile"
    r"|Dial by your location|Find your local number|teams\.microsoft\.com|aka\.ms/JoinTeamsMeeting"
    r"|^\+?[\d\s()-]{8,},,\d+#|^Sent from (my |\[?Outlook)",
    re.I,
)
FORWARD_SUBJECT = re.compile(r"^\s*(fwd?|tr|wg)\s*:", re.I)
REPLY_PREFIX = re.compile(r"^\s*((re|aw|sv|tr|fwd?|wg)\s*(\[\d+\])?\s*:\s*)+", re.I)

_md = MarkItDown(enable_plugins=False)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)  # zero-width chars, soft hyphen
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # markdown images
    text = re.sub(r"\[cid:[^\]]*\]", "", text)  # inline image placeholders
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_quoted(text: str) -> str:
    """Cut the body at the first reply header and drop '>' quoted lines."""
    lines = text.split("\n")
    cut = len(lines)
    seen_content = False
    for i, line in enumerate(lines):
        s = line.strip()
        # A separator before any content (e.g. a Teams invite banner) is not quoted history
        if not seen_content:
            seen_content = bool(s) and not SEPARATOR_LINE.match(s)
            continue
        # Reply headers are often wrapped over two lines: "On 24.07.23 10:28, Anna\nwrote:"
        joined = f"{s} {lines[i + 1].strip()}" if i + 1 < len(lines) else s
        if any(p.match(s) or p.match(joined) for p in REPLY_HEADER_PATTERNS):
            cut = i
            break
        if FROM_LINE.match(s) and any(SENT_LINE.match(l.strip()) for l in lines[i + 1 : i + 5]):
            cut = i
            break
    kept = [l for l in lines[:cut] if not l.lstrip().startswith(">")]
    return clean_text("\n".join(kept))


def strip_signature(text: str) -> str:
    """Cut at the last sign-off line, but only if everything after it looks like a signature block."""
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if SIGN_OFF.match(lines[i].strip()):
            if not any(l.strip() for l in lines[:i]):
                break  # a "Thanks Anna" first line is a greeting, not a closing
            tail = [l.strip() for l in lines[i + 1 :] if l.strip()]
            if (
                len(tail) <= SIGNATURE_MAX_LINES
                and all(len(l) <= SIGNATURE_MAX_LINE_CHARS for l in tail)
                and not any("?" in l for l in tail)
            ):
                return "\n".join(lines[:i])
            break
    return text


def strip_boilerplate(text: str) -> str:
    """Remove meeting-invite lines, separator lines, legal disclaimers and the signature."""
    lines = [l for l in text.split("\n") if not (MEETING_LINE.search(l.strip()) or SEPARATOR_LINE.match(l.strip()))]
    paragraphs = re.split(r"\n\s*\n", "\n".join(lines))
    text = "\n\n".join(drop_disclaimer(p) for p in paragraphs)
    return clean_text(strip_signature(text))


def drop_disclaimer(paragraph: str) -> str:
    if not DISCLAIMER.search(paragraph):
        return paragraph
    if len(paragraph) <= DISCLAIMER_MAX_CHARS:
        return ""
    # Long block, e.g. a '>'-quoted forward without blank lines: only drop the matching lines
    return "\n".join(l for l in paragraph.split("\n") if not DISCLAIMER.search(l))


def html_to_markdown(html: str) -> str:
    result = _md.convert_stream(io.BytesIO(html.encode("utf-8")), stream_info=StreamInfo(extension=".html", charset="utf-8"))
    return result.markdown


def header(msg: EmailMessage, name: str) -> str:
    try:
        value = msg.get(name)
        return str(value).strip() if value is not None else ""
    except Exception:  # malformed header
        raw = msg.get_all(name, failobj=[""])[0]
        return str(raw).strip()


def addresses(msg: EmailMessage, name: str) -> list[dict]:
    values = msg.get_all(name, [])
    return [{"name": n, "email": a.lower()} for n, a in getaddresses([str(v) for v in values]) if a]


def message_ids(value: str) -> list[str]:
    return re.findall(r"<[^<>\s]+>", value or "")


def part_text(part: EmailMessage) -> str:
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace")


def body_text(msg: EmailMessage) -> str:
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    text = part_text(body)
    if body.get_content_subtype() == "html":
        text = html_to_markdown(text)
    return clean_text(text)


def ics_to_text(raw: str) -> str:
    raw = re.sub(r"\r?\n[ \t]", "", raw)  # unfold lines
    keep = ("SUMMARY", "DTSTART", "DTEND", "LOCATION", "DESCRIPTION", "ORGANIZER", "ATTENDEE")
    out = []
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        if key.split(";")[0] in keep:
            out.append(f"{key.split(';')[0].title()}: {value.replace(chr(92) + 'n', chr(10))}")
    return "\n".join(out)


def attachment_text(part: EmailMessage, filename: str) -> str:
    data = part.get_payload(decode=True) or b""
    if part.get_content_type() == "text/calendar" or filename.lower().endswith(".ics"):
        return ics_to_text(data.decode(part.get_content_charset() or "utf-8", errors="replace"))
    ext = Path(filename).suffix.lower()
    if ext not in CONVERTIBLE_EXTENSIONS:
        return ""
    result = _md.convert_stream(io.BytesIO(data), stream_info=StreamInfo(extension=ext, mimetype=part.get_content_type()))
    return result.markdown


def iter_attachments(msg: EmailMessage):
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        ctype = part.get_content_type()
        if ctype == "text/calendar":
            yield part, filename or "invite.ics"
        elif filename and not ctype.startswith("image/"):
            yield part, filename


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def extract_message(msg: EmailMessage, index: int) -> list[dict]:
    message_id = header(msg, "Message-ID") or f"<missing-{index}@mailrag>"
    references = message_ids(header(msg, "References"))
    in_reply_to = (message_ids(header(msg, "In-Reply-To")) or [None])[0]
    thread_root = references[0] if references else (in_reply_to or message_id)

    try:
        date = msg["Date"].datetime.isoformat() if msg["Date"] else None
    except Exception:
        date = None

    subject = header(msg, "Subject")
    senders = addresses(msg, "From")
    base = {
        "message_id": message_id,
        "thread_id": stable_id(thread_root),
        "date": date,
        "subject": subject,
        "normalized_subject": REPLY_PREFIX.sub("", subject).strip().lower(),
        "from": senders[0] if senders else None,
    }

    full = body_text(msg)
    # Forwards keep their content (often the only copy); replies lose the quoted history
    text = full if FORWARD_SUBJECT.match(subject) else (strip_quoted(full) or full)
    before_cleanup = len(text)
    text = strip_boilerplate(text)

    docs = []
    email_id = stable_id(message_id)
    attachment_names = []
    for part, filename in iter_attachments(msg):
        attachment_names.append(filename)
        try:
            att_text = clean_text(attachment_text(part, filename))
        except Exception as e:
            log.warning("attachment %r in %s failed: %s", filename, message_id, e)
            continue
        if not att_text:
            continue
        original_chars = len(att_text)
        att_text = att_text[:MAX_ATTACHMENT_CHARS]
        docs.append({
            "id": stable_id(message_id, filename, str(len(docs))),
            "doc_type": "attachment",
            "parent_id": email_id,
            **base,
            "filename": filename,
            "mime_type": part.get_content_type(),
            "truncated": original_chars > MAX_ATTACHMENT_CHARS,
            "original_chars": original_chars,
            "text": att_text,
        })

    email_doc = {
        "id": email_id,
        "doc_type": "email",
        **base,
        "to": addresses(msg, "To"),
        "cc": addresses(msg, "Cc"),
        "in_reply_to": in_reply_to,
        "attachments": attachment_names,
        "quoted_stripped": before_cleanup != len(full),
        "boilerplate_removed": before_cleanup - len(text),
        "text": text,
    }
    return [email_doc, *docs]


def run(mbox_path: Path, out_path: Path) -> None:
    box = mailbox.mbox(
        mbox_path,
        factory=lambda f: email.message_from_binary_file(f, policy=email.policy.default),
        create=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"messages": 0, "duplicates": 0, "failed": 0, "email": 0, "attachment": 0, "empty": 0}

    with out_path.open("w", encoding="utf-8") as out:
        for i, msg in enumerate(box):
            counts["messages"] += 1
            mid = header(msg, "Message-ID")
            if mid and mid in seen:
                counts["duplicates"] += 1
                continue
            seen.add(mid)
            try:
                docs = extract_message(msg, i)
            except Exception as e:
                counts["failed"] += 1
                log.warning("message %d (%s) failed: %s", i, mid, e)
                continue
            for doc in docs:
                if doc["doc_type"] == "email" and not doc["text"]:
                    counts["empty"] += 1
                counts[doc["doc_type"]] += 1
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    log.info("done: %s -> %s", counts, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mbox", type=Path, help="mbox file, e.g. data/EMAILS.mbox")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed/documents.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    run(args.mbox, args.output)


if __name__ == "__main__":
    main()
