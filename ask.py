"""Ask a question about the mail archive.

    uv run ask.py "When did we sign the contract with Example Corp?"
    uv run ask.py "invoice currency" --from example.com --since 2023-01-01 --show-chunks
    uv run ask.py "INV-2023-042" --search-only
"""

import argparse
import sys
import textwrap
import time

import config
from rag import MODES, build_messages, estimate_tokens, generate, search, source_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only mails on or after this date")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="only mails on or before this date")
    p.add_argument("--from", dest="sender", metavar="TEXT", help="only mails whose sender name/email contains TEXT")
    p.add_argument("--k", type=int, default=config.TOP_K, help=f"number of sources (default {config.TOP_K})")
    p.add_argument("--mode", choices=MODES, default="hybrid", help="search mode (default hybrid)")
    p.add_argument("--search-only", action="store_true", help="show retrieved sources, skip the LLM")
    p.add_argument("--show-chunks", action="store_true", help="also print the text of each source")
    p.add_argument("--think", action="store_true", help="let the model reason first (slower, sometimes better)")
    return p.parse_args()


def print_sources(sources: list[dict], show_chunks: bool) -> None:
    print("\nSources:")
    for i, r in enumerate(sources, 1):
        print(f"  [{i}] {source_label(r)}  (score {r['score']:.3f})")
        if show_chunks:
            print(textwrap.indent(r["text"].strip(), "      │ "), "\n")


def main() -> None:
    args = parse_args()
    try:
        sources = search(args.question, k=args.k, mode=args.mode, since=args.since, until=args.until, sender=args.sender)
        if not sources:
            sys.exit("No matching mails (check the filters).")
        if args.search_only:
            return print_sources(sources, args.show_chunks)

        # Ollama silently drops the start of a prompt longer than num_ctx
        needed = estimate_tokens(build_messages(args.question, sources))
        if needed > config.NUM_CTX * 0.8:
            print(f"warning: prompt is ~{needed} tokens, close to num_ctx={config.NUM_CTX}; use a lower --k\n")

        start = time.perf_counter()
        print(f"(reading {len(sources)} sources, ~{needed} tokens)\n", file=sys.stderr)
        in_thinking = False
        for chunk in generate(args.question, sources, think=args.think):
            if chunk.message.thinking:  # dimmed
                if not in_thinking:
                    print("\033[2m[thinking]\n", end="")
                    in_thinking = True
                print(chunk.message.thinking, end="", flush=True)
            if chunk.message.content:
                if in_thinking:
                    print("\033[0m\n")
                    in_thinking = False
                print(chunk.message.content, end="", flush=True)
            if chunk.done:
                print(f"\n\n({time.perf_counter() - start:.0f}s, {chunk.eval_count} tokens generated)", file=sys.stderr)
        print_sources(sources, args.show_chunks)

    except ConnectionError:
        sys.exit("Cannot reach Ollama. Start it with:  brew services start ollama")


if __name__ == "__main__":
    main()
