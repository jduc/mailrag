"""Compare search modes on questions with known answers.

    uv run evaluate.py [data/eval.json]

The eval file is a JSON list (see eval.example.json). A retrieved chunk counts as correct when its
`field` (subject, text or filename) equals `equals`, or contains every string in `contains`.
"""

import json
import sys
from pathlib import Path

import config
from rag import MODES, search

DEFAULT_PATH = Path("data/eval.json")


def is_correct(case: dict, chunk: dict) -> bool:
    value = chunk[case["field"]]
    if "equals" in case:
        return value == case["equals"]
    return all(s in value for s in case["contains"])


def first_hit_rank(case: dict, results: list[dict]) -> int | None:
    return next((i + 1 for i, r in enumerate(results) if is_correct(case, r)), None)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"{path} not found: copy eval.example.json there and write questions about your own mails.")
    cases = json.loads(path.read_text(encoding="utf-8"))

    ranks = {mode: [] for mode in MODES}
    print(f"Rank of the first correct chunk in the top {config.TOP_K} ('-' = not found)\n")
    print(f"  {'question':<52}{'kind':<26}" + "".join(f"{m:>8}" for m in MODES))
    for case in cases:
        row = f"  {case['question'][:50]:<52}{case.get('kind', ''):<26}"
        for mode in MODES:
            rank = first_hit_rank(case, search(case["question"], mode=mode))
            ranks[mode].append(rank)
            row += f"{rank or '-':>8}"
        print(row)

    print()
    for name, metric in [
        ("Hit@1", lambda rs: sum(r == 1 for r in rs) / len(rs)),
        (f"Hit@{config.TOP_K}", lambda rs: sum(r is not None for r in rs) / len(rs)),
        ("MRR", lambda rs: sum(1 / r for r in rs if r) / len(rs)),
    ]:
        print(f"  {name:<78}" + "".join(f"{metric(ranks[m]):>8.2f}" for m in MODES))


if __name__ == "__main__":
    main()
