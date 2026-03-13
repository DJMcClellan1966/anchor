"""
Suggest minimal set of dictionary terms for a set of queries (Theorem 5: minimal realizing set).
Reads queries from a file (one per line) or from repeated --query args; uses get_concept_bundle
per query; outputs the union of terms. Optionally writes a pruned Webster JSON with only those terms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor.retrieval import get_concept_bundle
from anchor.webster_engine import WebsterEngine


def run(
    queries: list[str],
    webster_path: Path | None = None,
    output_path: Path | None = None,
    pruned_json_path: Path | None = None,
) -> list[str]:
    """
    For each query, get concept bundle and collect terms. Return sorted unique terms.
    If webster_path and pruned_json_path are set, write a pruned dictionary JSON.
    """
    if not queries:
        return []
    engine = None
    if webster_path and webster_path.exists():
        engine = WebsterEngine(webster_path)
    elif webster_path:
        raise FileNotFoundError(f"Webster path not found: {webster_path}")

    terms_set: set[str] = set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        if engine is None:
            # No engine: cannot resolve terms; need webster for get_concept_bundle
            continue
        bundle = get_concept_bundle(engine, q)
        for t in bundle.get("terms") or []:
            if isinstance(t, str) and t.strip():
                terms_set.add(t.strip())

    terms_sorted = sorted(terms_set)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for t in terms_sorted:
                f.write(t + "\n")

    if pruned_json_path is not None and webster_path and webster_path.exists() and terms_sorted:
        with open(webster_path, encoding="utf-8") as f:
            full = json.load(f)
        if isinstance(full, dict):
            terms_lower = {t.lower() for t in terms_sorted}
            pruned = {k: v for k, v in full.items() if isinstance(k, str) and k.lower() in terms_lower}
            pruned_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pruned_json_path, "w", encoding="utf-8") as f:
                json.dump(pruned, f, ensure_ascii=False, indent=0)

    return terms_sorted


def main() -> None:
    p = argparse.ArgumentParser(
        description="Suggest minimal dictionary terms for a set of queries (Theorem 5)."
    )
    p.add_argument(
        "--queries",
        type=Path,
        help="File with one query per line",
    )
    p.add_argument(
        "-q", "--query",
        action="append",
        default=[],
        dest="query_list",
        help="Query string (repeat for multiple)",
    )
    p.add_argument(
        "--webster",
        type=Path,
        help="Path to dictionary JSON (required for resolving terms)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        help="Write term list (one per line) to this file",
    )
    p.add_argument(
        "--pruned",
        type=Path,
        dest="pruned_json",
        help="Write pruned Webster JSON containing only the suggested terms (requires --webster)",
    )
    args = p.parse_args()

    queries: list[str] = []
    if args.queries and args.queries.exists():
        with open(args.queries, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    queries.extend(args.query_list or [])

    if not queries:
        print("No queries provided. Use --queries <file> or -q/--query <query>.", file=sys.stderr)
        sys.exit(1)
    if not args.webster:
        print("--webster is required to resolve terms from queries.", file=sys.stderr)
        sys.exit(1)

    terms = run(
        queries,
        webster_path=args.webster,
        output_path=args.output,
        pruned_json_path=args.pruned_json,
    )

    for t in terms:
        print(t)
    if args.pruned_json and terms:
        print(f"Wrote pruned dictionary to {args.pruned_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
