"""
Ingest the dictionary repo's compiled corpora into Anchor's sentences.jsonl.

The dictionary repo (https://github.com/DJMcClellan1966/dictionary) can compile:
- ConceptNet (~34M assertions, filtered to English)
- GooAQ (3M+ Q&A pairs from Google)
- The Stack (Python docstrings)
- Python stdlib docs

Each produces compiled_*.json: word -> { noun: { def, definitions }, source }.
This script reads those files and emits Anchor sentences.jsonl so the same
large corpus drives Anchor's graph and retrieval.

Usage:
  python scripts/build_corpus_from_dictionary.py path/to/dictionary/data -o data --append
  python scripts/build_corpus_from_dictionary.py path/to/dictionary/data -o data --sources gooaq conceptnet --max 500000

Requires the dictionary repo to have been run at least once, e.g.:
  cd path/to/dictionary && python compile_corpus.py --source gooaq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_DEF_LEN = 1000
# Which compiled files to look for and their genre_id in Anchor
SOURCES = {
    "compiled_conceptnet.json": "conceptnet",
    "compiled_corpus.json": "gooaq",
    "compiled_stack.json": "stack",
    "compiled_stdlib.json": "stdlib",
}


def _emit_sentences(compiled_path: Path, genre_id: str, max_sentences: int | None) -> list[dict]:
    """Read compiled_*.json (word -> entry with noun.def/definitions) and yield sentence records."""
    if not compiled_path.exists():
        return []
    with open(compiled_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for word_key, entry in data.items():
        if max_sentences and len(out) >= max_sentences:
            break
        if not isinstance(entry, dict):
            continue
        noun = entry.get("noun") or entry.get("definitions") or {}
        if isinstance(noun, list):
            defs = noun
        else:
            defs = []
            single = (noun.get("def") if isinstance(noun, dict) else None) or entry.get("def")
            if single and isinstance(single, str):
                defs.append(single)
            defs.extend((noun.get("definitions") or []) if isinstance(noun, dict) else [])
        term = (word_key or "").replace("_", " ").strip() or "term"
        for d in defs:
            if not isinstance(d, str) or not d.strip():
                continue
            if max_sentences and len(out) >= max_sentences:
                break
            text = d.strip()[:MAX_DEF_LEN]
            if term and term != "term":
                text = f"{term}: {text}"
            out.append({"text": text, "genre_id": genre_id})
        # GooAQ: also add questions as sentences when available
        if genre_id == "gooaq" and isinstance(entry.get("questions"), list):
            for q in entry["questions"][:2]:
                if max_sentences and len(out) >= max_sentences:
                    break
                if isinstance(q, str) and q.strip():
                    out.append({"text": q.strip()[:500], "genre_id": genre_id})
    return out


def run(
    dictionary_data_dir: Path,
    output_dir: Path,
    sources: list[str] | None = None,
    append: bool = False,
    max_sentences: int | None = None,
) -> int:
    """Read dictionary compiled JSONs, write or append to output_dir/corpus/sentences.jsonl. Returns total lines written."""
    data_dir = Path(dictionary_data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dictionary data dir not found: {data_dir}")
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / "sentences.jsonl"
    lines_out: list[str] = []
    if append and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            lines_out = [line.rstrip("\n") for line in f if line.strip()]
    written = 0
    which = sources or list(SOURCES.keys())
    for filename, genre_id in SOURCES.items():
        if filename not in which:
            continue
        path = data_dir / filename
        records = _emit_sentences(path, genre_id, max_sentences)
        for rec in records:
            lines_out.append(json.dumps(rec, ensure_ascii=False))
            written += 1
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ingest dictionary repo compiled corpora (ConceptNet, GooAQ, Stack, Stdlib) into Anchor sentences.jsonl",
    )
    p.add_argument(
        "dictionary_data",
        type=Path,
        help="Path to dictionary repo data/ directory (contains compiled_*.json)",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("data"), help="Anchor output directory (default: data)")
    p.add_argument(
        "--sources",
        nargs="*",
        choices=list(SOURCES.keys()),
        default=None,
        help="Which compiled files to use (default: all found)",
    )
    p.add_argument("--append", action="store_true", help="Append to existing sentences.jsonl")
    p.add_argument("--max", type=int, default=None, metavar="N", help="Max sentences per source (default: no limit)")
    args = p.parse_args()
    n = run(args.dictionary_data, args.output, sources=args.sources, append=args.append, max_sentences=args.max)
    print(f"Wrote {n} lines to {args.output / 'corpus' / 'sentences.jsonl'}")


if __name__ == "__main__":
    main()
