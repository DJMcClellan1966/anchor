"""
Ingest Webster's dictionary JSON into the combined corpus (sentences.jsonl).
Each entry becomes one line: {"text": "term: definition", "genre_id": "definitional"} (or --genre).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_DEF_LEN = 1000
MAX_LINES = 500_000


def run(
    webster_json_path: Path,
    output_dir: Path,
    genre_id: str = "definitional",
    append: bool = False,
    max_lines: int = MAX_LINES,
) -> int:
    """Load Webster JSON, write or append to output_dir/corpus/sentences.jsonl. Returns lines written."""
    if not webster_json_path.exists():
        raise FileNotFoundError(str(webster_json_path))
    with open(webster_json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return 0
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / "sentences.jsonl"
    lines_out: list[str] = []
    if append and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            lines_out = [line.rstrip("\n") for line in f if line.strip()]
    written = 0
    for term, definition in data.items():
        if written >= max_lines:
            break
        if not isinstance(term, str) or not isinstance(definition, str):
            continue
        def_trim = definition[:MAX_DEF_LEN] if len(definition) > MAX_DEF_LEN else definition
        text = f"{term}: {def_trim}"
        lines_out.append(json.dumps({
            "text": text,
            "genre_id": genre_id,
            "source": "dictionary",
            "term": term,
        }, ensure_ascii=False))
        written += 1
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest Webster dictionary JSON into corpus sentences.jsonl")
    p.add_argument("webster_json", type=Path, help="Path to dictionary.json or dictionary_compact.json")
    p.add_argument("-o", "--output", type=Path, default=Path("data"), help="Output directory (default: data)")
    p.add_argument("--genre", default="definitional", help="genre_id for all lines (default: definitional)")
    p.add_argument("--append", action="store_true", help="Append to existing sentences.jsonl")
    p.add_argument("--max-lines", type=int, default=MAX_LINES, help=f"Max new lines to add (default: {MAX_LINES})")
    args = p.parse_args()
    n = run(args.webster_json, args.output, genre_id=args.genre, append=args.append, max_lines=args.max_lines)
    print(f"Wrote {n} lines to {args.output / 'corpus' / 'sentences.jsonl'}")


if __name__ == "__main__":
    main()
