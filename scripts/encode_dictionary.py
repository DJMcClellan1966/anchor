"""
Encode dictionary definitions as token ID sequences (word/number dictionary).
Uses corpus vocab so the same numbers are used across corpus and dictionary.
Output: encoded_dictionary.jsonl for merge into build_graph (recurring patterns).

Usage:
  python scripts/encode_dictionary.py --vocab data/corpus/vocab.json --webster path/to/dictionary.json --output data/corpus/encoded_dictionary.jsonl
  python scripts/encode_dictionary.py --vocab data/corpus/vocab.json --definitions path/to/terms_defs.txt --output data/corpus/encoded_dictionary.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

from anchor.corpus_vocab import tokenize


def load_vocab(vocab_path: Path) -> dict[str, int]:
    """Load word_to_id from corpus vocab.json."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("word_to_id", {}))


def definitions_from_webster(webster_path: Path) -> list[tuple[str, str]]:
    """Yield (term, definition) from Webster JSON."""
    with open(webster_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []
    return [(term, defn) for term, defn in data.items() if isinstance(term, str) and isinstance(defn, str)]


def definitions_from_file(definitions_path: Path) -> list[tuple[str, str]]:
    """Yield (term, definition) from file with term\\tdefinition per line."""
    out: list[tuple[str, str]] = []
    with open(definitions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            term = (parts[0] or "").strip()
            defn = (parts[1] if len(parts) > 1 else "").strip()
            if term:
                out.append((term, defn))
    return out


def run(
    vocab_path: Path,
    output_path: Path,
    webster_path: Path | None = None,
    definitions_path: Path | None = None,
) -> int:
    """Encode definitions to token_ids; write JSONL. Returns number of entries written."""
    word_to_id = load_vocab(vocab_path)
    if webster_path and webster_path.exists():
        pairs = definitions_from_webster(webster_path)
    elif definitions_path and definitions_path.exists():
        pairs = definitions_from_file(definitions_path)
    else:
        raise FileNotFoundError("Provide --webster or --definitions with an existing file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for term, definition in pairs:
            tokens = tokenize(definition)
            token_ids = [word_to_id[t] for t in tokens if t in word_to_id]
            if not token_ids:
                continue
            f.write(json.dumps({"term": term, "token_ids": token_ids}, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode dictionary definitions as token ID sequences (word/number dictionary)."
    )
    parser.add_argument("--vocab", type=Path, required=True, help="Path to corpus/vocab.json")
    parser.add_argument("--output", type=Path, required=True, help="Output path (e.g. corpus/encoded_dictionary.jsonl)")
    parser.add_argument("--webster", type=Path, default=None, help="Path to Webster dictionary.json")
    parser.add_argument("--definitions", type=Path, default=None, help="Path to file with term\\tdefinition per line")
    args = parser.parse_args()

    if not args.webster and not args.definitions:
        parser.error("Provide --webster or --definitions")
    if args.webster and args.definitions:
        parser.error("Provide only one of --webster or --definitions")

    n = run(args.vocab, args.output, webster_path=args.webster, definitions_path=args.definitions)
    print(f"Wrote {n} encoded entries to {args.output}")


if __name__ == "__main__":
    main()
