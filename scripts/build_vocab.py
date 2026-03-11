"""
Build vocabulary and encoded sentences from corpus (Option C).
Run after build_corpus.py. Optionally seed vocab from dictionary terms file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add anchor to path when run as script
_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

from anchor.corpus_vocab import run_build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocab and encoded sentences from corpus.")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Data directory containing corpus/sentences.jsonl (default: data)",
    )
    parser.add_argument(
        "--dictionary-terms-file",
        type=Path,
        default=None,
        help="Optional file with one dictionary term per line to seed vocab",
    )
    parser.add_argument(
        "--corpus-file",
        default="corpus/sentences.jsonl",
        help="Path relative to data_dir (default: corpus/sentences.jsonl)",
    )
    parser.add_argument(
        "--vocab-file",
        default="corpus/vocab.json",
        help="Path relative to data_dir (default: corpus/vocab.json)",
    )
    parser.add_argument(
        "--encoded-file",
        default="corpus/encoded_sentences.jsonl",
        help="Path relative to data_dir (default: corpus/encoded_sentences.jsonl)",
    )
    args = parser.parse_args()

    dictionary_terms = None
    if args.dictionary_terms_file and args.dictionary_terms_file.exists():
        dictionary_terms = [
            line.strip()
            for line in args.dictionary_terms_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    vocab_size, n_encoded = run_build(
        args.data_dir,
        corpus_sentences_file=args.corpus_file,
        vocab_file=args.vocab_file,
        encoded_file=args.encoded_file,
        dictionary_terms=dictionary_terms,
    )
    print(f"Vocab size: {vocab_size}. Encoded sentences: {n_encoded}.")
    print(f"Wrote {args.data_dir / args.vocab_file} and {args.data_dir / args.encoded_file}")


if __name__ == "__main__":
    main()
