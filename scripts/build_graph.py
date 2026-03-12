"""
Build word/sentence graph from encoded corpus (Option C).
Run after build_vocab.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

from anchor.corpus_graph import build_graph, export_corpus_model, save_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus graph from encoded sentences.")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Data directory containing corpus/encoded_sentences.jsonl (default: data)",
    )
    parser.add_argument(
        "--encoded-file",
        default="corpus/encoded_sentences.jsonl",
        help="Path relative to data_dir (default: corpus/encoded_sentences.jsonl)",
    )
    parser.add_argument(
        "--graph-file",
        default="corpus/graph.json",
        help="Output path relative to data_dir (default: corpus/graph.json)",
    )
    parser.add_argument(
        "--top-similar",
        type=int,
        default=20,
        help="Top-k similar sentences per sentence (default: 20)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=5,
        help="Context length for inverted index (default: 5)",
    )
    parser.add_argument(
        "--corpus-model",
        default="corpus/corpus_model.json",
        help="Output path for by-product corpus model (default: corpus/corpus_model.json)",
    )
    parser.add_argument(
        "--no-corpus-model",
        action="store_true",
        help="Skip writing corpus_model.json",
    )
    args = parser.parse_args()

    encoded_path = args.data_dir / args.encoded_file
    if not encoded_path.exists():
        print(f"Missing {encoded_path}. Run build_vocab.py first.")
        raise SystemExit(1)

    graph = build_graph(
        encoded_path,
        top_similar_per_sentence=args.top_similar,
        context_length=args.context_length,
    )
    out_path = args.data_dir / args.graph_file
    save_graph(graph, out_path)
    print(f"Wrote graph to {out_path}")

    if not args.no_corpus_model:
        vocab_path = args.data_dir / "corpus" / "vocab.json"
        if vocab_path.exists():
            corpus_model_path = args.data_dir / args.corpus_model
            export_corpus_model(graph, vocab_path, corpus_model_path)
            print(f"Wrote corpus model to {corpus_model_path}")
        else:
            print("Skipping corpus model (vocab.json not found)")


if __name__ == "__main__":
    main()
