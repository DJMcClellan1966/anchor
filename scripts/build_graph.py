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

from anchor.corpus_graph import build_graph, save_graph


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
    args = parser.parse_args()

    encoded_path = args.data_dir / args.encoded_file
    if not encoded_path.exists():
        print(f"Missing {encoded_path}. Run build_vocab.py first.")
        raise SystemExit(1)

    graph = build_graph(encoded_path, top_similar_per_sentence=args.top_similar)
    out_path = args.data_dir / args.graph_file
    save_graph(graph, out_path)
    print(f"Wrote graph to {out_path}")


if __name__ == "__main__":
    main()
