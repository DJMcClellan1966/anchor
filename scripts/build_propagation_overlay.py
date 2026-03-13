"""
Build propagation_overlay.json from feedback_weights.json (learnable propagation).
For each query, sentence IDs that co-occurred in accepted responses get edge boosts.
Format: { "word_word": {"w|w2": boost}, "sentence_sentence": {"s|s2": boost} }.
Run after build_feedback_weights.py; set propagation_overlay_path in config to use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))


def build_propagation_overlay(
    feedback_weights_path: Path,
    output_path: Path,
    sentence_pair_boost: float = 0.5,
) -> None:
    """
    Read feedback_weights (query_key -> [sentence_id, ...]). For each query, add
    sentence_sentence overlay for each pair (s1, s2) in the list with boost.
    """
    word_word: dict[str, float] = {}
    sentence_sentence: dict[str, float] = {}
    if not feedback_weights_path.exists():
        overlay = {"word_word": word_word, "sentence_sentence": sentence_sentence}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(overlay, f, indent=0)
        return
    with open(feedback_weights_path, encoding="utf-8") as f:
        weights = json.load(f)
    if not isinstance(weights, dict):
        weights = {}
    for _query_key, sids in weights.items():
        if not isinstance(sids, list):
            continue
        sids = [int(x) for x in sids if isinstance(x, (int, float))]
        for i, s1 in enumerate(sids):
            for s2 in sids[i:]:
                if s1 == s2:
                    key = f"{s1}|{s2}"
                    sentence_sentence[key] = sentence_sentence.get(key, 0.0) + sentence_pair_boost
                else:
                    key1 = f"{s1}|{s2}"
                    key2 = f"{s2}|{s1}"
                    sentence_sentence[key1] = sentence_sentence.get(key1, 0.0) + sentence_pair_boost
                    sentence_sentence[key2] = sentence_sentence.get(key2, 0.0) + sentence_pair_boost
    overlay = {"word_word": word_word, "sentence_sentence": sentence_sentence}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build propagation_overlay.json from feedback_weights.json for learnable propagation."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--feedback-weights",
        default="feedback_weights.json",
        help="Feedback weights file relative to data_dir (default: feedback_weights.json)",
    )
    parser.add_argument(
        "--output",
        default="propagation_overlay.json",
        help="Output overlay file relative to data_dir (default: propagation_overlay.json)",
    )
    parser.add_argument(
        "--sentence-pair-boost",
        type=float,
        default=0.5,
        help="Boost for each sentence-sentence edge from accepted responses (default: 0.5)",
    )
    args = parser.parse_args()

    feedback_weights_path = args.data_dir / args.feedback_weights
    output_path = args.data_dir / args.output

    build_propagation_overlay(
        feedback_weights_path,
        output_path,
        sentence_pair_boost=args.sentence_pair_boost,
    )
    print(f"Wrote propagation overlay to {output_path}")


if __name__ == "__main__":
    main()
