"""
Build Anchor sentence corpus from Hugging Face datasets (OpenSubtitles, C4, etc.).
Writes output_dir/corpus/sentences.jsonl in the same format as build_corpus.py.
Requires: pip install datasets
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("This script requires the 'datasets' package. Install with: pip install datasets", file=sys.stderr)
    sys.exit(1)

# Reuse same sentence-split and genre logic as build_corpus (duplicated to avoid path/import issues)
def _sentence_split(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _assign_genre(sentence: str, default_genre: str = "general") -> str:
    s = sentence.lower()
    if any(w in s for w in ("definition", "defined as", "means ", "refers to")):
        return "definitional"
    if any(w in s for w in ("retire", "retirement", "pension", "savings")):
        return "retirement"
    if any(w in s for w in ("story", "once", "then ", "narrative")):
        return "narrative"
    return default_genre


def _extract_sentences(
    text: str,
    text_field: str,
    min_len: int,
    max_len: int,
    default_genre: str,
    split_long: int,
) -> list[tuple[str, str]]:
    """Return list of (text, genre_id). Treats long text as passage and sentence-splits."""
    raw = text if isinstance(text, str) else (text.get(text_field, "") if isinstance(text, dict) else "")
    if not (raw or "").strip():
        return []
    out: list[tuple[str, str]] = []
    if len(raw) > split_long:
        for sent in _sentence_split(raw):
            sent = sent.strip()
            if not sent or len(sent) < 3:
                continue
            if min_len <= len(sent) <= max_len:
                g = _assign_genre(sent, default_genre)
                out.append((sent, g))
    else:
        raw = raw.strip()
        if len(raw) >= 3 and min_len <= len(raw) <= max_len:
            g = _assign_genre(raw, default_genre)
            out.append((raw, g))
    return out


def build_from_hf(
    dataset: str,
    output_dir: Path,
    config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    min_len: int = 8,
    max_len: int = 500,
    max_sentences: int = 100_000,
    streaming: bool = False,
    also_write_per_genre: bool = True,
    default_genre: str = "general",
    split_long_threshold: int = 300,
) -> int:
    seen: set[tuple[str, str]] = set()
    all_sentences: list[dict] = []
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    try:
        if config:
            ds = load_dataset(dataset, config, split=split, streaming=streaming)
        else:
            ds = load_dataset(dataset, split=split, streaming=streaming)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        print("Check dataset and config names (e.g. open_subtitles/en, allenai/c4/en).", file=sys.stderr)
        raise

    for ex in ds:
        if len(all_sentences) >= max_sentences:
            break
        if isinstance(ex, dict):
            text = ex.get(text_field, "")
        else:
            text = getattr(ex, text_field, "") or ""
        for sent, g in _extract_sentences(
            text, text_field, min_len, max_len, default_genre, split_long_threshold
        ):
            key = (sent, g)
            if key in seen:
                continue
            seen.add(key)
            all_sentences.append({"text": sent, "genre_id": g})
            if len(all_sentences) >= max_sentences:
                break

    for i, rec in enumerate(all_sentences):
        rec["sentence_id"] = i

    sentences_path = corpus_dir / "sentences.jsonl"
    with open(sentences_path, "w", encoding="utf-8") as f:
        for rec in all_sentences:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if also_write_per_genre:
        by_genre: dict[str, list[dict]] = {}
        for rec in all_sentences:
            by_genre.setdefault(rec["genre_id"], []).append(rec)
        for genre_id, recs in by_genre.items():
            genre_dir = output_dir / genre_id
            genre_dir.mkdir(parents=True, exist_ok=True)
            with open(genre_dir / "genre_sentences.jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    return len(all_sentences)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Anchor sentence corpus from a Hugging Face dataset (OpenSubtitles, C4, etc.)."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name (e.g. open_subtitles, allenai/c4)")
    parser.add_argument("--config", default=None, help="Dataset config (e.g. en). Omit if dataset has no config.")
    parser.add_argument("--split", default="train", help="Split to use (default: train)")
    parser.add_argument("--text-field", default="text", help="Key for text in each example (default: text)")
    parser.add_argument("--min-len", type=int, default=8, help="Min character length per sentence (default: 8)")
    parser.add_argument("--max-len", type=int, default=500, help="Max character length per sentence (default: 500)")
    parser.add_argument("--max-sentences", type=int, default=100_000, help="Cap total sentences written (default: 100000)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data"), help="Output root (default: data)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming for large datasets (e.g. C4)")
    parser.add_argument("--no-per-genre", action="store_true", help="Do not write per-genre genre_sentences.jsonl")
    parser.add_argument("--default-genre", default="general", help="Default genre_id (default: general)")
    parser.add_argument(
        "--split-long",
        type=int,
        default=300,
        help="Treat examples longer than this as passages and sentence-split (default: 300)",
    )
    args = parser.parse_args()

    n = build_from_hf(
        dataset=args.dataset,
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        text_field=args.text_field,
        min_len=args.min_len,
        max_len=args.max_len,
        max_sentences=args.max_sentences,
        streaming=args.streaming,
        also_write_per_genre=not args.no_per_genre,
        default_genre=args.default_genre,
        split_long_threshold=args.split_long,
    )
    print(f"Wrote {n} sentences to {args.output_dir / 'corpus' / 'sentences.jsonl'}")


if __name__ == "__main__":
    main()
