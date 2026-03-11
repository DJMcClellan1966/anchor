"""
Build combined sentence corpus with genre tags (Option C).
Reads from local text files or a directory; sentence-splits, assigns genre, writes
data/corpus/sentences.jsonl. Optionally writes per-genre genre_sentences.jsonl for
backward compatibility with get_style_sentences().
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences. Simple rule: split on . ! ? followed by space or end."""
    if not (text or "").strip():
        return []
    # Split on sentence-ending punctuation followed by space or end of string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _assign_genre(sentence: str, default_genre: str = "general") -> str:
    """
    Assign genre_id to a sentence. Rule-based placeholder; override with metadata if available.
    """
    s = sentence.lower()
    if any(w in s for w in ("definition", "defined as", "means ", "refers to")):
        return "definitional"
    if any(w in s for w in ("retire", "retirement", "pension", "savings")):
        return "retirement"
    if any(w in s for w in ("story", "once", "then ", "narrative")):
        return "narrative"
    return default_genre


def build_from_directory(
    input_dir: Path,
    output_dir: Path,
    default_genre: str = "general",
    extensions: tuple[str, ...] = (".txt", ".jsonl"),
    also_write_per_genre: bool = True,
) -> int:
    """
    Scan input_dir for text files, extract sentences, tag with genre, write
    output_dir/corpus/sentences.jsonl. If also_write_per_genre, write
    output_dir/<genre_id>/genre_sentences.jsonl for each genre.
    Returns number of sentences written.
    """
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sentences_path = corpus_dir / "sentences.jsonl"

    all_sentences: list[dict] = []
    seen: set[str] = set()

    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in extensions:
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = obj.get("text") if isinstance(obj, dict) else None
                    if isinstance(t, str) and t.strip():
                        g = obj.get("genre_id") or _assign_genre(t, default_genre)
                        key = (t.strip(), g)
                        if key not in seen:
                            seen.add(key)
                            all_sentences.append({"text": t.strip(), "genre_id": g})
                except (json.JSONDecodeError, TypeError):
                    continue
        else:
            for sent in _sentence_split(text):
                if not sent or len(sent) < 3:
                    continue
                g = _assign_genre(sent, default_genre)
                key = (sent, g)
                if key not in seen:
                    seen.add(key)
                    all_sentences.append({"text": sent, "genre_id": g})

    for i, rec in enumerate(all_sentences):
        rec["sentence_id"] = i

    with open(sentences_path, "w", encoding="utf-8") as f:
        for rec in all_sentences:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if also_write_per_genre:
        by_genre: dict[str, list[dict]] = {}
        for rec in all_sentences:
            g = rec["genre_id"]
            by_genre.setdefault(g, []).append(rec)
        for genre_id, recs in by_genre.items():
            genre_dir = output_dir / genre_id
            genre_dir.mkdir(parents=True, exist_ok=True)
            genre_path = genre_dir / "genre_sentences.jsonl"
            with open(genre_path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    return len(all_sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined corpus with genre tags (Option C).")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .txt or .jsonl files to process",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output root (default: data); writes corpus/sentences.jsonl and optionally <genre>/genre_sentences.jsonl",
    )
    parser.add_argument(
        "--default-genre",
        default="general",
        help="Default genre_id when not inferred (default: general)",
    )
    parser.add_argument(
        "--no-per-genre",
        action="store_true",
        help="Do not write per-genre genre_sentences.jsonl",
    )
    args = parser.parse_args()
    n = build_from_directory(
        args.input_dir,
        args.output_dir,
        default_genre=args.default_genre,
        also_write_per_genre=not args.no_per_genre,
    )
    print(f"Wrote {n} sentences to {args.output_dir / 'corpus' / 'sentences.jsonl'}")


if __name__ == "__main__":
    main()
