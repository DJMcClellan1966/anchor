"""
Build feedback_weights.json from feedback.jsonl (accepted responses).
Maps query key -> list of sentence_ids that appeared in the accepted response.
Run with data_dir containing corpus/encoded_sentences.jsonl and feedback.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))


def _load_encoded_index(encoded_path: Path) -> dict[int, str]:
    """sentence_id -> text."""
    index: dict[int, str] = {}
    if not encoded_path.exists():
        return index
    with open(encoded_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sentence_id", len(index))
                text = (obj.get("text") or "").strip()
                if text:
                    index[sid] = text
            except (json.JSONDecodeError, TypeError):
                continue
    return index


def _sentence_ids_in_response(response: str, sid_to_text: dict[int, str]) -> list[int]:
    """Find sentence_ids whose text appears in the response (as substring or line)."""
    if not response or not sid_to_text:
        return []
    response_lower = response.strip().lower()
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    found: list[int] = []
    for sid, text in sid_to_text.items():
        if not text:
            continue
        text_lower = text.lower()
        if text_lower in response_lower or any(text_lower in ln for ln in lines):
            found.append(sid)
        elif any(ln in text_lower for ln in lines if len(ln) > 10):
            found.append(sid)
    return found


def build_feedback_weights(
    feedback_path: Path,
    encoded_path: Path,
    output_path: Path,
) -> None:
    """Read feedback.jsonl (accepted only), infer sentence_ids from response, write feedback_weights.json."""
    sid_to_text = _load_encoded_index(encoded_path)
    weights: dict[str, list[int]] = {}
    if not feedback_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False)
        return
    with open(feedback_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not rec.get("accepted"):
                    continue
                query = (rec.get("query") or "").strip()
                response = (rec.get("response") or "").strip()
                key = query.lower()[:200]
                sids = _sentence_ids_in_response(response, sid_to_text)
                if not sids:
                    continue
                existing = set(weights.get(key, []))
                existing.update(sids)
                weights[key] = sorted(existing)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feedback_weights.json from feedback.jsonl for Graph LLM adaptation."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--feedback-file",
        default="feedback.jsonl",
        help="Feedback log relative to data_dir (default: feedback.jsonl)",
    )
    parser.add_argument(
        "--encoded-file",
        default="corpus/encoded_sentences.jsonl",
        help="Encoded sentences relative to data_dir (default: corpus/encoded_sentences.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="feedback_weights.json",
        help="Output file relative to data_dir (default: feedback_weights.json)",
    )
    args = parser.parse_args()

    feedback_path = args.data_dir / args.feedback_file
    encoded_path = args.data_dir / args.encoded_file
    output_path = args.data_dir / args.output

    if not encoded_path.exists():
        print(f"Missing {encoded_path}. Run build_vocab.py first.", file=sys.stderr)
        raise SystemExit(1)

    build_feedback_weights(feedback_path, encoded_path, output_path)
    print(f"Wrote feedback weights to {output_path}")


if __name__ == "__main__":
    main()
