"""
Vocabulary (word/punctuation list) and sentence encoding for Option C corpus.
Builds a single deduplicated vocab from dictionary terms + corpus, assigns IDs,
encodes sentences as token ID sequences. Persists vocab and encoded_sentences.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Tokenization aligned with critic: words [a-zA-Z][a-zA-Z0-9]*, rest as punctuation runs
_TOKENIZE_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*|[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens."""
    if not (text or "").strip():
        return []
    return [t for t in _TOKENIZE_RE.findall(text) if t]


def build_vocab(
    corpus_sentences_path: Path,
    dictionary_terms: list[str] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build word_to_id and id_to_word from dictionary terms (first) then corpus tokens.
    Deduplicates; dictionary terms get stable low IDs when provided first.
    """
    word_to_id: dict[str, int] = {}
    # Reserve 0 for padding/unknown if needed
    next_id = 0

    if dictionary_terms:
        for w in dictionary_terms:
            w = (w or "").strip()
            if w and w not in word_to_id:
                word_to_id[w] = next_id
                next_id += 1

    if not corpus_sentences_path.exists():
        id_to_word = {v: k for k, v in word_to_id.items()}
        return word_to_id, id_to_word

    with open(corpus_sentences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str):
                    continue
                for t in tokenize(text):
                    if t not in word_to_id:
                        word_to_id[t] = next_id
                        next_id += 1
            except (json.JSONDecodeError, TypeError):
                continue

    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def encode_sentences(
    corpus_sentences_path: Path,
    word_to_id: dict[str, int],
    encoded_output_path: Path,
    unk_id: int | None = None,
) -> int:
    """
    Encode each sentence in corpus to token_ids; write encoded_sentences.jsonl.
    Each line: {"sentence_id": i, "genre_id": "...", "token_ids": [...], "text": "..."}.
    Returns number of sentences encoded.
    """
    if unk_id is None and word_to_id:
        unk_id = max(word_to_id.values()) + 1
    elif unk_id is None:
        unk_id = 0

    count = 0
    with open(corpus_sentences_path, encoding="utf-8") as fin, open(
        encoded_output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") if isinstance(obj, dict) else None
                genre_id = obj.get("genre_id", "general") if isinstance(obj, dict) else "general"
                sentence_id = obj.get("sentence_id", count) if isinstance(obj, dict) else count
                if not isinstance(text, str):
                    continue
                token_ids = [
                    word_to_id.get(t, unk_id)
                    for t in tokenize(text)
                ]
                out = {
                    "sentence_id": sentence_id,
                    "genre_id": genre_id,
                    "token_ids": token_ids,
                    "text": text,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, TypeError):
                continue
    return count


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load word_to_id and id_to_word from vocab.json. Format: {"id_to_word": {"0": "word", ...}} or {"word_to_id": {"word": 0, ...}}."""
    word_to_id = {}
    id_to_word = {}
    if not vocab_path.exists():
        return word_to_id, id_to_word
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "id_to_word" in data:
            id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
            word_to_id = {v: k for k, v in id_to_word.items()}
        elif "word_to_id" in data:
            word_to_id = data["word_to_id"]
            id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def run_build(
    data_dir: Path,
    corpus_sentences_file: str = "corpus/sentences.jsonl",
    vocab_file: str = "corpus/vocab.json",
    encoded_file: str = "corpus/encoded_sentences.jsonl",
    dictionary_terms: list[str] | None = None,
) -> tuple[int, int]:
    """
    Build vocab from dictionary_terms + corpus, then encode sentences.
    Writes data_dir/corpus/vocab.json and data_dir/corpus/encoded_sentences.jsonl.
    Returns (vocab_size, num_encoded_sentences).
    """
    corpus_path = data_dir / corpus_sentences_file
    corpus_dir = corpus_path.parent
    corpus_dir.mkdir(parents=True, exist_ok=True)

    word_to_id, id_to_word = build_vocab(corpus_path, dictionary_terms=dictionary_terms)
    vocab_path = data_dir / vocab_file
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"word_to_id": word_to_id, "id_to_word": {str(k): v for k, v in id_to_word.items()}}, f, ensure_ascii=False)

    n_encoded = encode_sentences(corpus_path, word_to_id, data_dir / encoded_file)
    return len(word_to_id), n_encoded
