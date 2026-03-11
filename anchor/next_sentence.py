"""
Next-sentence prediction (retrieval-based) for Option C corpus.
Uses the word/sentence graph to find similar sentences, genre-filtered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_graph import CorpusGraph


def _load_encoded_index(encoded_path: Path) -> dict[int, dict[str, Any]]:
    """Load encoded_sentences.jsonl into sentence_id -> {genre_id, text, token_ids}."""
    index: dict[int, dict[str, Any]] = {}
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
                index[sid] = {
                    "genre_id": obj.get("genre_id", "general"),
                    "text": obj.get("text", ""),
                    "token_ids": obj.get("token_ids", []),
                }
            except (json.JSONDecodeError, TypeError):
                continue
    return index


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_next_sentences(
    current_sentence_ids: list[int],
    genre_id: str,
    graph: CorpusGraph,
    encoded_path: Path,
    top_k: int = 5,
) -> list[str]:
    """
    Retrieval-based next sentences: find the graph sentence most similar to
    current_sentence_ids, then return similar sentences filtered by genre_id.
    Returns list of sentence texts.
    """
    index = _load_encoded_index(encoded_path)
    if not index:
        return []
    current_set = set(current_sentence_ids)
    best_sid: int | None = None
    best_score = -1.0
    for sid in graph.sentence_ids():
        tids = graph.sentence_token_ids(sid)
        if not tids:
            continue
        score = _jaccard(current_set, set(tids))
        if score > best_score:
            best_score = score
            best_sid = sid
    if best_sid is None:
        return []
    similar = graph.similar_sentences(best_sid, top_k=top_k * 2)
    out: list[str] = []
    for sid, _ in similar:
        rec = index.get(sid)
        if not rec or rec.get("genre_id") != genre_id:
            continue
        text = (rec.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= top_k:
            break
    return out


def get_next_sentences_from_text(
    current_text: str,
    genre_id: str,
    graph: CorpusGraph,
    encoded_path: Path,
    word_to_id: dict[str, int],
    top_k: int = 5,
) -> list[str]:
    """
    Same as get_next_sentences but takes current sentence as text; tokenizes and
    maps to IDs using word_to_id (unknown tokens get a single id if present, else skip).
    """
    from .corpus_vocab import tokenize
    token_ids = [word_to_id[t] for t in tokenize(current_text) if t in word_to_id]
    if not token_ids:
        return []
    return get_next_sentences(token_ids, genre_id, graph, encoded_path, top_k=top_k)
