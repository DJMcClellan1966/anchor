"""
Load word vectors built from the corpus graph and optionally boost sentence visit scores
by query–sentence vector similarity (Graph LLM vector geometry).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_word_vectors(path: Path) -> dict[int, list[float]] | None:
    """Load word_vectors.json. Returns word_id -> vector (list of float) or None if missing/invalid."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    wv = data.get("word_vectors") or data.get("vectors")
    if not isinstance(wv, dict):
        return None
    return {int(k): list(map(float, v)) for k, v in wv.items() if isinstance(v, list)}


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0 if either norm is 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _mean_vector(vectors: list[list[float]]) -> list[float] | None:
    """Element-wise mean; None if empty or length mismatch."""
    if not vectors:
        return None
    n = len(vectors[0])
    if any(len(v) != n for v in vectors):
        return None
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(n)]


def boost_sentence_visits_by_vectors(
    sentence_visits: dict[int, float],
    word_vectors: dict[int, list[float]],
    concept_bundle: dict[str, Any],
    word_to_id: dict[str, int],
    sentence_token_ids_fn: Any,
    boost: float = 0.5,
) -> None:
    """
    In-place boost sentence_visits[sid] by query–sentence cosine similarity.
    query vector = mean of concept term vectors; sentence vector = mean of token vectors.
    sentence_token_ids_fn(sid) returns list of word ids for that sentence.
    """
    terms = (concept_bundle.get("terms") or [])[:20]
    query_wids = [word_to_id[t] for t in terms if t in word_to_id]
    query_vecs = [word_vectors[w] for w in query_wids if w in word_vectors]
    query_vec = _mean_vector(query_vecs)
    if not query_vec:
        return
    for sid in list(sentence_visits.keys()):
        token_ids = sentence_token_ids_fn(sid)
        sent_vecs = [word_vectors[w] for w in token_ids if w in word_vectors]
        sent_vec = _mean_vector(sent_vecs)
        if not sent_vec:
            continue
        sim = _cosine(query_vec, sent_vec)
        if sim > 0:
            sentence_visits[sid] = sentence_visits.get(sid, 0) + boost * sim
