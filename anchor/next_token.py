"""
Next-token prediction: retrieval-based distribution, bigram fallback, and hybrid mixture.
Uses the corpus graph inverted index (context -> sentences) and word_next counts.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .corpus_graph import CorpusGraph


def _load_encoded_index(encoded_path: Path) -> dict[int, dict[str, Any]]:
    """Load encoded_sentences.jsonl into sentence_id -> {genre_id, token_ids}."""
    index: dict[int, dict[str, Any]] = {}
    if not encoded_path.exists():
        return index
    import json
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
                    "token_ids": obj.get("token_ids", []),
                }
            except (json.JSONDecodeError, TypeError):
                continue
    return index


def get_next_token_distribution(
    context_token_ids: list[int],
    graph: CorpusGraph,
    encoded_path: Path,
    genre_id: str | None = None,
    min_hits: int = 2,
) -> dict[int, float] | None:
    """
    Return P(next_token_id) from retrieval: find sentences containing this context,
    collect the actual next token at each position, normalize to distribution.
    Returns None if fewer than min_hits (caller should fall back to bigram).
    """
    pairs = graph.get_sentences_with_context(context_token_ids)
    if not pairs:
        return None
    index = _load_encoded_index(encoded_path)
    if genre_id is not None and index:
        pairs = [(sid, pos) for sid, pos in pairs if index.get(sid, {}).get("genre_id") == genre_id]
    if len(pairs) < min_hits:
        return None
    counts: dict[int, int] = {}
    for sid, pos in pairs:
        tids = graph.sentence_token_ids(sid)
        if pos < len(tids):
            nxt = tids[pos]
            counts[nxt] = counts.get(nxt, 0) + 1
    if not counts:
        return None
    total = sum(counts.values())
    return {tid: c / total for tid, c in counts.items()}


def get_bigram_distribution(
    curr_token_id: int,
    graph: CorpusGraph,
    vocab_size: int,
    smoothing: float = 0.01,
) -> dict[int, float]:
    """
    Return P(next|curr) from graph word_next counts with Laplace smoothing.
    Returns only observed next tokens; probs are normalized to sum to 1.
    """
    next_counts = graph.next_word_counts(curr_token_id)
    if not next_counts:
        return {}
    total = sum(next_counts.values()) + len(next_counts) * smoothing
    return {nxt: (c + smoothing) / total for nxt, c in next_counts.items()}


def get_hybrid_next_token_distribution(
    context_token_ids: list[int],
    graph: CorpusGraph,
    encoded_path: Path,
    vocab_size: int,
    genre_id: str | None = None,
    beta: float = 0.7,
    min_retrieval_hits: int = 2,
    bigram_smoothing: float = 0.01,
) -> dict[int, float]:
    """
    Mixture: beta * P_retrieval + (1-beta) * P_bigram. Uses last token of context for bigram.
    If retrieval has insufficient support, returns bigram only.
    """
    if not context_token_ids:
        return {}
    last_id = context_token_ids[-1]
    p_bigram = get_bigram_distribution(last_id, graph, vocab_size, smoothing=bigram_smoothing)
    if not p_bigram:
        return {}

    p_ret = get_next_token_distribution(
        context_token_ids,
        graph,
        encoded_path,
        genre_id=genre_id,
        min_hits=min_retrieval_hits,
    )
    if p_ret is None:
        return p_bigram

    all_ids = set(p_bigram) | set(p_ret)
    mixed: dict[int, float] = {}
    for tid in all_ids:
        mixed[tid] = beta * p_ret.get(tid, 0.0) + (1 - beta) * p_bigram.get(tid, 0.0)
    total = sum(mixed.values())
    if total <= 0:
        return p_bigram
    return {tid: p / total for tid, p in mixed.items()}


def sample_next_token(
    distribution: dict[int, float],
    temperature: float = 1.0,
    rng: Any = None,
) -> int | None:
    """
    Sample one token ID from the distribution. temperature=1.0 is unchanged;
    <1 sharpens, >1 flattens. Uses optional rng (default random).
    """
    if not distribution:
        return None
    if rng is None:
        import random
        rng = random
    ids = list(distribution.keys())
    probs = [distribution[i] for i in ids]
    if temperature != 1.0 and temperature > 0:
        logits = [math.log(p + 1e-10) / temperature for p in probs]
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        total = sum(exp_l)
        probs = [e / total for e in exp_l]
    return rng.choices(ids, weights=probs, k=1)[0]
