"""
Naturalize: extend response using recurring patterns (word_next) for more natural language.
Optional post-step after generation; uses graph + vocab to sample next tokens from the tail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .corpus_graph import load_corpus_graph
from .corpus_vocab import load_vocab, tokenize
from .next_token import get_bigram_distribution, sample_next_token


def naturalize(
    response_text: str,
    data_dir: Path,
    config: dict[str, Any],
) -> str:
    """
    Optionally extend response tail using graph next-token distribution.
    If use_naturalize is false or graph/vocab missing, return response_text unchanged.
    """
    if not config.get("use_naturalize", False):
        return response_text
    data_path = Path(data_dir)
    graph_path = data_path / "corpus" / "graph.json"
    vocab_path = data_path / "corpus" / "vocab.json"
    if not graph_path.exists() or not vocab_path.exists():
        return response_text

    graph = load_corpus_graph(data_path)
    if graph is None:
        return response_text
    word_to_id, id_to_word = load_vocab(vocab_path)
    if not word_to_id or not id_to_word:
        return response_text

    max_tokens = int(config.get("naturalize_max_tokens", 12))
    context_length = int(config.get("naturalize_context_length", 5))
    if max_tokens <= 0:
        return response_text

    tokens = tokenize(response_text)
    token_ids = [word_to_id[t] for t in tokens if t in word_to_id]
    if not token_ids:
        return response_text

    vocab_size = len(word_to_id)
    extended = list(token_ids)
    stop_ids = {word_to_id[t] for t in (".", "!", "?") if t in word_to_id}

    for _ in range(max_tokens):
        context = extended[-context_length:] if len(extended) >= context_length else extended
        curr = context[-1]
        dist = get_bigram_distribution(curr, graph, vocab_size, smoothing=0.01)
        if not dist:
            break
        next_id = sample_next_token(dist)
        if next_id is None:
            break
        extended.append(next_id)
        if next_id in stop_ids:
            break

    words = [id_to_word.get(i, "") for i in extended]
    return " ".join(w for w in words if w).strip() or response_text
