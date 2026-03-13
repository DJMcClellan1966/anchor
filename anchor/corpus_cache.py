"""
Cache for corpus data (graph, vocab, encoded index) to avoid reloading on every query.
Keyed by data_path and file mtimes; invalidates when graph/vocab/encoded files change.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .corpus_graph import CorpusGraph, load_corpus_graph
from .corpus_vocab import load_vocab

_lock = threading.Lock()
_cache: dict[tuple[Any, ...], tuple[CorpusGraph, dict[str, int], dict[int, str], dict[int, dict[str, Any]]]] = {}


def _load_encoded_index(encoded_path: Path) -> dict[int, dict[str, Any]]:
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
                    "text": obj.get("text", ""),
                    "genre_id": obj.get("genre_id", "general"),
                    "source": obj.get("source"),
                    "term": obj.get("term"),
                }
            except (json.JSONDecodeError, TypeError):
                continue
    return index


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime if p.exists() else 0.0
    except OSError:
        return 0.0


def get_cached_corpus_data(
    data_path: Path,
) -> tuple[CorpusGraph, dict[str, int], dict[int, str], dict[int, dict[str, Any]]] | None:
    """
    Return (graph, word_to_id, id_to_word, encoded_index) for data_path, loading once per
    (path, mtimes) and reusing. Returns None if graph or vocab missing/invalid.
    """
    data_path = Path(data_path).resolve()
    graph_path = data_path / "corpus" / "graph.json"
    vocab_path = data_path / "corpus" / "vocab.json"
    encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
    key = (str(data_path), _mtime(graph_path), _mtime(vocab_path), _mtime(encoded_path))

    with _lock:
        if key in _cache:
            return _cache[key]

    graph = load_corpus_graph(data_path)
    if graph is None:
        return None
    word_to_id, id_to_word = load_vocab(vocab_path)
    if not word_to_id:
        return None
    encoded_index = _load_encoded_index(encoded_path)
    if not encoded_index:
        return None

    with _lock:
        _cache[key] = (graph, word_to_id, id_to_word, encoded_index)
    return (graph, word_to_id, id_to_word, encoded_index)
