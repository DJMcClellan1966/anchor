"""
Word/sentence graph for Option C corpus.
Nodes: word IDs and sentence IDs. Edges: co-occurrence, sequence, sentence-word,
sentence-sentence similarity. Persists to JSON for fast load and query.
By-product: transition matrix (corpus_model.json) for 1-layer LM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _merge_token_sequence_into_graph(
    token_ids: list[int],
    word_cooccurrence: dict[int, set[int]],
    word_next: dict[int, dict[int, int]],
) -> None:
    """Update word_cooccurrence and word_next from a single token ID sequence (e.g. a definition)."""
    if not token_ids:
        return
    for i, wid in enumerate(token_ids):
        word_cooccurrence.setdefault(wid, set())
        for w in token_ids:
            if w != wid:
                word_cooccurrence[wid].add(w)
        if i + 1 < len(token_ids):
            nxt = token_ids[i + 1]
            word_next.setdefault(wid, {})
            word_next[wid][nxt] = word_next[wid].get(nxt, 0) + 1


def build_graph(
    encoded_sentences_path: Path,
    top_similar_per_sentence: int = 20,
    context_length: int = 5,
    encoded_dictionary_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build graph from encoded_sentences.jsonl.
    If encoded_dictionary_path is provided, merge definition token sequences into
    word_next and word_cooccurrence (recurring patterns from dictionary + corpus).
    Returns dict with:
      sentence_words: {sentence_id: [token_ids]}
      word_cooccurrence: {word_id: [word_id, ...]} (list of co-occurring word ids)
      word_next: {word_id: {next_word_id: count}}
      sentence_similar: {sentence_id: [[other_sentence_id, jaccard], ...]} top-k
      context_to_sentences: inverted index context_key -> [[sentence_id, position], ...]
    """
    sentence_words: dict[int, list[int]] = {}
    word_cooccurrence: dict[int, set[int]] = {}
    word_next: dict[int, dict[int, int]] = {}
    context_to_sentences: dict[str, list[list[int]]] = {}

    with open(encoded_sentences_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sentence_id", len(sentence_words))
                token_ids = obj.get("token_ids", [])
                if not token_ids:
                    continue
                sentence_words[sid] = token_ids

                # Inverted index: for each position i >= context_length, record (sid, i)
                k = context_length
                for i in range(k, len(token_ids)):
                    ctx = token_ids[i - k : i]
                    key = ",".join(str(t) for t in ctx)
                    context_to_sentences.setdefault(key, []).append([sid, i])

                # Co-occurrence: all pairs in this sentence
                for i, wid in enumerate(token_ids):
                    word_cooccurrence.setdefault(wid, set())
                    for w in token_ids:
                        if w != wid:
                            word_cooccurrence[wid].add(w)
                    # Next-word
                    if i + 1 < len(token_ids):
                        nxt = token_ids[i + 1]
                        word_next.setdefault(wid, {})
                        word_next[wid][nxt] = word_next[wid].get(nxt, 0) + 1
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    if encoded_dictionary_path and encoded_dictionary_path.exists():
        with open(encoded_dictionary_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    token_ids = obj.get("token_ids", [])
                    if isinstance(token_ids, list) and token_ids:
                        token_ids = [int(x) for x in token_ids]
                        _merge_token_sequence_into_graph(token_ids, word_cooccurrence, word_next)
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    continue

    # Convert co-occurrence sets to lists for JSON
    word_cooccurrence_list: dict[int, list[int]] = {
        k: list(v) for k, v in word_cooccurrence.items()
    }

    # Sentence-sentence similarity (Jaccard on token sets)
    sentence_sets = {sid: set(tids) for sid, tids in sentence_words.items()}
    sentence_similar: dict[int, list[list[float | int]]] = {}
    sids = list(sentence_sets.keys())
    for i, sid in enumerate(sids):
        a = sentence_sets[sid]
        candidates = []
        for j, other in enumerate(sids):
            if other == sid:
                continue
            score = _jaccard(a, sentence_sets[other])
            if score > 0:
                candidates.append((other, score))
        candidates.sort(key=lambda x: -x[1])
        sentence_similar[sid] = [
            [int(c[0]), float(c[1])]
            for c in candidates[:top_similar_per_sentence]
        ]

    # JSON keys must be strings
    return {
        "sentence_words": {str(k): v for k, v in sentence_words.items()},
        "word_cooccurrence": {str(k): v for k, v in word_cooccurrence_list.items()},
        "word_next": {
            str(k): {str(nk): nv for nk, nv in v.items()}
            for k, v in word_next.items()
        },
        "sentence_similar": {str(k): v for k, v in sentence_similar.items()},
        "context_to_sentences": {
            "key_format": "comma_separated_ids",
            "context_length": context_length,
            "index": context_to_sentences,
        },
    }


def save_graph(graph: dict[str, Any], path: Path) -> None:
    """Write graph to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)


def load_graph(path: Path) -> dict[str, Any] | None:
    """Load graph from JSON. Returns None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class CorpusGraph:
    """
    Minimal interface for retrieval and next-sentence: neighbors, sentence similarity, next-word stats.
    Keys in stored graph are strings; this class exposes int keys for sentence_id and word_id.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._sentence_words = {int(k): v for k, v in data.get("sentence_words", {}).items()}
        self._word_cooccurrence = {int(k): v for k, v in data.get("word_cooccurrence", {}).items()}
        self._word_next = {
            int(k): {int(nk): nv for nk, nv in v.items()}
            for k, v in data.get("word_next", {}).items()
        }
        self._sentence_similar = {
            int(k): [[int(p[0]), float(p[1])] for p in v]
            for k, v in data.get("sentence_similar", {}).items()
        }
        ctx_data = data.get("context_to_sentences") or {}
        self._context_length = int(ctx_data.get("context_length", 5))
        self._context_index = dict(ctx_data.get("index", {}))

    def has_context_index(self) -> bool:
        """True if the graph has an inverted index for context -> sentences."""
        return len(self._context_index) > 0

    def sentences_containing_word(self, word_id: int) -> list[int]:
        """Sentence IDs that contain this word."""
        return [
            sid for sid, tids in self._sentence_words.items()
            if word_id in tids
        ]

    def similar_sentences(self, sentence_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (sentence_id, jaccard) for sentences similar to the given one."""
        pairs = self._sentence_similar.get(sentence_id, [])[:top_k]
        return [(int(p[0]), float(p[1])) for p in pairs]

    def cooccurring_words(self, word_id: int) -> list[int]:
        """Word IDs that co-occur with this word in any sentence."""
        return list(self._word_cooccurrence.get(word_id, []))

    def next_word_counts(self, word_id: int) -> dict[int, int]:
        """For a word_id, dict of next_word_id -> count."""
        return dict(self._word_next.get(word_id, {}))

    def sentence_token_ids(self, sentence_id: int) -> list[int]:
        """Token ID sequence for a sentence."""
        return list(self._sentence_words.get(sentence_id, []))

    def sentence_ids(self) -> list[int]:
        """All sentence IDs in the graph."""
        return list(self._sentence_words.keys())

    def get_sentences_with_context(self, context_token_ids: list[int]) -> list[tuple[int, int]]:
        """
        Return (sentence_id, position) pairs where the sentence contains this context
        at that position. context_token_ids should have length context_length (uses last K).
        """
        if not self._context_index:
            return []
        k = self._context_length
        if len(context_token_ids) < k:
            return []
        ctx = context_token_ids[-k:]
        key = ",".join(str(t) for t in ctx)
        raw = self._context_index.get(key, [])
        return [(int(p[0]), int(p[1])) for p in raw]


def load_corpus_graph(data_dir: Path, graph_path: str = "corpus/graph.json") -> CorpusGraph | None:
    """Load graph from data_dir/graph_path; return CorpusGraph or None."""
    path = data_dir / graph_path
    data = load_graph(path)
    if data is None:
        return None
    return CorpusGraph(data)


def build_transition_matrix(
    word_next: dict[int, dict[int, int]],
    vocab_size: int,
    smoothing: float = 0.01,
) -> dict[int, dict[int, float]]:
    """
    Build row-normalized log P(next|curr) from word_next counts with Laplace smoothing.
    Returns sparse dict: curr_id -> {next_id: log_prob}.
    """
    result: dict[int, dict[int, float]] = {}
    for curr, next_counts in word_next.items():
        total = sum(next_counts.values()) + vocab_size * smoothing
        result[curr] = {
            nxt: math.log((c + smoothing) / total)
            for nxt, c in next_counts.items()
        }
    return result


def export_corpus_model(
    graph_dict: dict[str, Any],
    vocab_path: Path,
    output_path: Path,
    smoothing: float = 0.01,
) -> None:
    """
    Build transition matrix from graph word_next and vocab, write corpus_model.json.
    Format: {"vocab_size": N, "id_to_word": {"0": "word", ...}, "transition": {"curr_id": {"next_id": log_prob}, ...}}.
    """
    from .corpus_vocab import load_vocab
    word_to_id, id_to_word = load_vocab(vocab_path)
    vocab_size = len(word_to_id)
    raw_next = graph_dict.get("word_next", {})
    word_next_int: dict[int, dict[int, int]] = {
        int(k): {int(nk): nv for nk, nv in v.items()}
        for k, v in raw_next.items()
    }
    transition = build_transition_matrix(word_next_int, vocab_size, smoothing=smoothing)
    id_to_word_str = {str(k): v for k, v in id_to_word.items()}
    transition_str: dict[str, dict[str, float]] = {
        str(k): {str(nk): v for nk, v in row.items()}
        for k, row in transition.items()
    }
    payload = {
        "vocab_size": vocab_size,
        "id_to_word": id_to_word_str,
        "transition": transition_str,
        "smoothing": smoothing,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
