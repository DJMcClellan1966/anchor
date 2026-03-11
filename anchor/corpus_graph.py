"""
Word/sentence graph for Option C corpus.
Nodes: word IDs and sentence IDs. Edges: co-occurrence, sequence, sentence-word,
sentence-sentence similarity. Persists to JSON for fast load and query.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_graph(
    encoded_sentences_path: Path,
    top_similar_per_sentence: int = 20,
) -> dict[str, Any]:
    """
    Build graph from encoded_sentences.jsonl.
    Returns dict with:
      sentence_words: {sentence_id: [token_ids]}
      word_cooccurrence: {word_id: [word_id, ...]} (list of co-occurring word ids)
      word_next: {word_id: {next_word_id: count}}
      sentence_similar: {sentence_id: [[other_sentence_id, jaccard], ...]} top-k
    """
    sentence_words: dict[int, list[int]] = {}
    word_cooccurrence: dict[int, set[int]] = {}
    word_next: dict[int, dict[int, int]] = {}

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


def load_corpus_graph(data_dir: Path, graph_path: str = "corpus/graph.json") -> CorpusGraph | None:
    """Load graph from data_dir/graph_path; return CorpusGraph or None."""
    path = data_dir / graph_path
    data = load_graph(path)
    if data is None:
        return None
    return CorpusGraph(data)
