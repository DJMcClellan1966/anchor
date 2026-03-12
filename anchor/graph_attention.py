"""
Graph attention loop: query lights up the graph, traverse loops with attention,
detect repeating pattern, refine answer from pattern (grounded, non-hallucinatory).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_graph import CorpusGraph, load_corpus_graph
from .corpus_vocab import load_vocab
from . import retrieval


def _load_encoded_index(encoded_path: Path) -> dict[int, dict[str, Any]]:
    """Load encoded_sentences.jsonl into sentence_id -> {text, genre_id}."""
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
                }
            except (json.JSONDecodeError, TypeError):
                continue
    return index


def activate(
    concept_bundle: dict[str, Any],
    graph: CorpusGraph,
    word_to_id: dict[str, int],
) -> tuple[set[int], set[int]]:
    """
    Activate nodes from query: map concept terms to word_ids, then to sentence_ids
    that contain those words. Returns (activated_word_ids, activated_sentence_ids).
    """
    terms = concept_bundle.get("terms") or []
    activated_word_ids: set[int] = set()
    activated_sentence_ids: set[int] = set()
    for term in terms:
        term = (term or "").strip()
        if not term or term not in word_to_id:
            continue
        wid = word_to_id[term]
        activated_word_ids.add(wid)
        for sid in graph.sentences_containing_word(wid):
            activated_sentence_ids.add(sid)
    if not activated_word_ids and not activated_sentence_ids:
        # No terms in vocab: activate all sentences so we still have a path
        activated_sentence_ids = set(graph.sentence_ids())
    return activated_word_ids, activated_sentence_ids


def traverse_loops(
    activated_word_ids: set[int],
    activated_sentence_ids: set[int],
    graph: CorpusGraph,
    num_hops: int = 2,
    genre_id: str | None = None,
    encoded_index: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Propagate attention over the graph. Returns (word_visits, sentence_visits)
    as weighted visit counts. Genre filter applied when encoded_index and genre_id
    are provided (only count sentences that exist in index with that genre).
    """
    word_visits: dict[int, float] = {wid: 1.0 for wid in activated_word_ids}
    sentence_visits: dict[int, float] = {sid: 1.0 for sid in activated_sentence_ids}

    def sentence_ok(sid: int) -> bool:
        if encoded_index is None or genre_id is None:
            return True
        rec = encoded_index.get(sid)
        return rec is not None and rec.get("genre_id") == genre_id

    for _ in range(max(0, num_hops - 1)):
        w_copy = dict(word_visits)
        s_copy = dict(sentence_visits)
        for word_id, weight in w_copy.items():
            for sid in graph.sentences_containing_word(word_id):
                if sentence_ok(sid):
                    sentence_visits[sid] = sentence_visits.get(sid, 0.0) + weight
        for sentence_id, weight in s_copy.items():
            if not sentence_ok(sentence_id):
                continue
            for sid2, jaccard in graph.similar_sentences(sentence_id, top_k=10):
                if sentence_ok(sid2):
                    sentence_visits[sid2] = sentence_visits.get(sid2, 0.0) + weight * jaccard
            for wid in graph.sentence_token_ids(sentence_id):
                word_visits[wid] = word_visits.get(wid, 0.0) + weight

    return word_visits, sentence_visits


def detect_pattern(
    word_visits: dict[int, float],
    sentence_visits: dict[int, float],
    top_k: int = 10,
    min_visits: float | None = None,
) -> tuple[list[int], list[int]]:
    """
    Return top word_ids and top sentence_ids by visit count (pattern group).
    If min_visits is set, filter to nodes with count >= min_visits.
    """
    w_sorted = sorted(word_visits.items(), key=lambda x: -x[1])
    s_sorted = sorted(sentence_visits.items(), key=lambda x: -x[1])
    if min_visits is not None:
        w_sorted = [(wid, c) for wid, c in w_sorted if c >= min_visits]
        s_sorted = [(sid, c) for sid, c in s_sorted if c >= min_visits]
    top_word_ids = [wid for wid, _ in w_sorted[:top_k]]
    top_sentence_ids = [sid for sid, _ in s_sorted[:top_k]]
    return top_word_ids, top_sentence_ids


def refine_answer(
    pattern_word_ids: list[int],
    pattern_sentence_ids: list[int],
    concept_bundle: dict[str, Any],
    encoded_index: dict[int, dict[str, Any]],
    id_to_word: dict[int, str],
    genre_id: str = "general",
    max_sentences: int = 15,
    max_definitions: int = 10,
) -> str:
    """
    Build response from pattern: definitions for pattern keywords (from concept_bundle)
    and sentence texts (from encoded index, genre-filtered). Grounded only.
    """
    terms_set = set((concept_bundle.get("terms") or []))
    definitions = concept_bundle.get("definitions") or {}
    parts: list[str] = []
    seen_defs: set[str] = set()
    for wid in pattern_word_ids:
        term = id_to_word.get(wid)
        if term is None or term not in terms_set:
            continue
        if term in seen_defs:
            continue
        seen_defs.add(term)
        defn = definitions.get(term)
        if isinstance(defn, str) and defn.strip():
            parts.append(f"{term}: {defn.strip()[:500]}")
        elif isinstance(defn, list) and defn and isinstance(defn[0], str):
            parts.append(f"{term}: {defn[0].strip()[:500]}")
        if len(parts) >= max_definitions:
            break
    seen_sents: set[str] = set()
    for sid in pattern_sentence_ids:
        rec = encoded_index.get(sid)
        if not rec or rec.get("genre_id") != genre_id:
            continue
        text = (rec.get("text") or "").strip()
        if text and text not in seen_sents:
            seen_sents.add(text)
            parts.append(text)
        if len(parts) >= max_definitions + max_sentences:
            break
    if not parts:
        if terms_set:
            return "Concepts: " + ", ".join(list(terms_set)[:15])
        return "No pattern found for this query."
    return "\n".join(parts[: max_definitions + max_sentences])


def run(
    query: str,
    engine: Any,
    config: dict[str, Any],
    data_path: Path | None = None,
) -> str | None:
    """
    Entrypoint: get concept bundle, load graph/vocab/encoded index, activate,
    traverse, detect pattern, refine answer. Returns response text or None if data missing.
    If engine is None, uses wire.get_engine() so generator can call without passing engine.
    """
    if not data_path or not data_path.exists():
        return None
    if engine is None:
        from . import wire
        engine = wire.get_engine()
    concept_bundle = retrieval.get_concept_bundle(engine, query)
    graph = load_corpus_graph(data_path)
    if graph is None:
        return None
    vocab_path = data_path / "corpus" / "vocab.json"
    word_to_id, id_to_word = load_vocab(vocab_path)
    if not word_to_id:
        return None
    encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
    encoded_index = _load_encoded_index(encoded_path)
    if not encoded_index:
        return None

    genre_id = config.get("default_genre_id", "general")
    num_hops = int(config.get("attention_loop_hops", 2))
    top_k = int(config.get("attention_loop_top_k", 10))

    activated_word_ids, activated_sentence_ids = activate(
        concept_bundle, graph, word_to_id
    )
    word_visits, sentence_visits = traverse_loops(
        activated_word_ids,
        activated_sentence_ids,
        graph,
        num_hops=num_hops,
        genre_id=genre_id,
        encoded_index=encoded_index,
    )
    pattern_word_ids, pattern_sentence_ids = detect_pattern(
        word_visits, sentence_visits, top_k=top_k
    )
    return refine_answer(
        pattern_word_ids,
        pattern_sentence_ids,
        concept_bundle,
        encoded_index,
        id_to_word,
        genre_id=genre_id,
    )
