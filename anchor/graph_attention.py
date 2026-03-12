"""
Graph attention loop: query lights up the graph, traverse loops with attention,
detect repeating pattern, refine answer from pattern (grounded, non-hallucinatory).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_graph import CorpusGraph, load_corpus_graph
from .corpus_vocab import load_vocab, tokenize
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
    query_token_ids: list[int] | None = None,
) -> tuple[set[int], set[int]]:
    """
    Activate nodes from query: map concept terms to word_ids, then to sentence_ids
    that contain those words. When query_token_ids is provided, also activate from
    those word IDs (query-as-numbers). Returns (activated_word_ids, activated_sentence_ids).
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
    if query_token_ids:
        for wid in query_token_ids:
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
    genre_id: str | list[str] | None = None,
    encoded_index: dict[int, dict[str, Any]] | None = None,
    use_weights: bool = True,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Propagate attention over the graph. Returns (word_visits, sentence_visits)
    as weighted visit counts. Genre filter applied when encoded_index and genre_id
    are provided. When use_weights is True: word/sentence contributions are
    normalized by sentence length; word->word propagation uses next_word_counts.
    """
    word_visits: dict[int, float] = {wid: 1.0 for wid in activated_word_ids}
    sentence_visits: dict[int, float] = {sid: 1.0 for sid in activated_sentence_ids}

    allowed_genres: set[str] = set()
    if genre_id is not None:
        allowed_genres = {genre_id} if isinstance(genre_id, str) else set(genre_id)

    def sentence_ok(sid: int) -> bool:
        if encoded_index is None or not allowed_genres:
            return True
        rec = encoded_index.get(sid)
        return rec is not None and rec.get("genre_id") in allowed_genres

    for _ in range(max(0, num_hops - 1)):
        w_copy = dict(word_visits)
        s_copy = dict(sentence_visits)
        # Word -> sentence: weight by 1/len(sentence) when use_weights
        for word_id, weight in w_copy.items():
            for sid in graph.sentences_containing_word(word_id):
                if not sentence_ok(sid):
                    continue
                tokens = graph.sentence_token_ids(sid)
                add = weight / len(tokens) if use_weights and tokens else weight
                sentence_visits[sid] = sentence_visits.get(sid, 0.0) + add
        # Sentence -> sentence (Jaccard) and sentence -> word
        for sentence_id, weight in s_copy.items():
            if not sentence_ok(sentence_id):
                continue
            for sid2, jaccard in graph.similar_sentences(sentence_id, top_k=10):
                if sentence_ok(sid2):
                    sentence_visits[sid2] = sentence_visits.get(sid2, 0.0) + weight * jaccard
            tokens = graph.sentence_token_ids(sentence_id)
            per_word = (weight / len(tokens)) if use_weights and tokens else weight
            for wid in tokens:
                word_visits[wid] = word_visits.get(wid, 0.0) + per_word
        # Word -> word (next-token style): spread by next_word_counts
        if use_weights:
            for word_id, mass in w_copy.items():
                next_counts = graph.next_word_counts(word_id)
                if not next_counts:
                    continue
                total = sum(next_counts.values())
                if total <= 0:
                    continue
                for next_id, count in next_counts.items():
                    word_visits[next_id] = word_visits.get(next_id, 0.0) + mass * (count / total)

    return word_visits, sentence_visits


def detect_pattern(
    word_visits: dict[int, float],
    sentence_visits: dict[int, float],
    top_k: int = 10,
    min_visits: float | None = None,
    num_groups: int = 1,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Return top word_ids and top sentence_ids by visit count (pattern group).
    If min_visits is set, filter to nodes with count >= min_visits.
    When num_groups >= 2, also return secondary_word_ids and secondary_sentence_ids
    (next top_k by visit not in the first group). Always returns 4-tuple.
    """
    w_sorted = sorted(word_visits.items(), key=lambda x: -x[1])
    s_sorted = sorted(sentence_visits.items(), key=lambda x: -x[1])
    if min_visits is not None:
        w_sorted = [(wid, c) for wid, c in w_sorted if c >= min_visits]
        s_sorted = [(sid, c) for sid, c in s_sorted if c >= min_visits]
    top_word_ids = [wid for wid, _ in w_sorted[:top_k]]
    top_sentence_ids = [sid for sid, _ in s_sorted[:top_k]]
    if num_groups < 2:
        return top_word_ids, top_sentence_ids, [], []
    primary_w = set(top_word_ids)
    primary_s = set(top_sentence_ids)
    secondary_w = [wid for wid, _ in w_sorted if wid not in primary_w][:top_k]
    secondary_s = [sid for sid, _ in s_sorted if sid not in primary_s][:top_k]
    return top_word_ids, top_sentence_ids, secondary_w, secondary_s


def refine_answer(
    pattern_word_ids: list[int],
    pattern_sentence_ids: list[int],
    concept_bundle: dict[str, Any],
    encoded_index: dict[int, dict[str, Any]],
    id_to_word: dict[int, str],
    genre_id: str | list[str] = "general",
    max_sentences: int = 15,
    max_definitions: int = 10,
    secondary_word_ids: list[int] | None = None,
    secondary_sentence_ids: list[int] | None = None,
    next_span_sentence_ids: list[int] | None = None,
    sentence_visits: dict[int, float] | None = None,
    output_format: str = "list",
    paragraph_max_chars: int = 500,
    include_definitions: bool = True,
) -> str:
    """
    Build response from pattern: definitions for pattern keywords (from concept_bundle)
    and sentence texts (from encoded index, genre-filtered). Sentences are ordered by
    sentence_visits (desc) when provided; then secondary and next_span (genre-filtered, deduped).
    output_format "list" = newline-separated; "paragraph" = definitions then one fused paragraph.
    When include_definitions is False, only corpus sentences are included (LLM-like answer).
    Grounded only.
    """
    terms_set = set((concept_bundle.get("terms") or []))
    definitions = concept_bundle.get("definitions") or {}
    parts: list[str] = []
    seen_defs: set[str] = set()
    max_total = max_definitions + max_sentences

    def add_definition(wid: int) -> bool:
        term = id_to_word.get(wid)
        if term is None or term not in terms_set or term in seen_defs:
            return False
        seen_defs.add(term)
        defn = definitions.get(term)
        if isinstance(defn, str) and defn.strip():
            parts.append(f"{term}: {defn.strip()[:500]}")
            return True
        if isinstance(defn, list) and defn and isinstance(defn[0], str):
            parts.append(f"{term}: {defn[0].strip()[:500]}")
            return True
        return False

    if include_definitions:
        for wid in pattern_word_ids:
            add_definition(wid)
            if len(parts) >= max_definitions:
                break
        for wid in (secondary_word_ids or []):
            if len(parts) >= max_definitions:
                break
            add_definition(wid)

    # Collect candidate sentence IDs and order by visit score (desc) when available
    all_sids: list[int] = []
    seen_sid: set[int] = set()
    for sid in pattern_sentence_ids:
        if sid not in seen_sid:
            seen_sid.add(sid)
            all_sids.append(sid)
    for sid in (secondary_sentence_ids or []):
        if sid not in seen_sid:
            seen_sid.add(sid)
            all_sids.append(sid)
    for sid in (next_span_sentence_ids or []):
        if sid not in seen_sid:
            seen_sid.add(sid)
            all_sids.append(sid)
    if sentence_visits:
        all_sids.sort(key=lambda sid: -sentence_visits.get(sid, 0.0))

    seen_sents: set[str] = set()
    sentence_texts: list[str] = []
    sent_allowed = {genre_id} if isinstance(genre_id, str) else set(genre_id)
    for sid in all_sids:
        if len(sentence_texts) >= max_sentences:
            break
        rec = encoded_index.get(sid)
        if not rec or rec.get("genre_id") not in sent_allowed:
            continue
        text = (rec.get("text") or "").strip()
        if text and text not in seen_sents:
            seen_sents.add(text)
            sentence_texts.append(text)

    if output_format == "paragraph" and sentence_texts:
        fused = " ".join(sentence_texts)
        if len(fused) > paragraph_max_chars:
            cut = fused[: paragraph_max_chars - 3]
            last_space = cut.rfind(" ")
            fused = (cut[:last_space] if last_space > 0 else cut) + "..."
        parts.append("In the corpus: " + fused)
    else:
        for text in sentence_texts:
            parts.append(text)
            if len(parts) >= max_total:
                break

    if not parts:
        if terms_set:
            return "Concepts: " + ", ".join(list(terms_set)[:15])
        return "No pattern found for this query."
    return "\n".join(parts[:max_total])


def _next_span_sentence_ids(
    top_sentence_ids: list[int],
    graph: CorpusGraph,
    genre_id: str | list[str],
    encoded_index: dict[int, dict[str, Any]],
    top_k_per_sentence: int = 3,
) -> list[int]:
    """Collect sentence IDs one step from top sentences (similar_sentences), genre-filtered, deduped."""
    allowed = {genre_id} if isinstance(genre_id, str) else set(genre_id)
    seen = set(top_sentence_ids)
    result: list[int] = []
    for sid in top_sentence_ids:
        for sid2, _ in graph.similar_sentences(sid, top_k=top_k_per_sentence):
            if sid2 in seen:
                continue
            rec = encoded_index.get(sid2)
            if rec and rec.get("genre_id") in allowed:
                seen.add(sid2)
                result.append(sid2)
    return result


def run(
    query: str,
    engine: Any,
    config: dict[str, Any],
    data_path: Path | None = None,
) -> str | None:
    """
    Entrypoint: get concept bundle, load graph/vocab/encoded index, activate,
    traverse (with optional weights and max_iter), detect pattern (+ secondary),
    optionally next_span, refine answer. Returns response text or None if data missing.
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

    genre_ids_cfg = config.get("genre_ids")
    genre_id: str | list[str] = (
        genre_ids_cfg if isinstance(genre_ids_cfg, list) and genre_ids_cfg
        else config.get("default_genre_id", "general")
    )
    num_hops = int(config.get("attention_loop_hops", 4))
    top_k = int(config.get("attention_loop_top_k", 10))
    use_weights = config.get("attention_loop_use_weights", True)
    path_groups = int(config.get("attention_loop_path_groups", 2))
    next_span = config.get("attention_loop_next_span", True)
    max_iter = max(1, int(config.get("attention_loop_max_iter", 1)))

    query_token_ids: list[int] | None = None
    if config.get("use_query_token_ids", True) and (query or "").strip():
        query_token_ids = list(
            dict.fromkeys(
                word_to_id[t] for t in tokenize(query) if t in word_to_id
            )
        )
        if not query_token_ids:
            query_token_ids = None

    activated_word_ids, activated_sentence_ids = activate(
        concept_bundle, graph, word_to_id, query_token_ids=query_token_ids
    )
    word_visits: dict[int, float] = {}
    sentence_visits: dict[int, float] = {}
    for _ in range(max_iter):
        wv, sv = traverse_loops(
            activated_word_ids,
            activated_sentence_ids,
            graph,
            num_hops=num_hops,
            genre_id=genre_id,
            encoded_index=encoded_index,
            use_weights=use_weights,
        )
        for k, v in wv.items():
            word_visits[k] = word_visits.get(k, 0.0) + v
        for k, v in sv.items():
            sentence_visits[k] = sentence_visits.get(k, 0.0) + v

    use_graph_vectors = config.get("use_graph_vectors", False)
    if use_graph_vectors:
        vectors_path = data_path / "corpus" / "word_vectors.json"
        word_vectors = None
        try:
            from .graph_vectors import load_word_vectors, boost_sentence_visits_by_vectors
            word_vectors = load_word_vectors(vectors_path)
            if word_vectors:
                boost_sentence_visits_by_vectors(
                    sentence_visits,
                    word_vectors,
                    concept_bundle,
                    word_to_id,
                    graph.sentence_token_ids,
                    boost=float(config.get("graph_vectors_boost", 0.5)),
                )
        except Exception:
            pass

    feedback_weights_path = config.get("feedback_weights_path")
    if feedback_weights_path is None and data_path:
        feedback_weights_path = data_path / "feedback_weights.json"
    if feedback_weights_path:
        try:
            from .feedback import load_weights as load_feedback_weights, apply_boosts as apply_feedback_boosts
            fpath = Path(feedback_weights_path) if isinstance(feedback_weights_path, str) else feedback_weights_path
            if fpath.exists():
                fw = load_feedback_weights(fpath)
                if fw:
                    apply_feedback_boosts(
                        sentence_visits,
                        fw,
                        query,
                        boost=float(config.get("feedback_boost", 0.5)),
                    )
        except Exception:
            pass

    pattern_word_ids, pattern_sentence_ids, secondary_word_ids, secondary_sentence_ids = detect_pattern(
        word_visits, sentence_visits, top_k=top_k, num_groups=path_groups
    )
    next_span_ids: list[int] = []
    if next_span and pattern_sentence_ids:
        next_span_ids = _next_span_sentence_ids(
            pattern_sentence_ids, graph, genre_id, encoded_index, top_k_per_sentence=3
        )

    output_format = config.get("attention_loop_output_format", "list")
    paragraph_max_chars = int(config.get("attention_loop_paragraph_max_chars", 500))
    include_definitions = config.get("include_definitions_in_response", False)

    return refine_answer(
        pattern_word_ids,
        pattern_sentence_ids,
        concept_bundle,
        encoded_index,
        id_to_word,
        genre_id=genre_id,
        secondary_word_ids=secondary_word_ids,
        secondary_sentence_ids=secondary_sentence_ids,
        next_span_sentence_ids=next_span_ids,
        sentence_visits=sentence_visits,
        output_format=output_format,
        paragraph_max_chars=paragraph_max_chars,
        include_definitions=include_definitions,
    )
