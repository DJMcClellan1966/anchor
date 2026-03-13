"""
Graph attention loop: query lights up the graph, traverse loops with attention,
detect repeating pattern, refine answer from pattern (grounded, non-hallucinatory).

Refactored LLM-like form: Embed -> Layers (optional Norm) -> Output head (softmax over v_W).
Supports one-shot (pattern + refine) or autoregressive (sample next token loop).
"""
from __future__ import annotations

import json
import random
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
                    "source": obj.get("source"),
                    "term": obj.get("term"),
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


def normalize_visit_dict(d: dict[int, float]) -> dict[int, float]:
    """L1-normalize so values sum to 1. If sum is 0, return d unchanged."""
    total = sum(d.values())
    if total <= 0:
        return d
    return {k: v / total for k, v in d.items()}


def embed_anchor(
    concept_bundle: dict[str, Any],
    graph: CorpusGraph,
    word_to_id: dict[str, int],
    query_token_ids: list[int] | None = None,
    use_definition_words: bool = False,
    definition_word_weight: float = 0.5,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Build initial state H^(0) = (v_W_0, v_S_0) from query and concept bundle.
    Same as activation: terms + query_token_ids -> word and sentence nodes with weight 1.0.
    When use_definition_words is True, also add token IDs from definition text to v_W_0
    with definition_word_weight (get more from D: definition-aware propagation).
    """
    activated_word_ids, activated_sentence_ids = activate(
        concept_bundle, graph, word_to_id, query_token_ids=query_token_ids
    )
    v_W_0 = {wid: 1.0 for wid in activated_word_ids}
    v_S_0 = {sid: 1.0 for sid in activated_sentence_ids}

    if use_definition_words and definition_word_weight > 0:
        definitions = concept_bundle.get("definitions") or {}
        for term, defn in definitions.items():
            if not defn:
                continue
            texts = defn if isinstance(defn, list) else [defn]
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    continue
                for t in tokenize(text.strip()):
                    if t not in word_to_id:
                        continue
                    wid = word_to_id[t]
                    v_W_0[wid] = v_W_0.get(wid, 0.0) + definition_word_weight
        if v_W_0 and activated_sentence_ids:
            for wid in v_W_0:
                if wid not in activated_word_ids and wid not in (query_token_ids or []):
                    for sid in graph.sentences_containing_word(wid):
                        v_S_0[sid] = v_S_0.get(sid, 0.0) + definition_word_weight * 0.5

    return v_W_0, v_S_0


def propagation_layer(
    v_W: dict[int, float],
    v_S: dict[int, float],
    graph: CorpusGraph,
    genre_id: str | list[str] | None,
    encoded_index: dict[int, dict[str, Any]] | None,
    use_weights: bool,
    use_cooccurrence: bool = False,
    use_backward: bool = False,
    content_dependent_j: bool = False,
    overlay: dict[str, Any] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    One hop of propagation: word->sentence, sentence->sentence (J), sentence->word, word->word (P).
    Optional: Co(w) word->word, P_prev backward word->word, content-dependent J reweight, overlay boosts.
    Returns new visit dicts (additive update from current state).
    """
    allowed_genres: set[str] = set()
    if genre_id is not None:
        allowed_genres = {genre_id} if isinstance(genre_id, str) else set(genre_id)

    def sentence_ok(sid: int) -> bool:
        if encoded_index is None or not allowed_genres:
            return True
        rec = encoded_index.get(sid)
        return rec is not None and rec.get("genre_id") in allowed_genres

    w_copy = dict(v_W)
    s_copy = dict(v_S)
    word_visits = dict(v_W)
    sentence_visits = dict(v_S)

    for word_id, weight in w_copy.items():
        for sid in graph.sentences_containing_word(word_id):
            if not sentence_ok(sid):
                continue
            tokens = graph.sentence_token_ids(sid)
            add = weight / len(tokens) if use_weights and tokens else weight
            sentence_visits[sid] = sentence_visits.get(sid, 0.0) + add

    for sentence_id, weight in s_copy.items():
        if not sentence_ok(sentence_id):
            continue
        for sid2, jaccard in graph.similar_sentences(sentence_id, top_k=10):
            if not sentence_ok(sid2):
                continue
            j = jaccard
            if content_dependent_j and v_W:
                tokens2 = graph.sentence_token_ids(sid2)
                if tokens2:
                    focus = sum(v_W.get(w, 0.0) for w in tokens2) / len(tokens2)
                    j = j * (1.0 + focus)
            sentence_visits[sid2] = sentence_visits.get(sid2, 0.0) + weight * j
        tokens = graph.sentence_token_ids(sentence_id)
        per_word = (weight / len(tokens)) if use_weights and tokens else weight
        for wid in tokens:
            word_visits[wid] = word_visits.get(wid, 0.0) + per_word

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

    if use_cooccurrence:
        for word_id, mass in w_copy.items():
            co_list = graph.cooccurring_words(word_id)
            if not co_list:
                continue
            add = mass / len(co_list)
            for co_id in co_list:
                word_visits[co_id] = word_visits.get(co_id, 0.0) + add

    if use_backward:
        for w_prime, mass in list(word_visits.items()):
            prev_counts = graph.prev_word_counts(w_prime)
            if not prev_counts:
                continue
            total = sum(prev_counts.values())
            if total <= 0:
                continue
            for w_prev, count in prev_counts.items():
                word_visits[w_prev] = word_visits.get(w_prev, 0.0) + mass * (count / total)

    if overlay:
        _apply_propagation_overlay(word_visits, sentence_visits, w_copy, s_copy, overlay)

    return word_visits, sentence_visits


def _apply_propagation_overlay(
    word_visits: dict[int, float],
    sentence_visits: dict[int, float],
    w_copy: dict[int, float],
    s_copy: dict[int, float],
    overlay: dict[str, Any],
) -> None:
    """In-place: add overlay boosts to word_visits and sentence_visits."""
    ww = overlay.get("word_word") or {}
    ss = overlay.get("sentence_sentence") or {}
    for key, boost in ww.items():
        if isinstance(boost, (int, float)) and isinstance(key, str) and "|" in key:
            parts = key.split("|", 1)
            if len(parts) == 2:
                try:
                    w, w2 = int(parts[0]), int(parts[1])
                    m = w_copy.get(w, 0.0)
                    if m:
                        word_visits[w2] = word_visits.get(w2, 0.0) + m * float(boost)
                except ValueError:
                    pass
    for key, boost in ss.items():
        if isinstance(boost, (int, float)) and isinstance(key, str) and "|" in key:
            parts = key.split("|", 1)
            if len(parts) == 2:
                try:
                    s, s2 = int(parts[0]), int(parts[1])
                    m = s_copy.get(s, 0.0)
                    if m:
                        sentence_visits[s2] = sentence_visits.get(s2, 0.0) + m * float(boost)
                except ValueError:
                    pass


def load_propagation_overlay(path: Path | None) -> dict[str, Any]:
    """Load propagation overlay from JSON: word_word, sentence_sentence keyed by 'w|w2' / 's|s2'."""
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "word_word": data.get("word_word") or {},
        "sentence_sentence": data.get("sentence_sentence") or {},
    }


def save_propagation_overlay(path: Path, overlay: dict[str, Any]) -> None:
    """Save propagation overlay to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=0)


def run_layers(
    v_W_0: dict[int, float],
    v_S_0: dict[int, float],
    graph: CorpusGraph,
    num_hops: int,
    normalize: bool,
    genre_id: str | list[str] | None,
    encoded_index: dict[int, dict[str, Any]] | None,
    use_weights: bool = True,
    use_cooccurrence: bool = False,
    use_backward: bool = False,
    content_dependent_j: bool = False,
    overlay: dict[str, Any] | None = None,
    converge_tol: float | None = None,
    max_converge_iters: int = 50,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Run propagation steps. If converge_tol is set, iterate until L1 change < tol or max_converge_iters.
    Else run num_hops - 1 steps. After each step optionally L1-normalize.
    """
    v_W = dict(v_W_0)
    v_S = dict(v_S_0)
    n_steps = max(0, num_hops - 1)
    if converge_tol is not None and converge_tol > 0:
        for _ in range(max_converge_iters):
            v_W_old = dict(v_W)
            v_S_old = dict(v_S)
            v_W, v_S = propagation_layer(
                v_W, v_S, graph, genre_id, encoded_index, use_weights,
                use_cooccurrence=use_cooccurrence,
                use_backward=use_backward,
                content_dependent_j=content_dependent_j,
                overlay=overlay,
            )
            if normalize:
                v_W = normalize_visit_dict(v_W)
                v_S = normalize_visit_dict(v_S)
            dw = sum(abs(v_W.get(k, 0) - v_W_old.get(k, 0)) for k in set(v_W) | set(v_W_old))
            ds = sum(abs(v_S.get(k, 0) - v_S_old.get(k, 0)) for k in set(v_S) | set(v_S_old))
            if dw + ds < converge_tol:
                break
    else:
        for _ in range(n_steps):
            v_W, v_S = propagation_layer(
                v_W, v_S, graph, genre_id, encoded_index, use_weights,
                use_cooccurrence=use_cooccurrence,
                use_backward=use_backward,
                content_dependent_j=content_dependent_j,
                overlay=overlay,
            )
            if normalize:
                v_W = normalize_visit_dict(v_W)
                v_S = normalize_visit_dict(v_S)
    return v_W, v_S


def output_head(
    v_W: dict[int, float],
    dict_term_ids: set[int] | None = None,
    dict_boost: float = 0.0,
) -> dict[int, float]:
    """
    Optionally reweight v_W by boosting dictionary terms, then L1-normalize to distribution p.
    dict_boost: additive factor for terms in dict_term_ids, e.g. reweight[w] = v_W[w] * (1 + dict_boost).
    """
    if not dict_term_ids or dict_boost <= 0:
        return normalize_visit_dict(v_W)
    reweighted = {}
    for w, val in v_W.items():
        reweighted[w] = val * (1.0 + dict_boost) if w in dict_term_ids else val
    return normalize_visit_dict(reweighted)


def output_head_sentence_mixture(
    v_W: dict[int, float],
    v_S: dict[int, float],
    context_last_token_id: int | None,
    graph: CorpusGraph,
    dict_term_ids: set[int] | None = None,
    dict_boost: float = 0.0,
    mixture_weight: float = 0.0,
) -> dict[int, float]:
    """
    Blend base output_head distribution with a v_S-weighted mixture of per-sentence
    next-token distributions: p = (1 - alpha) * p_base + alpha * p_mix, where
    p_mix = sum_s v_S(s) * P_s(·|context_last_token_id) and P_s is bigram within sentence s.
    When mixture_weight <= 0 or no v_S or context_last_token_id is None, returns p_base only.
    """
    p_base = output_head(v_W, dict_term_ids=dict_term_ids, dict_boost=dict_boost)
    if mixture_weight <= 0 or not v_S or context_last_token_id is None:
        return p_base
    p_mix: dict[int, float] = {}
    for sid, weight in v_S.items():
        if weight <= 0:
            continue
        counts = graph.next_word_counts_in_sentence(sid, context_last_token_id)
        if not counts:
            continue
        total = sum(counts.values())
        if total <= 0:
            continue
        for w, c in counts.items():
            p_mix[w] = p_mix.get(w, 0.0) + weight * (c / total)
    if not p_mix:
        return p_base
    p_mix = normalize_visit_dict(p_mix)
    all_keys = set(p_base) | set(p_mix)
    blended = {}
    for w in all_keys:
        blended[w] = (1.0 - mixture_weight) * p_base.get(w, 0.0) + mixture_weight * p_mix.get(w, 0.0)
    return normalize_visit_dict(blended)


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
    return_sources: bool = False,
    **kwargs: Any,
) -> str | tuple[str, list[dict[str, Any]]]:
    """
    Build response from pattern: definitions for pattern keywords (from concept_bundle)
    and sentence texts (from encoded index, genre-filtered). When multiple definitions
    exist per term (senses), picks the sentence_id with highest sentence_visits, else first.
    Sentences are ordered by sentence_visits (desc) when provided; then secondary and next_span.
    output_format "list" = newline-separated; "paragraph" = definitions then one fused paragraph.
    When include_definitions is False, only corpus sentences are included (LLM-like answer).
    When return_sources is True, returns (y, source_records) where each record is
    {"type": "definition", "term": str} or {"type": "sentence", "sentence_id": int}. Grounded only.
    """
    terms_set = set((concept_bundle.get("terms") or []))
    definitions = concept_bundle.get("definitions") or {}
    # Definition-from-store: term -> list of sentence_ids (one per sense in unified store)
    term_to_sids: dict[str, list[int]] = {}
    for sid, rec in encoded_index.items():
        t = rec.get("term")
        if t:
            key = str(t)
            if key not in term_to_sids:
                term_to_sids[key] = []
            term_to_sids[key].append(sid)
    parts: list[str] = []
    source_records: list[dict[str, Any]] = []  # built when return_sources is True
    seen_defs: set[str] = set()
    max_total = max_definitions + max_sentences

    def add_definition(wid: int) -> bool:
        term = id_to_word.get(wid)
        if term is None or term not in terms_set or term in seen_defs:
            return False
        seen_defs.add(term)
        # Prefer definition from unified store: pick one sid by sentence_visits or first
        if term in term_to_sids:
            sids = term_to_sids[term]
            if sentence_visits:
                chosen_sid = max(sids, key=lambda sid: sentence_visits.get(sid, 0.0))
            else:
                chosen_sid = sids[0]
            rec = encoded_index.get(chosen_sid, {})
            text = rec.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip()[:500])
                if return_sources:
                    source_records.append({"type": "definition", "term": term})
                return True
        defn = definitions.get(term)
        if isinstance(defn, str) and defn.strip():
            parts.append(f"{term}: {defn.strip()[:500]}")
            if return_sources:
                source_records.append({"type": "definition", "term": term})
            return True
        if isinstance(defn, list) and defn and isinstance(defn[0], str):
            parts.append(f"{term}: {defn[0].strip()[:500]}")
            if return_sources:
                source_records.append({"type": "definition", "term": term})
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
    sentence_items: list[tuple[int, str]] = []  # (sid, text) for source attribution
    sent_allowed = {genre_id} if isinstance(genre_id, str) else set(genre_id)
    for sid in all_sids:
        if len(sentence_items) >= max_sentences:
            break
        rec = encoded_index.get(sid)
        if not rec or rec.get("genre_id") not in sent_allowed:
            continue
        text = (rec.get("text") or "").strip()
        if text and text not in seen_sents:
            seen_sents.add(text)
            sentence_items.append((sid, text))

    if output_format == "paragraph" and sentence_items:
        sentence_texts = [t for _, t in sentence_items]
        fused = " ".join(sentence_texts)
        if len(fused) > paragraph_max_chars:
            cut = fused[: paragraph_max_chars - 3]
            last_space = cut.rfind(" ")
            fused = (cut[:last_space] if last_space > 0 else cut) + "..."
        parts.append("In the corpus: " + fused)
        if return_sources:
            for sid, _ in sentence_items:
                source_records.append({"type": "sentence", "sentence_id": sid})
    else:
        for sid, text in sentence_items:
            parts.append(text)
            if return_sources:
                source_records.append({"type": "sentence", "sentence_id": sid})
            if len(parts) >= max_total:
                break

    if not parts:
        msg = "Concepts: " + ", ".join(list(terms_set)[:15]) if terms_set else "No pattern found for this query."
        if return_sources:
            return (msg, [])
        return msg
    y = "\n".join(parts[:max_total])
    if return_sources:
        return (y, source_records)
    return y


def _sample_from_distribution(p: dict[int, float]) -> int | None:
    """Sample one word ID from distribution p (keys = word_id, values = probability). Returns None if empty."""
    if not p:
        return None
    population = list(p.keys())
    weights = [p[k] for k in population]
    return random.choices(population, weights=weights, k=1)[0]


def generate_autoregressive(
    query: str,
    concept_bundle: dict[str, Any],
    config: dict[str, Any],
    graph: CorpusGraph,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    encoded_index: dict[int, dict[str, Any]],
    genre_id: str | list[str],
) -> str:
    """
    Generate response token-by-token: embed(context) -> layers -> softmax(v_W) -> sample next -> append.
    Stops at max_tokens or when a sentence-ending token (., ?, !) is sampled if configured.
    """
    max_tokens = int(config.get("autoregressive_max_tokens", 80))
    context_window = int(config.get("autoregressive_context_window", 10))
    stop_at_sentence_end = config.get("autoregressive_stop_at_sentence_end", True)
    num_hops = int(config.get("attention_loop_hops", 4))
    use_weights = config.get("attention_loop_use_weights", True)
    use_normalized = config.get("use_normalized_layers", True)
    include_definitions = config.get("include_definitions_in_response", False)
    use_cooccurrence = config.get("use_propagation_cooccurrence", False)
    use_backward = config.get("use_propagation_backward", False)
    content_dependent_j = config.get("use_content_dependent_j", False)
    overlay = {}  # generate_autoregressive does not load overlay by default (no data_path in signature)
    output_dict_boost = float(config.get("output_dict_boost", 0.0))
    use_sentence_mixture = config.get("use_sentence_mixture_output", True)
    sentence_mixture_weight = float(config.get("sentence_mixture_weight", 0.5))
    dict_term_ids: set[int] = set()
    if concept_bundle.get("terms"):
        dict_term_ids = {word_to_id[t] for t in (concept_bundle.get("terms") or []) if t in word_to_id}

    stop_ids: set[int] = set()
    for t in (".", "?", "!"):
        if t in word_to_id:
            stop_ids.add(word_to_id[t])

    # Initial context = query token IDs
    query_tokens = tokenize(query) if (query or "").strip() else []
    context = list(dict.fromkeys(word_to_id[t] for t in query_tokens if t in word_to_id))
    generated_tokens: list[int] = []

    for step in range(max_tokens):
        if not context:
            # No context: use concept terms only for embed
            query_token_ids_for_embed: list[int] | None = None
        elif step == 0:
            query_token_ids_for_embed = context
        else:
            query_token_ids_for_embed = context[-context_window:] if len(context) >= context_window else context

        use_def_words = config.get("use_definition_words_in_activation", True)
        def_weight = float(config.get("definition_word_weight", 0.5))
        v_W_0, v_S_0 = embed_anchor(
            concept_bundle, graph, word_to_id, query_token_ids=query_token_ids_for_embed,
            use_definition_words=use_def_words, definition_word_weight=def_weight,
        )
        if not v_W_0 and not v_S_0:
            break
        v_W, v_S = run_layers(
            v_W_0,
            v_S_0,
            graph,
            num_hops=num_hops,
            normalize=use_normalized,
            genre_id=genre_id,
            encoded_index=encoded_index,
            use_weights=use_weights,
            use_cooccurrence=use_cooccurrence,
            use_backward=use_backward,
            content_dependent_j=content_dependent_j,
            overlay=overlay if overlay else None,
        )
        if use_sentence_mixture and sentence_mixture_weight > 0 and context:
            p = output_head_sentence_mixture(
                v_W, v_S, context[-1], graph,
                dict_term_ids=dict_term_ids if dict_term_ids else None,
                dict_boost=output_dict_boost,
                mixture_weight=sentence_mixture_weight,
            )
        else:
            p = output_head(v_W, dict_term_ids=dict_term_ids if dict_term_ids else None, dict_boost=output_dict_boost)
        next_id = _sample_from_distribution(p)
        if next_id is None:
            # Fallback: sample from P(·|last token) if we have context
            if context:
                last = context[-1]
                next_counts = graph.next_word_counts(last)
                if next_counts:
                    total = sum(next_counts.values())
                    if total > 0:
                        population = list(next_counts.keys())
                        weights = [next_counts[k] for k in population]
                        next_id = random.choices(population, weights=weights, k=1)[0]
            if next_id is None:
                break
        context.append(next_id)
        generated_tokens.append(next_id)
        if stop_at_sentence_end and next_id in stop_ids:
            break

    parts: list[str] = []
    if include_definitions and concept_bundle.get("terms"):
        definitions = concept_bundle.get("definitions") or {}
        terms_set = set(concept_bundle.get("terms") or [])
        for term in terms_set:
            if term in definitions:
                defn = definitions[term]
                text = defn[0] if isinstance(defn, list) and defn else (defn if isinstance(defn, str) else "")
                if text:
                    parts.append(f"{term}: {str(text).strip()[:500]}")
        if parts:
            parts.append("")
    out_words = [id_to_word.get(i, "") for i in generated_tokens]
    parts.append(" ".join(w for w in out_words if w).strip())
    return "\n".join(parts).strip() or "No response generated."


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
    cached = None
    try:
        from .corpus_cache import get_cached_corpus_data
        cached = get_cached_corpus_data(data_path)
    except ImportError:
        pass
    if cached is not None:
        graph, word_to_id, id_to_word, encoded_index = cached
    else:
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

    use_autoregressive = config.get("use_autoregressive_generation", False)
    if use_autoregressive:
        return generate_autoregressive(
            query, concept_bundle, config, graph, word_to_id, id_to_word, encoded_index, genre_id
        )

    # One-shot path: refactored embed -> run_layers (with optional norm) -> pattern -> refine
    use_normalized_layers = config.get("use_normalized_layers", True)
    use_cooccurrence = config.get("use_propagation_cooccurrence", False)
    use_backward = config.get("use_propagation_backward", False)
    content_dependent_j = config.get("use_content_dependent_j", False)
    converge_tol = config.get("propagation_converge_tol")
    max_converge_iters = int(config.get("propagation_max_converge_iters", 50))
    overlay_path = config.get("propagation_overlay_path")
    if overlay_path and data_path:
        p = Path(overlay_path)
        overlay_path = (data_path / overlay_path) if not p.is_absolute() else p
    else:
        overlay_path = Path(overlay_path) if overlay_path else None
    overlay = load_propagation_overlay(overlay_path) if overlay_path else {}
    if overlay and not (overlay.get("word_word") or overlay.get("sentence_sentence")):
        overlay = {}

    use_def_words = config.get("use_definition_words_in_activation", True)
    def_weight = float(config.get("definition_word_weight", 0.5))
    v_W_0, v_S_0 = embed_anchor(
        concept_bundle, graph, word_to_id, query_token_ids=query_token_ids,
        use_definition_words=use_def_words, definition_word_weight=def_weight,
    )
    word_visits: dict[int, float] = {}
    sentence_visits: dict[int, float] = {}
    for _ in range(max_iter):
        v_W, v_S = run_layers(
            v_W_0,
            v_S_0,
            graph,
            num_hops=num_hops,
            normalize=use_normalized_layers,
            genre_id=genre_id,
            encoded_index=encoded_index,
            use_weights=use_weights,
            use_cooccurrence=use_cooccurrence,
            use_backward=use_backward,
            content_dependent_j=content_dependent_j,
            overlay=overlay if overlay else None,
            converge_tol=converge_tol,
            max_converge_iters=max_converge_iters,
        )
        for k, v in v_W.items():
            word_visits[k] = word_visits.get(k, 0.0) + v
        for k, v in v_S.items():
            sentence_visits[k] = sentence_visits.get(k, 0.0) + v

    output_dict_boost = float(config.get("output_dict_boost", 0.0))
    if output_dict_boost > 0 and concept_bundle.get("terms"):
        terms_set = set((concept_bundle.get("terms") or []))
        dict_term_ids = {word_to_id[t] for t in terms_set if t in word_to_id}
        if dict_term_ids:
            for w in dict_term_ids:
                if w in word_visits:
                    word_visits[w] = word_visits[w] * (1.0 + output_dict_boost)

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
