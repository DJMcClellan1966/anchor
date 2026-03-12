"""
Concept bundle and style sentences. Self-contained in anchor.
Option C: when corpus graph exists, get_style_sentences_from_graph for graph-based retrieval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_concept_bundle(engine: Any, query: str) -> dict[str, Any]:
    """
    Build concept bundle from dictionary engine (get_context_for_description).
    Returns {"terms": [], "definitions": {}}.
    """
    bundle: dict[str, Any] = {"terms": [], "definitions": {}}
    if engine is None or not (query or "").strip():
        return bundle
    try:
        ctx_fn = getattr(engine, "get_context_for_description", None)
        if not callable(ctx_fn):
            return bundle
        raw = ctx_fn(query.strip())
        if not isinstance(raw, dict):
            return bundle
        for k in ("concepts", "key_words", "key_words_list"):
            val = raw.get(k)
            if isinstance(val, list):
                for c in val:
                    term = c.get("name") if isinstance(c, dict) else c
                    if isinstance(term, str) and term.strip():
                        bundle["terms"].append(term.strip())
        defs = raw.get("definitions") or raw.get("definition_map")
        if isinstance(defs, dict):
            bundle["definitions"].update(defs)
        bundle["terms"] = list(dict.fromkeys(bundle["terms"]))
    except Exception:
        pass
    return bundle


def _read_sentences_from_jsonl(
    jsonl_path: Path,
    concept_bundle: dict[str, Any],
    genre_filter: str | list[str] | None,
    max_sentences: int = 20,
) -> list[str]:
    """Read sentences from a JSONL file; optional genre_filter on genre_id (str or list of allowed genres)."""
    terms_set = set((concept_bundle.get("terms") or []))
    sentences: list[str] = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") if isinstance(obj, dict) else None
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if genre_filter is not None:
                        g = obj.get("genre_id") if isinstance(obj, dict) else None
                        if isinstance(genre_filter, list):
                            if g not in genre_filter:
                                continue
                        elif g != genre_filter:
                            continue
                    if terms_set:
                        words = set(text.lower().split())
                        if not (words & {t.lower() for t in terms_set}):
                            continue
                    sentences.append(text.strip())
                    if len(sentences) >= max_sentences:
                        break
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return sentences


def get_style_sentences(
    engine: Any,
    data_dir: Path | str | None,
    concept_bundle: dict[str, Any],
    genre_id: str | list[str] = "retirement",
    register: str | None = None,
) -> list[str]:
    """
    Load genre sentences. Prefers data_dir/corpus/sentences.jsonl (Option C) filtered by genre_id
    (str or list of allowed genre_ids). Falls back to per-genre file when genre_id is a single str.
    """
    if not data_dir:
        return []
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    # Option C: single combined corpus with genre tags
    corpus_sentences_path = data_path / "corpus" / "sentences.jsonl"
    if corpus_sentences_path.exists():
        return _read_sentences_from_jsonl(
            corpus_sentences_path, concept_bundle, genre_filter=genre_id
        )
    # Backward compatibility: per-genre file (only when single genre)
    first_genre = genre_id[0] if isinstance(genre_id, list) else genre_id
    jsonl_path = data_path / first_genre / "genre_sentences.jsonl"
    if not jsonl_path.exists():
        return []
    return _read_sentences_from_jsonl(
        jsonl_path, concept_bundle, genre_filter=None
    )


def get_style_sentences_from_graph(
    data_dir: Path | str,
    concept_bundle: dict[str, Any],
    genre_id: str | list[str] = "retirement",
    max_sentences: int = 20,
) -> list[str]:
    """
    Load style sentences using the corpus graph: find sentences containing concept terms,
    filter by genre_id (str or list of allowed genre_ids). Use when corpus/graph.json and corpus/vocab.json exist.
    """
    from .corpus_graph import load_corpus_graph
    from .corpus_vocab import load_vocab

    data_path = Path(data_dir)
    graph = load_corpus_graph(data_path)
    if graph is None:
        return []
    word_to_id, _ = load_vocab(data_path / "corpus" / "vocab.json")
    if not word_to_id:
        return []

    encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
    if not encoded_path.exists():
        return []
    with open(encoded_path, encoding="utf-8") as f:
        index = {}
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
                }
            except (json.JSONDecodeError, TypeError):
                continue

    terms_set = set((concept_bundle.get("terms") or []))
    candidate_sids: set[int] = set()
    for term in terms_set:
        term = (term or "").strip()
        if not term:
            continue
        wid = word_to_id.get(term)
        if wid is None:
            continue
        for sid in graph.sentences_containing_word(wid):
            candidate_sids.add(sid)
    if not terms_set:
        candidate_sids = set(graph.sentence_ids())

    allowed_genres = {genre_id} if isinstance(genre_id, str) else set(genre_id)
    out: list[str] = []
    for sid in candidate_sids:
        rec = index.get(sid)
        if not rec or rec.get("genre_id") not in allowed_genres:
            continue
        text = (rec.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= max_sentences:
            break
    return out
