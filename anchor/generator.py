"""
Generate response: stub (from concept bundle) or optional scratchLLM/Align.
"""
from __future__ import annotations

from typing import Any


def generate(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
    generator_kind: str = "stub",
) -> str:
    """
    Produce response text. Stub uses terms and definitions; scratchLLM/Align use LM;
    corpus uses graph-based next-sentence (Option C).
    """
    if generator_kind == "corpus":
        return _generate_corpus(question, concept_bundle, style_sentences, config)
    if generator_kind == "scratchllm":
        return _generate_scratchllm(question, concept_bundle, style_sentences, config)
    if generator_kind == "align":
        return _generate_align(question, concept_bundle, style_sentences, config)
    return _generate_stub(question, concept_bundle, style_sentences, config)


def _generate_corpus(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> str:
    """Generate using corpus graph and next-sentence retrieval. Fallback to stub if graph/corpus missing."""
    from pathlib import Path

    data_dir = config.get("align_data_dir") or config.get("ANCHOR_DATA_DIR")
    if not data_dir:
        return _generate_stub(question, concept_bundle, style_sentences, config)
    data_path = Path(data_dir)
    graph_path = data_path / "corpus" / "graph.json"
    encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
    vocab_path = data_path / "corpus" / "vocab.json"
    if not graph_path.exists() or not encoded_path.exists() or not vocab_path.exists():
        return _generate_stub(question, concept_bundle, style_sentences, config)
    try:
        from .corpus_graph import load_corpus_graph
        from .corpus_vocab import load_vocab
        from .next_sentence import get_next_sentences_from_text

        graph = load_corpus_graph(data_path)
        if graph is None:
            return _generate_stub(question, concept_bundle, style_sentences, config)
        word_to_id, _ = load_vocab(vocab_path)
        genre_id = config.get("default_genre_id", "retirement")
        top_k = int(config.get("corpus_next_sentences_top_k", 3))
        parts = list(style_sentences[:2])
        if parts:
            current = parts[-1]
            next_sentences = get_next_sentences_from_text(
                current, genre_id, graph, encoded_path, word_to_id, top_k=top_k
            )
            for s in next_sentences:
                if s and s not in parts:
                    parts.append(s)
        if parts:
            return " ".join(parts)
        return _generate_stub(question, concept_bundle, style_sentences, config)
    except Exception:
        return _generate_stub(question, concept_bundle, style_sentences, config)


def _generate_stub(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> str:
    """Build short answer from terms and definitions."""
    terms = concept_bundle.get("terms") or []
    definitions = concept_bundle.get("definitions") or {}
    if not terms and not definitions:
        return "No concepts found for that query. Set dictionary_path in config/paths.json to use the dictionary."
    parts = []
    if terms:
        parts.append("Concepts: " + ", ".join(terms[:15]))
    for term, defn in list(definitions.items())[:5]:
        if isinstance(defn, str) and defn.strip():
            parts.append(f"{term}: {defn.strip()[:200]}")
        elif isinstance(defn, list) and defn:
            first = defn[0] if isinstance(defn[0], str) else str(defn[0])
            parts.append(f"{term}: {first[:200]}")
    if not parts:
        return "Concepts: " + ", ".join(terms[:15]) if terms else "No definitions available."
    return "\n".join(parts)


def _generate_scratchllm(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> str:
    """Optional: call scratchLLM respond. Falls back to stub if unavailable."""
    scratchllm_path = config.get("scratchllm_path")
    if not scratchllm_path:
        return _generate_stub(question, concept_bundle, style_sentences, config)
    try:
        import sys
        from pathlib import Path
        path = Path(scratchllm_path).resolve()
        if not path.exists() or str(path) in sys.path:
            return _generate_stub(question, concept_bundle, style_sentences, config)
        sys.path.insert(0, str(path))
        # ScratchLLM respond API varies; use stub if import or call fails
        from base.retrieve import retrieve_formal_only  # type: ignore
        truth_path = path / "data" / "truth_base.jsonl"
        if not truth_path.exists():
            return _generate_stub(question, concept_bundle, style_sentences, config)
        lines = []
        for s in style_sentences[:10]:
            lines.append('{"text": "%s", "tier": 2, "source": "genre"}' % s.replace('"', '\\"')[:500])
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in lines:
                f.write(line + "\n")
            tmp = f.name
        try:
            out = retrieve_formal_only(question, truth_base_path=tmp, top_k=5)
            return (out or "").strip() or _generate_stub(question, concept_bundle, style_sentences, config)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        pass
    return _generate_stub(question, concept_bundle, style_sentences, config)


def _generate_align(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> str:
    """Optional: call Align respond_bridge.query. Falls back to stub if unavailable."""
    try:
        import sys
        from pathlib import Path
        align_root = Path(__file__).resolve().parent.parent.parent / "align"
        if not align_root.exists() or str(align_root) in sys.path:
            return _generate_stub(question, concept_bundle, style_sentences, config)
        sys.path.insert(0, str(align_root))
        from Align.respond_bridge import query as align_query  # type: ignore
        response, _source, _critic = align_query(
            question,
            truth_base_path=config.get("mirror_truth_base_path"),
            use_dictionary=True,
        )
        return (response or "").strip() or _generate_stub(question, concept_bundle, style_sentences, config)
    except Exception:
        pass
    return _generate_stub(question, concept_bundle, style_sentences, config)
