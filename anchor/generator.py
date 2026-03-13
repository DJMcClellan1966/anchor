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
) -> tuple[str, dict[str, Any]]:
    """
    Produce response text and generator metadata. Returns (response_text, gen_meta).
    gen_meta has generator_actually_used; graph_attention adds graph_sentences, vocab_size, or fallback_reason.
    """
    if generator_kind == "graph_attention":
        return _generate_graph_attention(question, concept_bundle, style_sentences, config)
    if generator_kind == "corpus":
        return _generate_corpus(question, concept_bundle, style_sentences, config)
    if generator_kind == "scratchllm":
        return _generate_scratchllm(question, concept_bundle, style_sentences, config)
    if generator_kind == "align":
        return _generate_align(question, concept_bundle, style_sentences, config)
    text = _generate_stub(question, concept_bundle, style_sentences, config)
    return (text, {"generator_actually_used": "stub"})


def _generate_graph_attention(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate using graph attention loop; fall back to stub if data missing or run returns None."""
    from pathlib import Path

    from . import graph_attention

    fallback_reason = "Graph data missing or incomplete."
    data_dir = config.get("align_data_dir") or config.get("ANCHOR_DATA_DIR")
    if not data_dir:
        stub = _generate_stub(question, concept_bundle, style_sentences, config)
        return (stub, {"generator_actually_used": "stub", "fallback_reason": fallback_reason})
    data_path = Path(data_dir)
    if not (data_path / "corpus" / "graph.json").exists() or not (data_path / "corpus" / "vocab.json").exists():
        stub = _generate_stub(question, concept_bundle, style_sentences, config)
        return (stub, {"generator_actually_used": "stub", "fallback_reason": fallback_reason})
    out = graph_attention.run(question, None, config, data_path)
    if out is None or not (out or "").strip():
        stub = _generate_stub(question, concept_bundle, style_sentences, config)
        return (stub, {"generator_actually_used": "stub", "fallback_reason": fallback_reason})
    # Success: compute graph_sentences and vocab_size
    graph_sentences = 0
    vocab_size = 0
    try:
        encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
        if encoded_path.exists():
            with open(encoded_path, encoding="utf-8") as f:
                graph_sentences = sum(1 for _ in f if (_.strip() or ""))
        vocab_path = data_path / "corpus" / "vocab.json"
        if vocab_path.exists():
            import json
            with open(vocab_path, encoding="utf-8") as f:
                obj = json.load(f)
            word_to_id = obj.get("word_to_id", {})
            vocab_size = len(word_to_id) if isinstance(word_to_id, dict) else 0
    except Exception:
        pass
    generation_mode = "autoregressive" if config.get("use_autoregressive_generation", False) else "one_shot"
    return (
        out.strip(),
        {
            "generator_actually_used": "graph_attention",
            "generation_mode": generation_mode,
            "graph_sentences": graph_sentences,
            "vocab_size": vocab_size,
        },
    )


def _generate_corpus(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate using corpus graph: hybrid next-token when index present, else next-sentence retrieval."""
    from pathlib import Path

    stub_meta = {"generator_actually_used": "stub"}
    data_dir = config.get("align_data_dir") or config.get("ANCHOR_DATA_DIR")
    if not data_dir:
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    data_path = Path(data_dir)
    graph_path = data_path / "corpus" / "graph.json"
    encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
    vocab_path = data_path / "corpus" / "vocab.json"
    if not graph_path.exists() or not encoded_path.exists() or not vocab_path.exists():
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    try:
        from .corpus_graph import load_corpus_graph
        from .corpus_vocab import load_vocab
        from .next_sentence import get_next_sentences_from_text

        graph = load_corpus_graph(data_path)
        if graph is None:
            return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
        word_to_id, id_to_word = load_vocab(vocab_path)
        genre_id = config.get("default_genre_id", "retirement")
        top_k = int(config.get("corpus_next_sentences_top_k", 3))
        context_length = int(config.get("corpus_hybrid_context_length", 5))
        beta = float(config.get("corpus_hybrid_beta", 0.7))
        max_tokens = int(config.get("corpus_max_tokens", 80))

        if graph.has_context_index() and style_sentences and word_to_id and id_to_word:
            from .corpus_vocab import tokenize
            from .next_token import get_hybrid_next_token_distribution, sample_next_token

            first_text = style_sentences[0] if style_sentences else ""
            token_ids = [word_to_id[t] for t in tokenize(first_text) if t in word_to_id]
            if not token_ids:
                token_ids = [word_to_id[t] for t in tokenize(question) if t in word_to_id]
            if not token_ids:
                for s in style_sentences[:3]:
                    token_ids = [word_to_id[t] for t in tokenize(s) if t in word_to_id]
                    if token_ids:
                        break
            vocab_size = len(word_to_id)
            out_ids = list(token_ids)
            while len(out_ids) < max_tokens:
                context = out_ids[-context_length:] if len(out_ids) >= context_length else out_ids
                dist = get_hybrid_next_token_distribution(
                    context, graph, encoded_path, vocab_size,
                    genre_id=genre_id, beta=beta,
                )
                if not dist:
                    break
                next_id = sample_next_token(dist)
                if next_id is None:
                    break
                out_ids.append(next_id)
            tokens = [id_to_word.get(i, "") for i in out_ids]
            text = " ".join(t for t in tokens if t).strip()
            if not text:
                return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
            return (text, {"generator_actually_used": "corpus"})

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
            return (" ".join(parts), {"generator_actually_used": "corpus"})
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    except Exception:
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)


def _generate_stub(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> str:
    """Build short answer: answer-style (terms + sentences) or definition-style per config."""
    terms = concept_bundle.get("terms") or []
    definitions = concept_bundle.get("definitions") or {}
    if not terms and not definitions:
        return "No concepts found for that query. Set dictionary_path in config/paths.json to use the dictionary."
    include_definitions = config.get("include_definitions_in_response", False)
    parts: list[str] = []
    if include_definitions:
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
    # Answer-style: Related terms + style sentences only
    if terms:
        parts.append("Related: " + ", ".join(terms[:15]))
    for s in (style_sentences or [])[:2]:
        if (s or "").strip():
            parts.append((s or "").strip())
    if not parts:
        return "No concepts found for that query. Set dictionary_path in config/paths.json to use the dictionary."
    return " ".join(parts) if len(parts) == 1 else (parts[0] + " " + " ".join(parts[1:]))


def _generate_scratchllm(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Optional: call scratchLLM respond. Truth base = definitions + style sentences. Falls back to stub if unavailable."""
    import json
    import os
    import sys
    from pathlib import Path

    stub_meta = {"generator_actually_used": "stub"}
    scratchllm_path = config.get("scratchllm_path")
    if not scratchllm_path:
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    path = Path(scratchllm_path).resolve()
    if not path.exists() or not path.is_dir():
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    try:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from base.retrieve import retrieve_formal_only  # type: ignore
    except Exception:
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)

    lines = []
    system_prompt = (config.get("system_prompt") or "").strip()
    if system_prompt:
        lines.append(json.dumps({"text": system_prompt[:500], "tier": 1, "source": "system"}, ensure_ascii=False))
    definitions = concept_bundle.get("definitions") or {}
    for term, defn in list(definitions.items())[:10]:
        if isinstance(defn, str) and defn.strip():
            text = f"{term}: {defn.strip()[:500]}"
            lines.append(json.dumps({"text": text, "tier": 1, "source": "dictionary"}, ensure_ascii=False))
        elif isinstance(defn, list) and defn and isinstance(defn[0], str):
            text = f"{term}: {defn[0].strip()[:500]}"
            lines.append(json.dumps({"text": text, "tier": 1, "source": "dictionary"}, ensure_ascii=False))
    for s in (style_sentences or [])[:10]:
        if s and (s or "").strip():
            lines.append(json.dumps({"text": (s[:500]).strip(), "tier": 2, "source": "genre"}, ensure_ascii=False))
    grammar_path = config.get("grammar_examples_path")
    if grammar_path:
        gpath = Path(grammar_path)
        if gpath.exists() and gpath.is_file():
            try:
                grammar_lines: list[str] = []
                with open(gpath, encoding="utf-8") as gf:
                    for line in gf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and obj.get("text"):
                                grammar_lines.append((obj.get("text") or "")[:500].strip())
                            else:
                                grammar_lines.append(line[:500].strip())
                        except (json.JSONDecodeError, TypeError):
                            grammar_lines.append(line[:500].strip())
                for text in grammar_lines[:10]:
                    if text:
                        lines.append(json.dumps({"text": text, "tier": 2, "source": "grammar"}, ensure_ascii=False))
            except OSError:
                pass

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        tmp = f.name
    try:
        out = retrieve_formal_only(question, truth_base_path=tmp, top_k=5)
        text = (out or "").strip()
        if not text:
            return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
        return (text, {"generator_actually_used": "scratchllm"})
    except Exception:
        return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _generate_align(
    question: str,
    concept_bundle: dict[str, Any],
    style_sentences: list[str],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Optional: call Align respond_bridge.query. Falls back to stub if unavailable."""
    stub_meta = {"generator_actually_used": "stub"}
    try:
        import sys
        from pathlib import Path
        align_root = Path(__file__).resolve().parent.parent.parent / "align"
        if not align_root.exists() or str(align_root) in sys.path:
            return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
        sys.path.insert(0, str(align_root))
        from Align.respond_bridge import query as align_query  # type: ignore
        response, _source, _critic = align_query(
            question,
            truth_base_path=config.get("mirror_truth_base_path"),
            use_dictionary=True,
        )
        text = (response or "").strip()
        if not text:
            return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
        return (text, {"generator_actually_used": "align"})
    except Exception:
        pass
    return (_generate_stub(question, concept_bundle, style_sentences, config), stub_meta)
