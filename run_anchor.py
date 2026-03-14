#!/usr/bin/env python3
"""
Entry point: wire -> AnchorEngine.query(question) -> print response and critic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ANCHOR_ROOT = Path(__file__).resolve().parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

import json

from anchor.engine import AnchorEngine
from anchor.wire import get_config, get_engine, get_generator_kind


def _run_check(config: dict, engine: object) -> None:
    """Print setup status: dictionary, data dir, graph/vocab/encoded index, generator."""
    print("Dictionary:", "OK" if engine is not None else "missing")
    data_dir = config.get("align_data_dir")
    if not data_dir:
        print("Data dir: not set (align_data_dir)")
    else:
        data_path = Path(data_dir)
        print("Data dir:", data_path, "exists" if data_path.exists() else "missing")
        if data_path.exists():
            graph_path = data_path / "corpus" / "graph.json"
            vocab_path = data_path / "corpus" / "vocab.json"
            encoded_path = data_path / "corpus" / "encoded_sentences.jsonl"
            print("  corpus/graph.json:", "exists" if graph_path.exists() else "missing")
            if vocab_path.exists():
                try:
                    with open(vocab_path, encoding="utf-8") as f:
                        obj = json.load(f)
                    n = len(obj.get("word_to_id", {}))
                    print("  corpus/vocab.json: exists,", n, "words")
                except Exception:
                    print("  corpus/vocab.json: exists (read error)")
            else:
                print("  corpus/vocab.json: missing")
            if encoded_path.exists():
                try:
                    with open(encoded_path, encoding="utf-8") as f:
                        n = sum(1 for _ in f if _.strip())
                    print("  corpus/encoded_sentences.jsonl: exists,", n, "lines")
                except Exception:
                    print("  corpus/encoded_sentences.jsonl: exists (read error)")
            else:
                print("  corpus/encoded_sentences.jsonl: missing")
    generator_kind = get_generator_kind()
    print("Generator:", generator_kind)


def _print_result(
    question: str,
    response: str,
    critic_info: dict,
    generator_kind: str,
    verbose: bool = False,
    concept_bundle: dict | None = None,
    style_count: int | None = None,
    generator_meta: dict | None = None,
) -> None:
    """Print response and optional pipeline details. generator_meta can contain fallback_reason, graph_sentences, vocab_size."""
    gen_meta = generator_meta or {}
    if gen_meta.get("fallback_reason"):
        print(f"[Anchor] {gen_meta['fallback_reason']}")
    g_sentences = gen_meta.get("graph_sentences")
    g_vocab = gen_meta.get("vocab_size")
    if g_sentences is not None and g_vocab is not None:
        print(f"Generator: graph_attention (graph: {g_sentences} sentences, vocab: {g_vocab})")
    else:
        print(f"Generator: {generator_kind}")
    if gen_meta.get("epistemic_conflict"):
        print("[Anchor] Corpus is divided; showing both sides.")
        if verbose:
            sides = gen_meta.get("epistemic_sides") or []
            if len(sides) >= 2:
                a, b = sides[0], sides[1]
                n_a = len(a.get("texts") or [])
                n_b = len(b.get("texts") or [])
                m_a = a.get("total_mass", 0)
                m_b = b.get("total_mass", 0)
                print(f"  Side A: {n_a} sentences, mass {m_a:.4f}; Side B: {n_b} sentences, mass {m_b:.4f}")
    if verbose and concept_bundle is not None:
        terms = concept_bundle.get("terms") or []
        print("---")
        print(f"Concepts: {', '.join(terms[:12])}{'...' if len(terms) > 12 else ''}")
        if style_count is not None:
            print(f"Style sentences: {style_count}")
        voice_terms = gen_meta.get("voice_central_terms")
        if voice_terms:
            print(f"Voice of corpus (central terms): {', '.join(voice_terms[:15])}{'...' if len(voice_terms) > 15 else ''}")
        print("---")
    print("Response:")
    print(response.strip() or "(no text)")
    score = critic_info.get("score")
    decision = critic_info.get("decision")
    msg = critic_info.get("message", "")
    grounded = critic_info.get("num_grounded")
    content = critic_info.get("num_content")
    if verbose and grounded is not None and content is not None:
        print(f"\n[Anchor] generator={generator_kind} score={score} grounded={grounded}/{content} decision={decision} {msg}")
    else:
        print(f"\n[Anchor] generator={generator_kind} score={score} decision={decision} {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor: dictionary as anchor for LM output")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--repl", action="store_true", help="Read-eval-print loop")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show pipeline: generator, concepts, style count, critic details")
    parser.add_argument("--stream", action="store_true", help="Stream response in chunks (LLM-like)")
    parser.add_argument("--feedback", choices=["accept", "reject"], help="Record feedback for this response (accept or reject)")
    parser.add_argument("--check", action="store_true", help="Check setup and exit (dictionary, data dir, graph, generator).")
    args = parser.parse_args()

    config = get_config()
    engine = get_engine()

    if args.check:
        _run_check(config, engine)
        sys.exit(0 if engine is not None else 1)

    if engine is None:
        print("Anchor: dictionary_path not set or dictionary not found.", file=sys.stderr)
        print("Set dictionary_path in config/paths.json or ANCHOR_DICTIONARY_PATH.", file=sys.stderr)
        sys.exit(1)

    generator_kind = get_generator_kind()
    anchor_engine = AnchorEngine(engine, config, generator_kind=generator_kind)

    if args.repl:
        conversation_history: list[tuple[str, str]] = []
        turn_limit = int(config.get("conversation_turn_limit", 2))
        while True:
            try:
                line = input("anchor> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            response, critic_info, extras = anchor_engine.query(
                line, return_extras=args.verbose, conversation_history=conversation_history[-turn_limit:]
            )
            conversation_history.append((line, response))
            if args.verbose and extras:
                _print_result(line, response, critic_info, generator_kind, verbose=True, concept_bundle=extras.get("concept_bundle"), style_count=len(extras.get("style_sentences") or []), generator_meta=extras.get("generator_meta"))
            else:
                _print_result(line, response, critic_info, generator_kind, generator_meta=extras.get("generator_meta"))
        return

    question = args.question
    if not question:
        print("Provide a question or use --repl.", file=sys.stderr)
        sys.exit(1)

    if args.stream:
        print("Response:")
        response_parts: list[str] = []
        critic_info = {}
        extras = None
        for chunk, crit, ext in anchor_engine.query(question, return_extras=args.verbose, stream=True):
            if chunk is not None:
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
            else:
                critic_info = crit or {}
                extras = ext
        response = "".join(response_parts)
        print()
        score = critic_info.get("score")
        decision = critic_info.get("decision")
        msg = critic_info.get("message", "")
        gen_meta = (extras or {}).get("generator_meta") or {}
        if gen_meta.get("fallback_reason"):
            print(f"[Anchor] {gen_meta['fallback_reason']}")
        g_sentences = gen_meta.get("graph_sentences")
        g_vocab = gen_meta.get("vocab_size")
        if g_sentences is not None and g_vocab is not None:
            print(f"Generator: graph_attention (graph: {g_sentences} sentences, vocab: {g_vocab})")
        else:
            print(f"Generator: {generator_kind}")
        if gen_meta.get("epistemic_conflict"):
            print("[Anchor] Corpus is divided; showing both sides.")
        if args.verbose and extras:
            terms = (extras.get("concept_bundle") or {}).get("terms") or []
            print("---")
            print(f"Concepts: {', '.join(terms[:12])}{'...' if len(terms) > 12 else ''}")
            print(f"Style sentences: {len(extras.get('style_sentences') or [])}")
            print("---")
        grounded = critic_info.get("num_grounded")
        content = critic_info.get("num_content")
        if args.verbose and grounded is not None and content is not None:
            print(f"[Anchor] generator={generator_kind} score={score} grounded={grounded}/{content} decision={decision} {msg}")
        else:
            print(f"[Anchor] generator={generator_kind} score={score} decision={decision} {msg}")
    elif args.verbose:
        response, critic_info, extras = anchor_engine.query(question, return_extras=True)
        extras = extras or {}
        concept_bundle = extras.get("concept_bundle")
        style_sentences = extras.get("style_sentences") or []
        _print_result(question, response, critic_info, generator_kind, verbose=True, concept_bundle=concept_bundle, style_count=len(style_sentences), generator_meta=extras.get("generator_meta"))
    else:
        response, critic_info, extras = anchor_engine.query(question)
        _print_result(question, response, critic_info, generator_kind, generator_meta=(extras or {}).get("generator_meta"))

    if args.feedback:
        from pathlib import Path
        from anchor.feedback import record
        feedback_path = config.get("feedback_path")
        if feedback_path:
            feedback_path = Path(feedback_path)
        record("cli", question, response, accepted=(args.feedback == "accept"), path=feedback_path)
        print(f"[Anchor] Recorded feedback: {args.feedback}")


if __name__ == "__main__":
    main()
