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

from anchor.engine import AnchorEngine
from anchor.wire import get_config, get_engine, get_generator_kind


def _print_result(
    question: str,
    response: str,
    critic_info: dict,
    generator_kind: str,
    verbose: bool = False,
    concept_bundle: dict | None = None,
    style_count: int | None = None,
) -> None:
    """Print response and optional pipeline details."""
    if verbose and concept_bundle is not None:
        terms = concept_bundle.get("terms") or []
        print("---")
        print(f"Generator: {generator_kind}")
        print(f"Concepts: {', '.join(terms[:12])}{'...' if len(terms) > 12 else ''}")
        if style_count is not None:
            print(f"Style sentences: {style_count}")
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
    args = parser.parse_args()

    config = get_config()
    engine = get_engine()
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
                _print_result(line, response, critic_info, generator_kind, verbose=True, concept_bundle=extras.get("concept_bundle"), style_count=len(extras.get("style_sentences") or []))
            else:
                _print_result(line, response, critic_info, generator_kind)
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
        if args.verbose and extras:
            terms = (extras.get("concept_bundle") or {}).get("terms") or []
            print("---")
            print(f"Generator: {generator_kind}")
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
        _print_result(question, response, critic_info, generator_kind, verbose=True, concept_bundle=concept_bundle, style_count=len(style_sentences))
    else:
        response, critic_info, _ = anchor_engine.query(question)
        _print_result(question, response, critic_info, generator_kind)

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
