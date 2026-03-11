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


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor: dictionary as anchor for LM output")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--repl", action="store_true", help="Read-eval-print loop")
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
        while True:
            try:
                line = input("anchor> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            response, critic_info = anchor_engine.query(line)
            print(response)
            print(f"  [score={critic_info.get('score')} decision={critic_info.get('decision')}] {critic_info.get('message', '')}")
        return

    question = args.question
    if not question:
        print("Provide a question or use --repl.", file=sys.stderr)
        sys.exit(1)

    response, critic_info = anchor_engine.query(question)
    print(response)
    print(f"\n[Anchor] score={critic_info.get('score')} decision={critic_info.get('decision')} {critic_info.get('message', '')}")


if __name__ == "__main__":
    main()
