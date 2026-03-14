#!/usr/bin/env python3
"""
Evidence engine + agent CLI: evaluate one or more claims against the corpus.
Returns structured verdict (supported / divided / silent), support and contradict sentences.
No dictionary required; concept bundle is built from query tokens when not provided.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ANCHOR_ROOT = Path(__file__).resolve().parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

from anchor.agent import run_task, AgentReport
from anchor.wire import get_config


def _print_report(report: AgentReport, verbose: bool = False) -> None:
    """Print one AgentReport: claim, verdict, support and contradict sentences."""
    print("---")
    print("Claim:", report.claim)
    print("Verdict:", report.evidence.verdict)
    print("Confidence:", round(report.evidence.confidence, 4))
    if report.evidence.support_sentences:
        print("Support:")
        for s in report.evidence.support_sentences:
            print("  -", s[:200] + ("..." if len(s) > 200 else ""))
    if report.evidence.contradict_sentences:
        print("Contradict:")
        for s in report.evidence.contradict_sentences:
            print("  -", s[:200] + ("..." if len(s) > 200 else ""))
    if verbose and report.evidence.sides:
        for i, side in enumerate(report.evidence.sides):
            print(f"  Side {i + 1}: {len(side.get('texts', []))} sentences, mass={side.get('total_mass', 0):.4f}")
    if report.summary:
        print("Summary:", report.summary)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate claim(s) against the corpus (evidence engine + agent)."
    )
    parser.add_argument("--claim", type=str, help="Single claim to evaluate.")
    parser.add_argument(
        "--claims-file",
        type=Path,
        help="Path to file with one claim per line.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Corpus data directory (graph, vocab, encoded_sentences). Overrides config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to anchor config JSON (default: config/anchor.json via wire).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show sides and extra detail.")
    args = parser.parse_args()

    config = get_config()
    if args.config and args.config.exists():
        import json
        with open(args.config, encoding="utf-8") as f:
            config = {**config, **json.load(f)}

    data_path = args.data
    if data_path is None:
        data_dir = config.get("align_data_dir") or config.get("ANCHOR_DATA_DIR")
        data_path = Path(data_dir) if data_dir and str(data_dir).strip() else None
    else:
        data_path = args.data.resolve() if args.data else None

    if not data_path or not data_path.exists():
        print("Data path missing or invalid. Use --data <path> or set align_data_dir in config.", file=sys.stderr)
        sys.exit(1)

    claims: list[str] | str
    if args.claims_file:
        if not args.claims_file.exists():
            print(f"Claims file not found: {args.claims_file}", file=sys.stderr)
            sys.exit(1)
        claims = [
            line.strip()
            for line in args.claims_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not claims:
            print("No non-empty claims in file.", file=sys.stderr)
            sys.exit(1)
    elif args.claim:
        claims = args.claim.strip()
        if not claims:
            print("Empty --claim.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide --claim \"...\" or --claims-file <path>.", file=sys.stderr)
        sys.exit(1)

    result = run_task(claims, data_path, config, concept_bundle=None, engine=None)

    if isinstance(result, AgentReport):
        _print_report(result, verbose=args.verbose)
    else:
        for report in result:
            _print_report(report, verbose=args.verbose)
        if result and result[0].summary:
            print("Aggregate:", result[0].summary)


if __name__ == "__main__":
    main()
