"""
Evidence engine: evaluate a claim against the corpus and return a structured
verdict (supported / divided / silent) with support and contradict sentences.
No prose generation, no dictionary required, no critic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import graph_attention


@dataclass
class EvidenceResult:
    """Structured result of evaluating a claim against the corpus."""

    verdict: str  # "supported" | "divided" | "silent"
    support_sentences: list[str]
    contradict_sentences: list[str]
    sides: list[dict[str, Any]]  # [{sentence_ids, texts, total_mass}, ...]
    confidence: float = 0.0
    run_extras: dict[str, Any] | None = None


def evaluate(
    claim: str,
    data_path: Path | None,
    config: dict[str, Any],
    concept_bundle: dict[str, Any] | None = None,
    engine: Any = None,
) -> EvidenceResult:
    """
    Evaluate a claim against the corpus. Returns EvidenceResult with verdict,
    support_sentences, contradict_sentences, and sides.
    If concept_bundle is None, run_evidence builds it from tokenized claim (no dictionary).
    If data is missing or run_evidence returns None, returns verdict="silent" and empty lists.
    """
    silent = EvidenceResult(
        verdict="silent",
        support_sentences=[],
        contradict_sentences=[],
        sides=[],
        confidence=0.0,
        run_extras=None,
    )
    if not data_path or not data_path.exists():
        return silent
    result = graph_attention.run_evidence(
        claim, engine, config, data_path, concept_bundle=concept_bundle
    )
    if result is None:
        return silent
    evidence_dict, run_extras = result
    return EvidenceResult(
        verdict=evidence_dict.get("verdict", "silent"),
        support_sentences=list(evidence_dict.get("support_sentences") or []),
        contradict_sentences=list(evidence_dict.get("contradict_sentences") or []),
        sides=list(evidence_dict.get("sides") or []),
        confidence=float(evidence_dict.get("confidence", 0.0)),
        run_extras=run_extras,
    )
