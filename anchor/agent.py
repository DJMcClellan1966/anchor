"""
Agent: orchestrates the evidence engine for one or many claims and returns a report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import evidence_engine
from .evidence_engine import EvidenceResult


@dataclass
class AgentReport:
    """Report for one claim: the claim and its evidence result, optional summary."""

    claim: str
    evidence: EvidenceResult
    summary: str | None = None


def run_task(
    task: str | list[str],
    data_path: Path | None,
    config: dict[str, Any],
    concept_bundle: dict[str, Any] | None = None,
    engine: Any = None,
) -> AgentReport | list[AgentReport]:
    """
    Run the evidence engine on one claim or a list of claims.
    Returns one AgentReport for a single claim, or a list of AgentReports for multiple claims.
    When task is a list, a simple summary (N supported, M divided, K silent) is set on each
    report's summary field for the aggregate; per-report summary is left None.
    """
    if isinstance(task, str):
        evidence = evidence_engine.evaluate(
            task, data_path, config,
            concept_bundle=concept_bundle,
            engine=engine,
        )
        return AgentReport(claim=task, evidence=evidence, summary=None)

    reports: list[AgentReport] = []
    for claim in task:
        evidence = evidence_engine.evaluate(
            claim, data_path, config,
            concept_bundle=concept_bundle,
            engine=engine,
        )
        reports.append(AgentReport(claim=claim, evidence=evidence, summary=None))

    supported = sum(1 for r in reports if r.evidence.verdict == "supported")
    divided = sum(1 for r in reports if r.evidence.verdict == "divided")
    silent = sum(1 for r in reports if r.evidence.verdict == "silent")
    summary = f"{supported} supported, {divided} divided, {silent} silent."
    for r in reports:
        r.summary = summary
    return reports
