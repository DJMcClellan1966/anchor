"""
Tests for anchor.agent: AgentReport, run_task.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.agent import AgentReport, run_task
from anchor.evidence_engine import EvidenceResult


def _minimal_corpus(tmp_path: Path) -> None:
    """Write minimal graph, vocab, encoded_sentences into tmp_path/corpus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {"word_to_id": {"a": 0, "b": 1}, "id_to_word": {"0": "a", "1": "b"}},
            f,
        )
    with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"sentence_id": 0, "genre_id": "general", "text": "A b.", "token_ids": [0, 1]}) + "\n")
    with open(corpus / "graph.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sentence_words": {"0": [0, 1]},
                "word_cooccurrence": {"0": [1], "1": [0]},
                "word_next": {"0": {"1": 1}},
                "sentence_similar": {"0": []},
            },
            f,
        )


class TestAgent:
    def test_run_task_single_claim_returns_one_agent_report(self, tmp_path: Path):
        """run_task(claim, data_path, config) returns one AgentReport with evidence.verdict in supported/divided/silent."""
        _minimal_corpus(tmp_path)
        config = {
            "default_genre_id": "general",
            "use_query_token_ids": True,
            "attention_loop_hops": 2,
            "attention_loop_top_k": 5,
        }
        report = run_task("a b", tmp_path, config, concept_bundle=None, engine=None)
        assert isinstance(report, AgentReport)
        assert report.claim == "a b"
        assert isinstance(report.evidence, EvidenceResult)
        assert report.evidence.verdict in ("supported", "divided", "silent")
        assert report.summary is None

    def test_run_task_multiple_claims_returns_list_of_agent_reports(self, tmp_path: Path):
        """run_task([c1, c2], data_path, config) returns list of two AgentReports with summary set."""
        _minimal_corpus(tmp_path)
        config = {
            "default_genre_id": "general",
            "use_query_token_ids": True,
            "attention_loop_hops": 2,
            "attention_loop_top_k": 5,
        }
        reports = run_task(["a", "b"], tmp_path, config, concept_bundle=None, engine=None)
        assert isinstance(reports, list)
        assert len(reports) == 2
        assert reports[0].claim == "a"
        assert reports[1].claim == "b"
        for r in reports:
            assert isinstance(r, AgentReport)
            assert isinstance(r.evidence, EvidenceResult)
            assert r.evidence.verdict in ("supported", "divided", "silent")
        assert reports[0].summary is not None
        assert "supported" in reports[0].summary or "divided" in reports[0].summary or "silent" in reports[0].summary
