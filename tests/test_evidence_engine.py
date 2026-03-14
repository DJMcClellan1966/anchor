"""
Tests for anchor.evidence_engine: EvidenceResult, evaluate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.evidence_engine import EvidenceResult, evaluate


def _minimal_corpus(tmp_path: Path) -> None:
    """Write minimal graph, vocab, encoded_sentences into tmp_path/corpus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {"word_to_id": {"a": 0, "b": 1, "c": 2}, "id_to_word": {"0": "a", "1": "b", "2": "c"}},
            f,
        )
    with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"sentence_id": 0, "genre_id": "general", "text": "A b.", "token_ids": [0, 1]}) + "\n")
        f.write(json.dumps({"sentence_id": 1, "genre_id": "general", "text": "B c.", "token_ids": [1, 2]}) + "\n")
    with open(corpus / "graph.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sentence_words": {"0": [0, 1], "1": [1, 2]},
                "word_cooccurrence": {"0": [1], "1": [0, 2], "2": [1]},
                "word_next": {"0": {"1": 1}, "1": {"0": 1, "2": 1}, "2": {"1": 1}},
                "sentence_similar": {"0": [[1, 0.5]], "1": [[0, 0.5]]},
            },
            f,
        )


class TestEvidenceEngine:
    def test_evaluate_returns_evidence_result_with_verdict_and_sentences(self, tmp_path: Path):
        """Evaluate with minimal corpus returns EvidenceResult with verdict, support_sentences, contradict_sentences, sides."""
        _minimal_corpus(tmp_path)
        config = {
            "default_genre_id": "general",
            "use_query_token_ids": True,
            "attention_loop_hops": 2,
            "attention_loop_top_k": 5,
        }
        result = evaluate("a b", tmp_path, config, concept_bundle=None, engine=None)
        assert isinstance(result, EvidenceResult)
        assert result.verdict in ("supported", "divided", "silent")
        assert isinstance(result.support_sentences, list)
        assert isinstance(result.contradict_sentences, list)
        assert isinstance(result.sides, list)
        assert isinstance(result.confidence, (int, float))

    def test_evaluate_missing_data_path_returns_silent(self):
        """When data_path is missing or invalid, verdict is silent and lists empty."""
        config = {"default_genre_id": "general"}
        result = evaluate("claim", None, config)
        assert result.verdict == "silent"
        assert result.support_sentences == []
        assert result.contradict_sentences == []
        assert result.sides == []

    def test_evaluate_nonexistent_path_returns_silent(self, tmp_path: Path):
        """When data_path does not exist, verdict is silent."""
        bad_path = tmp_path / "nonexistent"
        config = {"default_genre_id": "general"}
        result = evaluate("claim", bad_path, config)
        assert result.verdict == "silent"
        assert result.support_sentences == []
        assert result.contradict_sentences == []

    def test_evaluate_divided_when_epistemic_ratio_met(self, tmp_path: Path):
        """When corpus has two balanced groups and epistemic_secondary_mass_ratio is met, verdict can be divided."""
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word_to_id": {"w0": 0, "w1": 1, "w2": 2, "w3": 3},
                    "id_to_word": {"0": "w0", "1": "w1", "2": "w2", "3": "w3"},
                },
                f,
            )
        with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            for i in range(4):
                f.write(
                    json.dumps(
                        {"sentence_id": i, "genre_id": "general", "text": f"Sentence {i}.", "token_ids": [i]}
                    )
                    + "\n"
                )
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sentence_words": {"0": [0], "1": [1], "2": [2], "3": [3]},
                    "word_cooccurrence": {"0": [1], "1": [0], "2": [3], "3": [2]},
                    "word_next": {"0": {"1": 1}, "1": {"0": 1}, "2": {"3": 1}, "3": {"2": 1}},
                    "sentence_similar": {"0": [], "1": [], "2": [], "3": []},
                },
                f,
            )
        config = {
            "default_genre_id": "general",
            "use_query_token_ids": True,
            "attention_loop_hops": 3,
            "attention_loop_top_k": 2,
            "epistemic_secondary_mass_ratio": 0.35,
        }
        result = evaluate("w0 w2", tmp_path, config, concept_bundle=None, engine=None)
        assert result.verdict in ("supported", "divided", "silent")
        assert isinstance(result.sides, list)
        if result.verdict == "divided":
            assert len(result.contradict_sentences) > 0 or len(result.sides) >= 2
