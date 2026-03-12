"""
Tests for anchor.next_token: retrieval next-token, bigram, hybrid mixture, sample.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anchor.corpus_graph import CorpusGraph
from anchor.next_token import (
    get_bigram_distribution,
    get_hybrid_next_token_distribution,
    get_next_token_distribution,
    sample_next_token,
)


class TestGetNextTokenDistribution:
    def test_empty_graph_returns_none(self, tmp_path: Path):
        encoded = tmp_path / "encoded.jsonl"
        encoded.write_text("")
        data = {"sentence_words": {"0": [1, 2]}, "word_cooccurrence": {}, "word_next": {}, "sentence_similar": {}}
        cg = CorpusGraph(data)
        assert get_next_token_distribution([1, 2, 3, 4, 5], cg, encoded) is None

    def test_with_context_index_returns_distribution(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "token_ids": [10, 20, 30, 40, 50, 99], "genre_id": "general"}) + "\n")
        data = {
            "sentence_words": {"0": [10, 20, 30, 40, 50, 99]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
            "context_to_sentences": {"context_length": 5, "index": {"10,20,30,40,50": [[0, 5]]}},
        }
        cg = CorpusGraph(data)
        dist = get_next_token_distribution([10, 20, 30, 40, 50], cg, tmp_path / "encoded.jsonl", min_hits=1)
        assert dist is not None
        assert 99 in dist
        assert abs(sum(dist.values()) - 1.0) < 1e-6


class TestGetBigramDistribution:
    def test_empty_returns_empty(self):
        cg = CorpusGraph({"sentence_words": {}, "word_cooccurrence": {}, "word_next": {}, "sentence_similar": {}})
        assert get_bigram_distribution(1, cg, 10) == {}

    def test_returns_normalized_probs(self):
        data = {
            "sentence_words": {"0": [1, 2, 2, 3]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 2, "3": 1}},
            "sentence_similar": {},
        }
        cg = CorpusGraph(data)
        dist = get_bigram_distribution(1, cg, 10, smoothing=0.01)
        assert 2 in dist and 3 in dist
        assert abs(sum(dist.values()) - 1.0) < 1e-6


class TestGetHybridNextTokenDistribution:
    def test_fallback_to_bigram_when_no_retrieval(self, tmp_path: Path):
        encoded = tmp_path / "encoded.jsonl"
        encoded.write_text("")
        data = {
            "sentence_words": {"0": [1, 2, 3]},
            "word_cooccurrence": {},
            "word_next": {"2": {"3": 1}},
            "sentence_similar": {},
        }
        cg = CorpusGraph(data)
        dist = get_hybrid_next_token_distribution(
            [1, 2], cg, encoded, vocab_size=10, genre_id="general", beta=0.7
        )
        assert dist
        assert 3 in dist

    def test_mixture_when_both_present(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "token_ids": [1, 2, 3, 4, 5, 100], "genre_id": "general"}) + "\n")
        data = {
            "sentence_words": {"0": [1, 2, 3, 4, 5, 100]},
            "word_cooccurrence": {},
            "word_next": {"5": {"100": 1, "6": 1}},
            "sentence_similar": {},
            "context_to_sentences": {"context_length": 5, "index": {"1,2,3,4,5": [[0, 5]]}},
        }
        cg = CorpusGraph(data)
        dist = get_hybrid_next_token_distribution(
            [1, 2, 3, 4, 5], cg, tmp_path / "encoded.jsonl", vocab_size=200, beta=0.7
        )
        assert dist
        assert 100 in dist or 6 in dist


class TestSampleNextToken:
    def test_returns_none_for_empty_distribution(self):
        assert sample_next_token({}) is None

    def test_returns_one_of_the_keys(self):
        dist = {1: 0.5, 2: 0.5}
        rng = MagicMock()
        rng.choices = lambda ids, weights, k: [1]
        out = sample_next_token(dist, rng=rng)
        assert out == 1

    def test_temperature_applied(self):
        dist = {1: 0.9, 2: 0.1}
        out = sample_next_token(dist, temperature=0.01)
        assert out in (1, 2)
