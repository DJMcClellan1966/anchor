"""
Tests for anchor.corpus_model: load_corpus_model, next_token_log_probs, sample_next_token.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_model import (
    load_corpus_model,
    next_token_log_probs,
    sample_next_token,
)


class TestLoadCorpusModel:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_corpus_model(tmp_path / "missing.json") is None

    def test_invalid_json_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json {")
        assert load_corpus_model(p) is None

    def test_valid_file_returns_dict(self, tmp_path: Path):
        p = tmp_path / "model.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": 3,
                "id_to_word": {"0": "a", "1": "b", "2": "c"},
                "transition": {"0": {"1": -0.5, "2": -0.5}, "1": {"2": 0.0}},
                "smoothing": 0.01,
            }, f)
        data = load_corpus_model(p)
        assert data is not None
        assert data["vocab_size"] == 3
        assert "0" in data["transition"] and "1" in data["transition"]


class TestNextTokenLogProbs:
    def test_returns_log_probs_for_curr(self):
        model = {"transition": {"5": {"10": -0.7, "11": -0.3}}}
        out = next_token_log_probs(5, model)
        assert 10 in out and 11 in out
        assert out[10] == pytest.approx(-0.7)
        assert out[11] == pytest.approx(-0.3)

    def test_unknown_curr_returns_empty(self):
        model = {"transition": {"1": {"2": 0.0}}}
        assert next_token_log_probs(99, model) == {}


class TestSampleNextTokenCorpusModel:
    def test_returns_none_when_no_row(self):
        model = {"transition": {}, "id_to_word": {}}
        assert sample_next_token(model, curr_token_id=0) is None

    def test_returns_one_of_support(self):
        model = {"transition": {"0": {"1": -0.5, "2": -0.5}}, "id_to_word": {}}
        out = sample_next_token(model, curr_token_id=0)
        assert out in (1, 2)
