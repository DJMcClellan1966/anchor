"""
Tests for anchor.critic: extract_content_terms, dictionary_score, critic_decision, score_and_decide.
"""
from __future__ import annotations

import pytest

from anchor.critic import (
    critic_decision,
    dictionary_score,
    extract_content_terms,
    score_and_decide,
    terms_in_graph,
)


class TestExtractContentTerms:
    def test_empty_string(self):
        assert extract_content_terms("") == []
        assert extract_content_terms("   ") == []

    def test_stopwords_excluded(self):
        out = extract_content_terms("the cat and the dog")
        assert "the" not in out
        assert "and" not in out
        assert "cat" in out
        assert "dog" in out

    def test_short_tokens_excluded(self):
        out = extract_content_terms("a bc de fgh", min_len=3)
        assert "a" not in out
        assert "bc" not in out
        assert "de" not in out
        assert "fgh" in out
        assert len(out) <= 80

    def test_max_terms_cap(self):
        words = " ".join(f"word{i}" for i in range(100))
        out = extract_content_terms(words, max_terms=10)
        assert len(out) <= 10


class TestTermsInGraph:
    def test_none_engine_returns_empty(self):
        assert terms_in_graph(["foo"], None) == set()

    def test_empty_terms_returns_empty(self):
        assert terms_in_graph([], object()) == set()


class TestDictionaryScore:
    def test_empty_text_returns_perfect_score(self):
        score, grounded, content = dictionary_score("", None)
        assert score == 1.0
        assert content == 0

    def test_none_engine_zero_grounded(self):
        score, grounded, content = dictionary_score("hello world", None)
        assert grounded == 0
        assert content > 0


class TestCriticDecision:
    def test_accept_above_threshold(self):
        decision, warn = critic_decision(0.5, accept_threshold=0.25)
        assert decision == "accept"
        assert warn is False

    def test_reject_below_low(self):
        decision, warn = critic_decision(0.1, accept_threshold=0.25, low_warn_threshold=0.15)
        assert decision == "reject"
        assert warn is True

    def test_warn_between(self):
        decision, warn = critic_decision(0.2, accept_threshold=0.25, low_warn_threshold=0.15)
        assert decision == "warn"
        assert warn is True

    def test_from_config(self):
        decision, _ = critic_decision(0.3, config={"critic_accept_threshold": 0.25})
        assert decision == "accept"


class TestScoreAndDecide:
    def test_returns_all_keys(self):
        out = score_and_decide("Some content here.", None)
        assert "score" in out
        assert "num_grounded" in out
        assert "num_content" in out
        assert "decision" in out
        assert "show_warning" in out
        assert "message" in out
        assert out["decision"] in ("accept", "warn", "reject")
