"""
Tests for anchor.naturalize: extend response using graph next-token patterns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.naturalize import naturalize


def test_naturalize_disabled_returns_unchanged():
    config = {"use_naturalize": False}
    assert naturalize("Hello world.", Path("data"), config) == "Hello world."


def test_naturalize_missing_graph_returns_unchanged(tmp_path: Path):
    config = {"use_naturalize": True}
    (tmp_path / "corpus").mkdir(exist_ok=True)
    assert naturalize("Hello world.", tmp_path, config) == "Hello world."


def test_naturalize_extends_when_possible(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    vocab = {"word_to_id": {"Hello": 0, "world": 1, "and": 2, "more": 3}, "id_to_word": {"0": "Hello", "1": "world", "2": "and", "3": "more"}}
    with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    graph_data = {
        "sentence_words": {"0": [0, 1]},
        "word_cooccurrence": {"0": [1], "1": [0, 2], "2": [3]},
        "word_next": {"0": {"1": 1}, "1": {"2": 1}, "2": {"3": 1}},
        "sentence_similar": {"0": []},
        "context_to_sentences": {"key_format": "ids", "context_length": 5, "index": {}},
    }
    with open(corpus / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f)
    config = {"use_naturalize": True, "naturalize_max_tokens": 5, "naturalize_context_length": 2}
    result = naturalize("Hello world", tmp_path, config)
    assert result is not None
    assert "Hello" in result and "world" in result
    assert len(result.split()) >= 2
