"""
Tests for anchor.next_sentence: get_next_sentences, get_next_sentences_from_text.
Errors, empty index/graph, genre filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_graph import CorpusGraph
from anchor.next_sentence import get_next_sentences, get_next_sentences_from_text


def _make_graph_two_sentences() -> CorpusGraph:
    data = {
        "sentence_words": {"0": [1, 2, 3], "1": [2, 3, 4]},
        "word_cooccurrence": {"1": [2, 3], "2": [1, 3, 4]},
        "word_next": {"1": {"2": 1}, "2": {"3": 1}, "3": {"4": 1}},
        "sentence_similar": {"0": [[1, 0.4]], "1": [[0, 0.4]]},
    }
    return CorpusGraph(data)


class TestGetNextSentences:
    """get_next_sentences: empty index, empty graph, genre filter."""

    def test_empty_index_returns_empty(self, tmp_path: Path):
        encoded = tmp_path / "encoded.jsonl"
        encoded.touch()
        graph = _make_graph_two_sentences()
        out = get_next_sentences([1, 2, 3], "general", graph, encoded, top_k=5)
        assert out == []

    def test_empty_graph_sentence_ids_handled(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "general", "token_ids": [1, 2], "text": "A B",
            }) + "\n")
        graph = CorpusGraph({
            "sentence_words": {},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        })
        out = get_next_sentences([1, 2], "general", graph, tmp_path / "encoded.jsonl")
        assert out == []

    def test_genre_filter(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "retirement", "token_ids": [1, 2, 3], "text": "One.",
            }) + "\n")
            f.write(json.dumps({
                "sentence_id": 1, "genre_id": "general", "token_ids": [2, 3, 4], "text": "Two.",
            }) + "\n")
        graph = _make_graph_two_sentences()
        # Best match to [1,2,3] is sentence 0; similar sentences include 1. Sentence 1 is "general".
        out_general = get_next_sentences([1, 2, 3], "general", graph, tmp_path / "encoded.jsonl")
        assert any("Two" in s for s in out_general)
        # Asking for "retirement": similar to best (0) is [1]; 1 is general, so no retirement in similar list
        out_retirement = get_next_sentences([1, 2, 3], "retirement", graph, tmp_path / "encoded.jsonl")
        assert isinstance(out_retirement, list)

    def test_missing_encoded_path_returns_empty(self):
        graph = _make_graph_two_sentences()
        out = get_next_sentences(
            [1, 2], "general", graph, Path("/nonexistent/encoded.jsonl")
        )
        assert out == []


class TestGetNextSentencesFromText:
    """get_next_sentences_from_text: empty word_to_id, unknown tokens."""

    def test_empty_word_to_id_returns_empty(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "general", "token_ids": [1], "text": "Hi",
            }) + "\n")
        graph = _make_graph_two_sentences()
        out = get_next_sentences_from_text("Hello world", "general", graph, tmp_path / "encoded.jsonl", {})
        assert out == []

    def test_no_overlap_with_vocab_returns_empty(self, tmp_path: Path):
        with open(tmp_path / "encoded.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "general", "token_ids": [1, 2], "text": "A B",
            }) + "\n")
        graph = _make_graph_two_sentences()
        word_to_id = {"xyz": 99}
        out = get_next_sentences_from_text("Hello world", "general", graph, tmp_path / "encoded.jsonl", word_to_id)
        assert out == []
