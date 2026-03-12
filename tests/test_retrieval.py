"""
Tests for anchor.retrieval: get_concept_bundle, get_style_sentences, get_style_sentences_from_graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor import retrieval


class TestGetConceptBundle:
    def test_none_engine_returns_empty_bundle(self):
        b = retrieval.get_concept_bundle(None, "anything")
        assert b["terms"] == []
        assert b["definitions"] == {}

    def test_empty_query_returns_empty_bundle(self, mock_engine_with_context):
        b = retrieval.get_concept_bundle(mock_engine_with_context, "")
        assert b["terms"] == []
        b2 = retrieval.get_concept_bundle(mock_engine_with_context, "   ")
        assert b2["terms"] == []

    def test_engine_with_context_returns_terms_and_defs(self, mock_engine_with_context):
        b = retrieval.get_concept_bundle(mock_engine_with_context, "function")
        assert "function" in b["terms"]
        assert "code" in b["terms"]
        assert "function" in b["definitions"]
        # definition_map may be used instead of definitions; at least one def present
        assert len(b["definitions"]) >= 1


class TestGetStyleSentences:
    def test_none_data_dir_returns_empty(self):
        assert retrieval.get_style_sentences(None, None, {}, "retirement") == []

    def test_missing_data_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "missing_dir"
        assert retrieval.get_style_sentences(None, missing, {}, "retirement") == []

    def test_corpus_sentences_jsonl_filtered_by_genre(self, sample_sentences_jsonl):
        data_dir = sample_sentences_jsonl
        out = retrieval.get_style_sentences(
            None, data_dir, {"terms": ["Retirement"]}, genre_id="retirement"
        )
        assert any("Retirement" in s for s in out)
        out_gen = retrieval.get_style_sentences(
            None, data_dir, {"terms": []}, genre_id="general"
        )
        assert len(out_gen) >= 1

    def test_empty_concept_bundle_still_returns_sentences_if_no_filter(self, sample_sentences_jsonl):
        out = retrieval.get_style_sentences(
            None, sample_sentences_jsonl, {"terms": []}, genre_id="general"
        )
        assert isinstance(out, list)

    def test_per_genre_fallback(self, tmp_path):
        (tmp_path / "retirement").mkdir(exist_ok=True)
        with open(tmp_path / "retirement" / "genre_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write('{"text": "Retirement is good."}\n')
        out = retrieval.get_style_sentences(
            None, tmp_path, {"terms": []}, genre_id="retirement"
        )
        assert any("Retirement" in s for s in out)


class TestGetStyleSentencesFromGraph:
    def test_missing_graph_returns_empty(self, tmp_path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        out = retrieval.get_style_sentences_from_graph(tmp_path, {}, "general")
        assert out == []

    def test_missing_vocab_returns_empty(self, tmp_path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        with open(tmp_path / "corpus" / "graph.json", "w", encoding="utf-8") as f:
            json.dump({
                "sentence_words": {"0": [1]},
                "word_cooccurrence": {},
                "word_next": {},
                "sentence_similar": {"0": []},
            }, f)
        out = retrieval.get_style_sentences_from_graph(tmp_path, {"terms": ["x"]}, "general")
        assert out == []

    def test_with_vocab_and_encoded_returns_sentences(self, tmp_path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        with open(tmp_path / "corpus" / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"word_to_id": {"Hello": 1, "world": 2}, "id_to_word": {"1": "Hello", "2": "world"}}, f)
        with open(tmp_path / "corpus" / "graph.json", "w", encoding="utf-8") as f:
            json.dump({
                "sentence_words": {"0": [1, 2]},
                "word_cooccurrence": {"1": [2], "2": [1]},
                "word_next": {"1": {"2": 1}},
                "sentence_similar": {"0": []},
            }, f)
        with open(tmp_path / "corpus" / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "general", "token_ids": [1, 2], "text": "Hello world.",
            }) + "\n")
        out = retrieval.get_style_sentences_from_graph(
            tmp_path, {"terms": ["Hello"]}, "general"
        )
        assert any("Hello" in s for s in out)
