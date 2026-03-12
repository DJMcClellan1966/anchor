"""
Tests for anchor.graph_attention: activate, traverse_loops, detect_pattern, refine_answer, run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_graph import CorpusGraph
from anchor import graph_attention


class TestActivate:
    def test_activates_word_and_sentence_ids_from_terms(self):
        concept_bundle = {"terms": ["hello", "world"], "definitions": {"hello": "a greeting"}}
        word_to_id = {"hello": 1, "world": 2, "a": 3}
        data = {
            "sentence_words": {"0": [1, 2], "1": [1, 3, 2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        w_ids, s_ids = graph_attention.activate(concept_bundle, graph, word_to_id)
        assert w_ids == {1, 2}
        assert s_ids == {0, 1}

    def test_empty_terms_activates_all_sentences(self):
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {}
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        w_ids, s_ids = graph_attention.activate(concept_bundle, graph, word_to_id)
        assert w_ids == set()
        assert s_ids == {0, 1}


class TestTraverseLoops:
    def test_propagates_to_sentences_and_back_to_words(self):
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3], "2": [1, 3]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {"0": [[1, 0.5]], "1": [[0, 0.5], [2, 0.3]], "2": [[1, 0.2]]},
        }
        graph = CorpusGraph(data)
        w_visits, s_visits = graph_attention.traverse_loops(
            {1}, {0}, graph, num_hops=2
        )
        assert 1 in w_visits and 2 in w_visits
        assert 0 in s_visits
        assert s_visits[0] >= 1.0
        assert w_visits[1] >= 1.0

    def test_genre_filter_when_encoded_index_provided(self):
        data = {
            "sentence_words": {"0": [1], "1": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {"0": [[1, 0.8]], "1": [[0, 0.8]]},
        }
        graph = CorpusGraph(data)
        encoded_index = {0: {"text": "Only general.", "genre_id": "general"}, 1: {"text": "Only retirement.", "genre_id": "retirement"}}
        w_visits, s_visits = graph_attention.traverse_loops(
            {1}, {0}, graph, num_hops=2,
            genre_id="general", encoded_index=encoded_index
        )
        assert 0 in s_visits
        assert s_visits.get(0) >= 1.0
        assert 1 not in s_visits


class TestDetectPattern:
    def test_returns_top_k_by_visit_count(self):
        word_visits = {1: 3.0, 2: 2.0, 3: 1.0}
        sentence_visits = {0: 5.0, 1: 2.0}
        top_w, top_s = graph_attention.detect_pattern(word_visits, sentence_visits, top_k=2)
        assert top_w == [1, 2]
        assert top_s == [0, 1]

    def test_min_visits_filters(self):
        word_visits = {1: 3.0, 2: 1.0, 3: 0.5}
        sentence_visits = {0: 2.0}
        top_w, top_s = graph_attention.detect_pattern(
            word_visits, sentence_visits, top_k=10, min_visits=1.5
        )
        assert 1 in top_w
        assert 2 not in top_w
        assert 3 not in top_w
        assert 0 in top_s


class TestRefineAnswer:
    def test_uses_only_definitions_and_corpus_text(self):
        pattern_word_ids = [1, 2]
        pattern_sentence_ids = [0]
        concept_bundle = {"terms": ["a", "b"], "definitions": {"a": "Def of a.", "b": "Def of b."}}
        id_to_word = {1: "a", 2: "b", 3: "c"}
        encoded_index = {0: {"text": "Sentence from corpus.", "genre_id": "general"}}
        out = graph_attention.refine_answer(
            pattern_word_ids, pattern_sentence_ids,
            concept_bundle, encoded_index, id_to_word, genre_id="general"
        )
        assert "Def of a" in out or "Def of b" in out
        assert "Sentence from corpus" in out
        assert "Def of a" in out or "Def of b" in out

    def test_no_invented_text(self):
        concept_bundle = {"terms": ["x"], "definitions": {"x": "Real def."}}
        encoded_index = {0: {"text": "Real sentence.", "genre_id": "general"}}
        id_to_word = {1: "x"}
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word, genre_id="general"
        )
        assert "Real def" in out or "Real sentence" in out
        assert "Real def." in out or "Real sentence." in out

    def test_empty_pattern_returns_concepts_or_message(self):
        concept_bundle = {"terms": ["t"], "definitions": {}}
        out = graph_attention.refine_answer(
            [], [], concept_bundle, {}, {}, genre_id="general"
        )
        assert "No pattern" in out or "Concepts" in out


class TestRun:
    def test_returns_none_when_data_path_missing(self):
        out = graph_attention.run("q", None, {}, None)
        assert out is None

    def test_returns_grounded_response_with_full_data(self, tmp_path: Path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"word_to_id": {"hello": 0, "world": 1}, "id_to_word": {"0": "hello", "1": "world"}}, f)
        with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "genre_id": "general", "text": "Hello world.", "token_ids": [0, 1]}) + "\n")
        graph = {
            "sentence_words": {"0": [0, 1]},
            "word_cooccurrence": {"0": [1], "1": [0]},
            "word_next": {"0": {"1": 1}},
            "sentence_similar": {"0": []},
        }
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f)

        class MockEngine:
            @staticmethod
            def get_context_for_description(q):
                return {"concepts": [{"name": "hello"}], "definitions": {"hello": "A greeting."}}

        config = {"default_genre_id": "general", "attention_loop_hops": 2, "attention_loop_top_k": 10}
        out = graph_attention.run("hello", MockEngine(), config, tmp_path)
        assert out is not None
        assert out.strip()
        assert "greeting" in out or "Hello world" in out or "hello" in out
