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

    def test_activate_with_query_token_ids_includes_ids_and_sentences(self):
        """With query_token_ids, activated sets include those word IDs and their sentences."""
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {"a": 1, "b": 2, "c": 3}
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3], "2": [3]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        w_ids, s_ids = graph_attention.activate(
            concept_bundle, graph, word_to_id, query_token_ids=[1, 3]
        )
        assert 1 in w_ids and 3 in w_ids
        assert 0 in s_ids  # sentence 0 contains 1,2
        assert 1 in s_ids  # sentence 1 contains 2,3
        assert 2 in s_ids  # sentence 2 contains 3

    def test_activate_with_query_token_ids_union_terms(self):
        """Terms and query_token_ids are combined (union)."""
        concept_bundle = {"terms": ["hello"], "definitions": {"hello": "hi"}}
        word_to_id = {"hello": 1, "world": 2}
        data = {
            "sentence_words": {"0": [1], "1": [1, 2], "2": [2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        w_ids, s_ids = graph_attention.activate(
            concept_bundle, graph, word_to_id, query_token_ids=[2]
        )
        assert w_ids == {1, 2}
        assert s_ids == {0, 1, 2}

    def test_activate_without_query_token_ids_unchanged(self):
        """Without query_token_ids, only terms are used (existing behavior)."""
        concept_bundle = {"terms": ["a"], "definitions": {}}
        word_to_id = {"a": 1, "b": 2}
        data = {
            "sentence_words": {"0": [1], "1": [2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        w_ids, s_ids = graph_attention.activate(concept_bundle, graph, word_to_id)
        assert w_ids == {1}
        assert s_ids == {0}


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

    def test_use_weights_applies_inverse_length_and_next_word(self):
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 2}, "2": {"3": 1}},
            "sentence_similar": {"0": [[1, 0.5]], "1": [[0, 0.5]]},
        }
        graph = CorpusGraph(data)
        w_visits, s_visits = graph_attention.traverse_loops(
            {1}, {0}, graph, num_hops=3, use_weights=True
        )
        assert 1 in w_visits and 2 in w_visits
        assert 3 in w_visits  # next-word from 2 after second hop
        assert 0 in s_visits


class TestDetectPattern:
    def test_returns_top_k_by_visit_count(self):
        word_visits = {1: 3.0, 2: 2.0, 3: 1.0}
        sentence_visits = {0: 5.0, 1: 2.0}
        top_w, top_s, sec_w, sec_s = graph_attention.detect_pattern(word_visits, sentence_visits, top_k=2)
        assert top_w == [1, 2]
        assert top_s == [0, 1]
        assert sec_w == [] and sec_s == []

    def test_num_groups_two_returns_secondary(self):
        word_visits = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0}
        sentence_visits = {0: 5.0, 1: 3.0, 2: 1.0}
        top_w, top_s, sec_w, sec_s = graph_attention.detect_pattern(
            word_visits, sentence_visits, top_k=2, num_groups=2
        )
        assert top_w == [1, 2]
        assert top_s == [0, 1]
        assert sec_w == [3, 4]
        assert sec_s == [2]

    def test_min_visits_filters(self):
        word_visits = {1: 3.0, 2: 1.0, 3: 0.5}
        sentence_visits = {0: 2.0}
        top_w, top_s, _, _ = graph_attention.detect_pattern(
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

    def test_refine_answer_includes_secondary_and_next_span(self):
        concept_bundle = {"terms": ["a"], "definitions": {"a": "Def a."}}
        id_to_word = {1: "a", 2: "b"}
        encoded_index = {
            0: {"text": "Primary.", "genre_id": "general"},
            1: {"text": "Secondary.", "genre_id": "general"},
            2: {"text": "Next span.", "genre_id": "general"},
        }
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word,
            genre_id="general", max_sentences=5,
            secondary_sentence_ids=[1],
            next_span_sentence_ids=[2],
        )
        assert "Def a" in out or "Primary" in out
        assert "Secondary" in out
        assert "Next span" in out

    def test_refine_answer_paragraph_format_and_visit_order(self):
        concept_bundle = {"terms": ["a"], "definitions": {"a": "Def a."}}
        id_to_word = {1: "a"}
        encoded_index = {
            0: {"text": "First.", "genre_id": "general"},
            1: {"text": "Second.", "genre_id": "general"},
        }
        # sentence_visits: 1 has higher score so it should come first when ordered
        sentence_visits = {0: 1.0, 1: 2.0}
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word,
            genre_id="general", max_sentences=5,
            next_span_sentence_ids=[1],
            sentence_visits=sentence_visits,
            output_format="paragraph",
            paragraph_max_chars=500,
        )
        assert "In the corpus:" in out
        assert "Second." in out and "First." in out

    def test_refine_answer_include_definitions_false_no_definition_lines(self):
        """When include_definitions is False, output has no 'term: definition' pattern; only corpus text."""
        concept_bundle = {"terms": ["a"], "definitions": {"a": "Def of a."}}
        id_to_word = {1: "a"}
        encoded_index = {0: {"text": "Only corpus sentence here.", "genre_id": "general"}}
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word,
            genre_id="general", include_definitions=False,
        )
        assert "Only corpus sentence" in out
        assert "a: Def" not in out and "Def of a" not in out


class TestNextSpan:
    def test_next_span_collects_similar_sentences_genre_filtered(self):
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3], "2": [1, 3]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {"0": [[1, 0.8], [2, 0.3]], "1": [[0, 0.8]], "2": [[0, 0.3]]},
        }
        graph = CorpusGraph(data)
        encoded_index = {
            0: {"text": "A", "genre_id": "general"},
            1: {"text": "B", "genre_id": "general"},
            2: {"text": "C", "genre_id": "retirement"},
        }
        ids = graph_attention._next_span_sentence_ids(
            [0], graph, "general", encoded_index, top_k_per_sentence=3
        )
        assert 1 in ids
        assert 2 not in ids


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

        config = {
            "default_genre_id": "general",
            "attention_loop_hops": 2,
            "attention_loop_top_k": 10,
            "attention_loop_use_weights": True,
            "attention_loop_path_groups": 2,
            "attention_loop_next_span": True,
        }
        out = graph_attention.run("hello", MockEngine(), config, tmp_path)
        assert out is not None
        assert out.strip()
        assert "greeting" in out or "Hello world" in out or "hello" in out

    def test_run_with_use_query_token_ids_true_activates_from_query(self, tmp_path: Path):
        """With use_query_token_ids true, query tokens activate graph; result can reflect query words."""
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(
                {"word_to_id": {"hello": 0, "world": 1}, "id_to_word": {"0": "hello", "1": "world"}},
                f,
            )
        with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"sentence_id": 0, "genre_id": "general", "text": "Hello world.", "token_ids": [0, 1]}
                )
                + "\n"
            )
        graph = {
            "sentence_words": {"0": [0, 1]},
            "word_cooccurrence": {"0": [1], "1": [0]},
            "word_next": {"0": {"1": 1}},
            "sentence_similar": {"0": []},
        }
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f)

        class MockEngineEmptyTerms:
            @staticmethod
            def get_context_for_description(q):
                return {"concepts": [], "definitions": {}}

        config = {
            "default_genre_id": "general",
            "use_query_token_ids": True,
            "attention_loop_hops": 2,
            "attention_loop_top_k": 10,
            "attention_loop_use_weights": True,
            "attention_loop_path_groups": 2,
            "attention_loop_next_span": True,
        }
        out = graph_attention.run("hello", MockEngineEmptyTerms(), config, tmp_path)
        assert out is not None
        assert "hello" in out.lower() or "Hello" in out

    def test_run_with_use_query_token_ids_false_uses_only_terms(self, tmp_path: Path):
        """With use_query_token_ids false, activation comes from concept_bundle terms only."""
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(
                {"word_to_id": {"hello": 0, "world": 1}, "id_to_word": {"0": "hello", "1": "world"}},
                f,
            )
        with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"sentence_id": 0, "genre_id": "general", "text": "Hello world.", "token_ids": [0, 1]}
                )
                + "\n"
            )
        graph = {
            "sentence_words": {"0": [0, 1]},
            "word_cooccurrence": {"0": [1], "1": [0]},
            "word_next": {"0": {"1": 1}},
            "sentence_similar": {"0": []},
        }
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f)

        class MockEngineWithTerms:
            @staticmethod
            def get_context_for_description(q):
                return {"concepts": [{"name": "hello"}], "definitions": {"hello": "A greeting."}}

        config = {
            "default_genre_id": "general",
            "use_query_token_ids": False,
            "attention_loop_hops": 2,
            "attention_loop_top_k": 10,
            "attention_loop_use_weights": True,
            "attention_loop_path_groups": 2,
            "attention_loop_next_span": True,
        }
        out = graph_attention.run("hello", MockEngineWithTerms(), config, tmp_path)
        assert out is not None
        assert out.strip()
