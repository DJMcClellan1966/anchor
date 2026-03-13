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

    def test_refine_answer_prefers_definition_from_store_when_term_in_encoded_index(self):
        """Unified store: when encoded_index has a row with term=X, use its text as definition (not concept_bundle)."""
        concept_bundle = {"terms": ["apple"], "definitions": {"apple": "Fallback from engine."}}
        id_to_word = {1: "apple"}
        # Sentence 0 = corpus; sentence 1 = definition row from store (source=dictionary, term=apple)
        encoded_index = {
            0: {"text": "Some corpus sentence.", "genre_id": "general"},
            1: {"text": "apple: A fruit from the store.", "genre_id": "definitional", "source": "dictionary", "term": "apple"},
        }
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word,
            genre_id="general", max_sentences=5, include_definitions=True,
        )
        # Should use store text (unified), not "Fallback from engine"
        assert "A fruit from the store" in out or "apple: A fruit from the store" in out
        assert "Fallback from engine" not in out

    def test_refine_answer_picks_highest_visit_sid_when_multiple_senses(self):
        """With term_to_sids (multiple sids per term), pick the sid with highest sentence_visits."""
        concept_bundle = {"terms": ["apple"], "definitions": {"apple": "Fallback."}}
        id_to_word = {1: "apple"}
        # Two definition rows for "apple" (two senses): sid 1 has lower visits, sid 2 has higher
        encoded_index = {
            0: {"text": "Corpus.", "genre_id": "general"},
            1: {"text": "apple: A fruit.", "genre_id": "definitional", "source": "dictionary", "term": "apple"},
            2: {"text": "apple: A tech company.", "genre_id": "definitional", "source": "dictionary", "term": "apple"},
        }
        sentence_visits = {0: 1.0, 1: 0.5, 2: 2.0}  # sid 2 has highest
        out = graph_attention.refine_answer(
            [1], [0], concept_bundle, encoded_index, id_to_word,
            genre_id="general", max_sentences=5, include_definitions=True,
            sentence_visits=sentence_visits,
        )
        # Should pick sid 2 (highest visit) for definition text
        assert "A tech company" in out
        assert "A fruit" not in out or "A tech company" in out

    def test_refine_answer_return_sources_returns_tuple_with_sources(self):
        """When return_sources=True, returns (text, source_records) with definition and sentence types."""
        concept_bundle = {"terms": ["a", "b"], "definitions": {"a": "Def a.", "b": "Def b."}}
        id_to_word = {1: "a", 2: "b"}
        encoded_index = {
            0: {"text": "First sentence.", "genre_id": "general"},
            1: {"text": "Second.", "genre_id": "general"},
        }
        result = graph_attention.refine_answer(
            [1, 2], [0, 1], concept_bundle, encoded_index, id_to_word,
            genre_id="general", max_sentences=5, include_definitions=True,
            return_sources=True,
        )
        assert isinstance(result, tuple)
        text, sources = result
        assert isinstance(text, str)
        assert isinstance(sources, list)
        assert "Def a" in text or "Def b" in text
        def_sources = [s for s in sources if s.get("type") == "definition"]
        sent_sources = [s for s in sources if s.get("type") == "sentence"]
        assert len(def_sources) >= 1
        assert any("term" in s for s in def_sources)
        assert len(sent_sources) >= 1
        assert any("sentence_id" in s for s in sent_sources)


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


class TestNormalizeVisitDict:
    def test_normalizes_to_sum_one(self):
        d = {1: 1.0, 2: 2.0, 3: 1.0}
        out = graph_attention.normalize_visit_dict(d)
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert out[1] == 0.25 and out[2] == 0.5 and out[3] == 0.25

    def test_empty_or_zero_sum_unchanged(self):
        assert graph_attention.normalize_visit_dict({}) == {}
        assert graph_attention.normalize_visit_dict({1: 0.0, 2: 0.0}) == {1: 0.0, 2: 0.0}


class TestEmbedAnchor:
    def test_returns_initial_v_W_and_v_S_from_terms_and_query_token_ids(self):
        concept_bundle = {"terms": ["hello", "world"], "definitions": {}}
        word_to_id = {"hello": 1, "world": 2}
        data = {
            "sentence_words": {"0": [1, 2], "1": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W_0, v_S_0 = graph_attention.embed_anchor(
            concept_bundle, graph, word_to_id, query_token_ids=None
        )
        assert v_W_0 == {1: 1.0, 2: 1.0}
        assert v_S_0 == {0: 1.0, 1: 1.0}

    def test_embed_anchor_with_query_token_ids_includes_them(self):
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {"a": 1, "b": 2}
        data = {
            "sentence_words": {"0": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W_0, v_S_0 = graph_attention.embed_anchor(
            concept_bundle, graph, word_to_id, query_token_ids=[1, 2]
        )
        assert 1 in v_W_0 and 2 in v_W_0
        assert 0 in v_S_0

    def test_embed_anchor_with_use_definition_words_adds_definition_tokens_to_v_W(self):
        concept_bundle = {
            "terms": ["apple"],
            "definitions": {"apple": "A fruit that grows on trees."},
        }
        word_to_id = {"apple": 1, "A": 2, "fruit": 3, "that": 4, "grows": 5, "on": 6, "trees": 7, ".": 8}
        data = {
            "sentence_words": {"0": [1, 2, 3], "1": [3, 5, 7]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W_0, v_S_0 = graph_attention.embed_anchor(
            concept_bundle, graph, word_to_id,
            use_definition_words=True, definition_word_weight=0.5,
        )
        assert 1 in v_W_0
        assert v_W_0[1] >= 1.0
        for w in ("A", "fruit", "that", "grows", "on", "trees"):
            if w in word_to_id:
                assert word_to_id[w] in v_W_0
                assert v_W_0[word_to_id[w]] >= 0.5

    def test_embed_anchor_list_defn_adds_tokens_from_all_senses(self):
        """When definitions[term] is a list (multiple senses), all sense strings contribute to v_W."""
        concept_bundle = {
            "terms": ["function"],
            "definitions": {
                "function": ["A relation from inputs to outputs.", "A role or purpose."],
            },
        }
        word_to_id = {"function": 1, "A": 2, "relation": 3, "role": 4, "purpose": 5, "inputs": 6, "outputs": 7}
        data = {
            "sentence_words": {"0": [1, 2, 3], "1": [1, 4, 5]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W_0, _ = graph_attention.embed_anchor(
            concept_bundle, graph, word_to_id,
            use_definition_words=True, definition_word_weight=0.5,
        )
        # Tokens from first sense
        assert word_to_id["relation"] in v_W_0 or word_to_id["inputs"] in v_W_0 or word_to_id["outputs"] in v_W_0
        # Tokens from second sense
        assert word_to_id["role"] in v_W_0 or word_to_id["purpose"] in v_W_0


class TestPropagationLayer:
    def test_use_cooccurrence_spreads_to_cooccurring_words(self):
        data = {
            "sentence_words": {"0": [1, 2, 3]},
            "word_cooccurrence": {"1": [2, 3], "2": [1, 3], "3": [1, 2]},
            "word_next": {},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W = {1: 1.0}
        v_S = {0: 1.0}
        v_W_out, v_S_out = graph_attention.propagation_layer(
            v_W, v_S, graph, genre_id=None, encoded_index=None, use_weights=False,
            use_cooccurrence=True, use_backward=False,
        )
        assert 2 in v_W_out and 3 in v_W_out
        assert v_W_out.get(2, 0) > 0 and v_W_out.get(3, 0) > 0

    def test_use_backward_spreads_via_prev_word(self):
        data = {
            "sentence_words": {"0": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 1}},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        v_W = {2: 1.0}
        v_S = {}
        v_W_out, _ = graph_attention.propagation_layer(
            v_W, v_S, graph, genre_id=None, encoded_index=None, use_weights=False,
            use_cooccurrence=False, use_backward=True,
        )
        assert 1 in v_W_out
        assert v_W_out[1] > 0


class TestRunLayers:
    def test_run_layers_with_normalize_true_yields_distributions(self):
        data = {
            "sentence_words": {"0": [1, 2], "1": [2, 3]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 1}, "2": {"3": 1}},
            "sentence_similar": {"0": [[1, 0.5]], "1": [[0, 0.5]]},
        }
        graph = CorpusGraph(data)
        v_W_0 = {1: 1.0}
        v_S_0 = {0: 1.0}
        v_W, v_S = graph_attention.run_layers(
            v_W_0, v_S_0, graph, num_hops=3, normalize=True,
            genre_id=None, encoded_index=None, use_weights=True,
        )
        total_w = sum(v_W.values())
        total_s = sum(v_S.values())
        assert abs(total_w - 1.0) < 1e-6 or total_w == 0
        assert abs(total_s - 1.0) < 1e-6 or total_s == 0


class TestOutputHead:
    def test_returns_distribution_sum_one(self):
        v_W = {1: 2.0, 2: 4.0, 3: 4.0}
        p = graph_attention.output_head(v_W)
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert 1 in p and 2 in p and 3 in p

    def test_empty_input_returns_empty(self):
        assert graph_attention.output_head({}) == {}

    def test_output_head_dict_boost_favors_dict_terms(self):
        v_W = {1: 1.0, 2: 1.0, 3: 1.0}
        p = graph_attention.output_head(v_W, dict_term_ids={1, 2}, dict_boost=1.0)
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert p[1] > p[3] and p[2] > p[3]


class TestGenerateAutoregressive:
    def test_returns_non_empty_string_with_mock_graph(self):
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {"a": 1, "b": 2}
        id_to_word = {1: "a", 2: "b"}
        data = {
            "sentence_words": {"0": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 1}},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        encoded_index = {0: {"text": "a b", "genre_id": "general"}}
        config = {
            "autoregressive_max_tokens": 5,
            "autoregressive_context_window": 10,
            "autoregressive_stop_at_sentence_end": False,
            "attention_loop_hops": 2,
            "attention_loop_use_weights": True,
            "use_normalized_layers": True,
            "include_definitions_in_response": False,
        }
        out = graph_attention.generate_autoregressive(
            "a", concept_bundle, config, graph,
            word_to_id, id_to_word, encoded_index, genre_id="general",
        )
        assert isinstance(out, str)
        assert out.strip()

    def test_stops_at_max_tokens(self):
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {"x": 1, "y": 2}
        id_to_word = {1: "x", 2: "y"}
        data = {
            "sentence_words": {"0": [1, 2]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 1}, "2": {"1": 1}},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        encoded_index = {0: {"text": "x y", "genre_id": "general"}}
        config = {
            "autoregressive_max_tokens": 2,
            "autoregressive_context_window": 10,
            "autoregressive_stop_at_sentence_end": False,
            "attention_loop_hops": 2,
            "attention_loop_use_weights": True,
            "use_normalized_layers": True,
            "include_definitions_in_response": False,
        }
        out = graph_attention.generate_autoregressive(
            "x", concept_bundle, config, graph,
            word_to_id, id_to_word, encoded_index, genre_id="general",
        )
        words = out.split()
        assert len(words) <= 3  # at most 2 generated + possible punctuation

    def test_stops_at_sentence_end_when_configured(self):
        concept_bundle = {"terms": [], "definitions": {}}
        word_to_id = {"a": 1, "b": 2, ".": 3}
        id_to_word = {1: "a", 2: "b", 3: "."}
        data = {
            "sentence_words": {"0": [1, 2, 3]},
            "word_cooccurrence": {},
            "word_next": {"1": {"2": 1}, "2": {"3": 1}},
            "sentence_similar": {},
        }
        graph = CorpusGraph(data)
        encoded_index = {0: {"text": "a b .", "genre_id": "general"}}
        config = {
            "autoregressive_max_tokens": 20,
            "autoregressive_context_window": 10,
            "autoregressive_stop_at_sentence_end": True,
            "attention_loop_hops": 2,
            "attention_loop_use_weights": True,
            "use_normalized_layers": True,
            "include_definitions_in_response": False,
        }
        out = graph_attention.generate_autoregressive(
            "a", concept_bundle, config, graph,
            word_to_id, id_to_word, encoded_index, genre_id="general",
        )
        assert isinstance(out, str)
        assert out.strip()


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

    def test_run_with_use_autoregressive_generation_returns_string(self, tmp_path: Path):
        """When use_autoregressive_generation is True, run() uses autoregressive path and returns a string."""
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

        class MockEngine:
            @staticmethod
            def get_context_for_description(q):
                return {"concepts": [], "definitions": {}}

        config = {
            "default_genre_id": "general",
            "use_autoregressive_generation": True,
            "autoregressive_max_tokens": 5,
            "autoregressive_context_window": 10,
            "autoregressive_stop_at_sentence_end": False,
            "attention_loop_hops": 2,
            "attention_loop_use_weights": True,
            "use_normalized_layers": True,
        }
        out = graph_attention.run("hello", MockEngine(), config, tmp_path)
        assert out is not None
        assert isinstance(out, str)
        assert out.strip()
