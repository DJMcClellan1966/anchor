"""
Try to break: invalid input, bad types, malformed data, edge values.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_vocab import build_vocab, encode_sentences, load_vocab, tokenize
from anchor.corpus_graph import CorpusGraph, build_graph, load_graph
from anchor.critic import critic_decision, dictionary_score, extract_content_terms, score_and_decide
from anchor.retrieval import get_concept_bundle, get_style_sentences_from_graph


class TestBreakageCorpusVocab:
    def test_tokenize_none(self):
        assert tokenize(None) == []

    def test_build_vocab_corpus_path_nonexistent(self, tmp_path):
        w2i, i2w = build_vocab(tmp_path / "does_not_exist.jsonl", dictionary_terms=["a"])
        assert w2i == {"a": 0}

    def test_encode_sentences_empty_vocab_unk(self, tmp_path):
        corpus = tmp_path / "c.jsonl"
        corpus.write_text('{"text": "x", "genre_id": "g", "sentence_id": 0}\n')
        out = tmp_path / "out.jsonl"
        n = encode_sentences(corpus, {}, out, unk_id=0)
        assert n == 1

    def test_load_vocab_empty_json_object(self, tmp_path):
        (tmp_path / "v.json").write_text("{}")
        w2i, i2w = load_vocab(tmp_path / "v.json")
        assert w2i == {} and i2w == {}

    def test_load_vocab_non_dict_returns_empty(self, tmp_path):
        (tmp_path / "v.json").write_text("[]")
        w2i, i2w = load_vocab(tmp_path / "v.json")
        assert w2i == {} and i2w == {}


class TestBreakageCorpusGraph:
    def test_build_graph_empty_token_ids_skipped(self, tmp_path):
        (tmp_path / "e.jsonl").write_text(
            json.dumps({"sentence_id": 0, "token_ids": [], "text": ""}) + "\n"
        )
        g = build_graph(tmp_path / "e.jsonl")
        assert g["sentence_words"] == {}

    def test_corpus_graph_empty_dict(self):
        cg = CorpusGraph({})
        assert cg.sentence_ids() == []
        assert cg.sentences_containing_word(999) == []
        assert cg.similar_sentences(999) == []
        assert cg.next_word_counts(999) == {}

    def test_corpus_graph_partial_keys(self):
        data = {"sentence_words": {"0": [1, 2]}}
        cg = CorpusGraph(data)
        assert cg.sentence_ids() == [0]
        assert cg.cooccurring_words(1) == []

    def test_load_graph_truncated_json_returns_none(self, tmp_path):
        (tmp_path / "g.json").write_text('{"sentence_words":')
        assert load_graph(tmp_path / "g.json") is None


class TestBreakageCritic:
    def test_extract_content_terms_none(self):
        assert extract_content_terms(None) == []

    def test_critic_decision_negative_score(self):
        d, w = critic_decision(-0.1, accept_threshold=0.25, low_warn_threshold=0.15)
        assert d == "reject"
        assert w is True

    def test_critic_decision_score_exactly_at_threshold(self):
        d, _ = critic_decision(0.25, accept_threshold=0.25)
        assert d == "accept"

    def test_score_and_decide_none_engine(self):
        out = score_and_decide("hello world", None, None)
        assert out["num_grounded"] == 0
        assert "decision" in out


class TestBreakageRetrieval:
    def test_get_concept_bundle_engine_raises_returns_empty(self):
        bad_engine = type("Bad", (), {"get_context_for_description": lambda q: 1 / 0})()
        b = get_concept_bundle(bad_engine, "x")
        assert b["terms"] == [] and b["definitions"] == {}

    def test_get_style_sentences_from_graph_empty_concept_bundle(self, tmp_path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        (tmp_path / "corpus" / "vocab.json").write_text('{"word_to_id": {}, "id_to_word": {}}')
        (tmp_path / "corpus" / "graph.json").write_text(
            json.dumps({
                "sentence_words": {},
                "word_cooccurrence": {},
                "word_next": {},
                "sentence_similar": {},
            })
        )
        (tmp_path / "corpus" / "encoded_sentences.jsonl").write_text("")
        out = get_style_sentences_from_graph(tmp_path, {"terms": []}, "general")
        assert out == []
