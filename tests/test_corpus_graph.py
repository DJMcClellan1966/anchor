"""
Tests for anchor.corpus_graph: build_graph, save/load, CorpusGraph.
Errors, completeness, robustness, breakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_graph import (
    CorpusGraph,
    build_graph,
    build_transition_matrix,
    export_corpus_model,
    load_corpus_graph,
    load_graph,
    save_graph,
)


class TestBuildGraph:
    """Build graph: empty file, single sentence, malformed lines."""

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "encoded.jsonl"
        p.touch()
        g = build_graph(p)
        assert g["sentence_words"] == {}
        assert g["word_cooccurrence"] == {}
        assert g["sentence_similar"] == {}

    def test_single_sentence(self, tmp_path: Path):
        p = tmp_path / "encoded.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0,
                "genre_id": "general",
                "token_ids": [1, 2, 3],
                "text": "A B C",
            }) + "\n")
        g = build_graph(p)
        assert "0" in g["sentence_words"]
        assert g["sentence_words"]["0"] == [1, 2, 3]
        assert "1" in g["word_cooccurrence"]
        assert "1" in g["word_next"]

    def test_malformed_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "encoded.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps({
                "sentence_id": 0, "token_ids": [1, 2], "text": "x y",
            }) + "\n")
        g = build_graph(p)
        assert len(g["sentence_words"]) == 1

    def test_sentence_with_empty_token_ids_skipped(self, tmp_path: Path):
        p = tmp_path / "encoded.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "token_ids": [], "text": ""}) + "\n")
            f.write(json.dumps({"sentence_id": 1, "token_ids": [1], "text": "x"}) + "\n")
        g = build_graph(p)
        assert "0" not in g["sentence_words"]
        assert "1" in g["sentence_words"]

    def test_encoded_dictionary_merged_into_word_next(self, tmp_path: Path):
        enc = tmp_path / "encoded.jsonl"
        with open(enc, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "token_ids": [1, 2], "text": "x y"}) + "\n")
        dict_path = tmp_path / "encoded_dictionary.jsonl"
        dict_path.write_text(json.dumps({"term": "foo", "token_ids": [5, 6, 7]}) + "\n")
        g = build_graph(enc, encoded_dictionary_path=dict_path)
        cg = CorpusGraph(g)
        assert cg.next_word_counts(5) == {6: 1}
        assert cg.next_word_counts(6) == {7: 1}

    def test_no_encoded_dictionary_path_skips_merge(self, tmp_path: Path):
        enc = tmp_path / "encoded.jsonl"
        with open(enc, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_id": 0, "token_ids": [1, 2], "text": "x y"}) + "\n")
        dict_path = tmp_path / "encoded_dictionary.jsonl"
        dict_path.write_text(json.dumps({"term": "foo", "token_ids": [5, 6, 7]}) + "\n")
        g = build_graph(enc, encoded_dictionary_path=None)
        cg = CorpusGraph(g)
        assert cg.next_word_counts(5) == {}

    def test_context_to_sentences_index_built(self, tmp_path: Path):
        p = tmp_path / "encoded.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0,
                "token_ids": [10, 20, 30, 40, 50, 60],
                "text": "A B C D E F",
            }) + "\n")
        g = build_graph(p, context_length=5)
        assert "context_to_sentences" in g
        assert g["context_to_sentences"]["context_length"] == 5
        idx = g["context_to_sentences"]["index"]
        key = "10,20,30,40,50"
        assert key in idx
        assert [0, 5] in idx[key]


class TestSaveLoadGraph:
    """Round-trip and load missing/invalid."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        g = {
            "sentence_words": {"0": [1, 2]},
            "word_cooccurrence": {"1": [2], "2": [1]},
            "word_next": {"1": {"2": 1}},
            "sentence_similar": {"0": [[1, 0.5]]},
        }
        path = tmp_path / "graph.json"
        save_graph(g, path)
        assert path.exists()
        loaded = load_graph(path)
        assert loaded is not None
        assert loaded["sentence_words"] == g["sentence_words"]

    def test_load_missing_returns_none(self, tmp_path: Path):
        assert load_graph(tmp_path / "nonexistent.json") is None

    def test_load_invalid_json_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json {")
        assert load_graph(p) is None


class TestCorpusGraph:
    """CorpusGraph: empty data, minimal data, methods."""

    def test_empty_data(self):
        cg = CorpusGraph({})
        assert cg.sentence_ids() == []
        assert cg.sentences_containing_word(0) == []
        assert cg.similar_sentences(0) == []
        assert cg.sentence_token_ids(0) == []

    def test_minimal_data(self):
        data = {
            "sentence_words": {"0": [1, 2, 3], "1": [2, 3, 4]},
            "word_cooccurrence": {"1": [2, 3], "2": [1, 3, 4]},
            "word_next": {"1": {"2": 1}, "2": {"3": 1}},
            "sentence_similar": {"0": [[1, 0.5]], "1": [[0, 0.5]]},
        }
        cg = CorpusGraph(data)
        assert cg.sentence_ids() == [0, 1]
        assert cg.sentences_containing_word(2) == [0, 1]
        assert cg.sentence_token_ids(0) == [1, 2, 3]
        assert cg.similar_sentences(0, top_k=1) == [(1, 0.5)]
        assert cg.next_word_counts(1) == {2: 1}
        assert 3 in cg.cooccurring_words(1)

    def test_missing_keys_treated_as_empty(self):
        cg = CorpusGraph({"sentence_words": {"0": [1]}})
        assert cg.sentence_ids() == [0]
        assert cg.similar_sentences(0) == []
        assert cg.cooccurring_words(99) == []

    def test_has_context_index_false_when_empty(self):
        cg = CorpusGraph({"sentence_words": {"0": [1, 2]}})
        assert cg.has_context_index() is False

    def test_get_sentences_with_context(self):
        data = {
            "sentence_words": {"0": [1, 2, 3, 4, 5, 6], "1": [7, 8, 1, 2, 3, 9]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
            "context_to_sentences": {
                "context_length": 5,
                "index": {"1,2,3,4,5": [[0, 5]], "7,8,1,2,3": [[1, 5]]},
            },
        }
        cg = CorpusGraph(data)
        assert cg.has_context_index() is True
        pairs = cg.get_sentences_with_context([1, 2, 3, 4, 5])
        assert (0, 5) in pairs
        pairs2 = cg.get_sentences_with_context([7, 8, 1, 2, 3])
        assert (1, 5) in pairs2
        assert cg.get_sentences_with_context([1, 2]) == []

    def test_next_word_counts_in_sentence_returns_only_in_sentence_bigrams(self):
        """Per-sentence next-word counts: sentence [1,2,3,2,4] gives for w=2: {3: 1, 4: 1}."""
        data = {
            "sentence_words": {"0": [1, 2, 3, 2, 4], "1": [2, 5, 6]},
            "word_cooccurrence": {},
            "word_next": {},
            "sentence_similar": {},
        }
        cg = CorpusGraph(data)
        # Sentence 0: tokens [1,2,3,2,4]; after 2 we see 3 once and 4 once
        assert cg.next_word_counts_in_sentence(0, 2) == {3: 1, 4: 1}
        assert cg.next_word_counts_in_sentence(0, 1) == {2: 1}
        assert cg.next_word_counts_in_sentence(0, 4) == {}
        # Sentence 1: [2,5,6]; after 2 only 5
        assert cg.next_word_counts_in_sentence(1, 2) == {5: 1}
        # Unknown sentence
        assert cg.next_word_counts_in_sentence(99, 2) == {}


class TestLoadCorpusGraph:
    """load_corpus_graph: missing dir, missing file."""

    def test_missing_graph_returns_none(self, tmp_path: Path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        assert load_corpus_graph(tmp_path) is None

    def test_valid_graph_returns_corpus_graph(self, tmp_path: Path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump({
                "sentence_words": {"0": [1, 2]},
                "word_cooccurrence": {},
                "word_next": {},
                "sentence_similar": {"0": []},
            }, f)
        cg = load_corpus_graph(tmp_path)
        assert cg is not None
        assert cg.sentence_ids() == [0]


class TestBuildTransitionMatrix:
    def test_build_transition_matrix_sparse_log_probs(self):
        import math
        word_next = {1: {2: 3, 3: 1}, 2: {3: 2}}
        trans = build_transition_matrix(word_next, vocab_size=10, smoothing=0.01)
        assert 1 in trans
        assert 2 in trans
        row1 = trans[1]
        assert 2 in row1 and 3 in row1
        for lp in row1.values():
            assert lp <= 0
        assert math.exp(row1[2]) > math.exp(row1[3])


class TestExportCorpusModel:
    def test_export_corpus_model_writes_file(self, tmp_path: Path):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"word_to_id": {"a": 0, "b": 1, "c": 2}, "id_to_word": {"0": "a", "1": "b", "2": "c"}}, f)
        graph = {
            "sentence_words": {"0": [0, 1, 2]},
            "word_next": {"0": {"1": 2}, "1": {"2": 1}},
            "word_cooccurrence": {},
            "sentence_similar": {},
        }
        out = tmp_path / "corpus" / "corpus_model.json"
        export_corpus_model(graph, corpus / "vocab.json", out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["vocab_size"] == 3
        assert "transition" in data
        assert "0" in data["transition"] and "1" in data["transition"]
