"""
Tests for anchor.anchor_math: doc-aligned API (V, tau, S_of_w, Embed_anchor, etc.).
"""
from __future__ import annotations

import pytest

from anchor.anchor_math import AnchorMath
from anchor.corpus_graph import CorpusGraph, build_graph
from anchor.corpus_vocab import load_vocab
from pathlib import Path
import json


def test_anchor_math_S_of_w_matches_graph_sentences_containing_word(tmp_path: Path):
    """S_of_w(w) matches graph.sentences_containing_word(w)."""
    enc = tmp_path / "encoded.jsonl"
    with open(enc, "w", encoding="utf-8") as f:
        f.write(json.dumps({"sentence_id": 0, "token_ids": [1, 2, 3], "text": "A B C"}) + "\n")
        f.write(json.dumps({"sentence_id": 1, "token_ids": [2, 4], "text": "B D"}) + "\n")
    g_dict = build_graph(enc)
    graph = CorpusGraph(g_dict)
    word_to_id = {"a": 1, "b": 2, "c": 3, "d": 4}
    id_to_word = {1: "a", 2: "b", 3: "c", 4: "d"}
    am = AnchorMath(graph, word_to_id, id_to_word)
    for w in [1, 2, 3, 4]:
        assert set(am.S_of_w(w)) == set(graph.sentences_containing_word(w))
    assert set(am.S_of_w(2)) == {0, 1}
    assert am.S_of_w(1) == [0]


def test_anchor_math_V_tau_t_s_W_of_s():
    """V, tau, tau_inv, S, t_s, W_of_s delegate correctly."""
    data = {
        "sentence_words": {"0": [1, 2, 3], "1": [2, 4]},
        "word_cooccurrence": {},
        "word_next": {},
        "sentence_similar": {},
    }
    graph = CorpusGraph(data)
    word_to_id = {"x": 1, "y": 2, "z": 3, "w": 4}
    id_to_word = {1: "x", 2: "y", 3: "z", 4: "w"}
    am = AnchorMath(graph, word_to_id, id_to_word)
    assert am.V == {1, 2, 3, 4}
    assert am.tau("y") == 2
    assert am.tau("missing") is None
    assert am.tau_inv(2) == "y"
    assert am.tau_inv(99) is None
    assert set(am.S()) == {0, 1}
    assert am.t_s(0) == [1, 2, 3]
    assert am.W_of_s(0) == {1, 2, 3}
