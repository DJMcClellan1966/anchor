"""
Tests for anchor.generator: generate, _generate_stub, _generate_corpus fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.generator import generate


class TestGenerateStub:
    def test_empty_bundle_returns_message(self):
        out, _ = generate(
            "q", {"terms": [], "definitions": {}}, [], {},
            generator_kind="stub"
        )
        assert "No concepts" in out or "dictionary" in out.lower()

    def test_terms_only(self):
        out, _ = generate(
            "q", {"terms": ["a", "b"], "definitions": {}}, [], {},
            generator_kind="stub"
        )
        assert "a" in out and "b" in out

    def test_definitions_formatted(self):
        out, _ = generate(
            "q",
            {"terms": [], "definitions": {"x": "Definition of x."}},
            [], {"include_definitions_in_response": True},
            generator_kind="stub"
        )
        assert "x" in out and "Definition" in out

    def test_list_definition_first_element_used(self):
        out, _ = generate(
            "q",
            {"terms": [], "definitions": {"y": ["first", "second"]}},
            [], {"include_definitions_in_response": True},
            generator_kind="stub"
        )
        assert "first" in out

    def test_stub_answer_style_when_definitions_disabled(self):
        """When include_definitions_in_response is false, no term: definition lines; may include style_sentences."""
        config = {"include_definitions_in_response": False}
        out, _ = generate(
            "q", {"terms": ["alpha", "beta"], "definitions": {"alpha": "Def A."}},
            ["First style sentence.", "Second."], config, generator_kind="stub"
        )
        assert "alpha" in out or "beta" in out
        assert "Def A." not in out
        assert "Related:" in out or "First style" in out

    def test_stub_definition_style_when_definitions_enabled(self):
        """When include_definitions_in_response is true, definition lines present."""
        config = {"include_definitions_in_response": True}
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {"x": "Definition of x."}},
            [], config, generator_kind="stub"
        )
        assert "x" in out and "Definition" in out


class TestGenerateCorpusFallback:
    def test_no_data_dir_falls_back_to_stub(self):
        out, _ = generate(
            "q", {"terms": ["t"], "definitions": {}}, [], {"align_data_dir": None},
            generator_kind="corpus"
        )
        assert "t" in out or "Concepts" in out

    def test_missing_graph_falls_back_to_stub(self, tmp_path):
        (tmp_path / "corpus").mkdir(exist_ok=True)
        out, _ = generate(
            "q", {"terms": ["t"], "definitions": {}}, [], {"align_data_dir": str(tmp_path)},
            generator_kind="corpus"
        )
        assert "t" in out or "Concepts" in out

    def test_unknown_generator_kind_uses_stub(self):
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {}}, [], {},
            generator_kind="unknown_kind"
        )
        assert "x" in out

    def test_corpus_hybrid_with_context_index_returns_text(self, tmp_path: Path):
        """When graph has context_to_sentences, generator uses hybrid next-token and returns text."""
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        word_to_id = {"Hello": 0, "world": 1, ".": 2}
        id_to_word = {0: "Hello", 1: "world", 2: "."}
        with open(corpus / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({"word_to_id": word_to_id, "id_to_word": {str(k): v for k, v in id_to_word.items()}}, f)
        with open(corpus / "encoded_sentences.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sentence_id": 0, "genre_id": "general", "token_ids": [0, 1, 2], "text": "Hello world ."
            }) + "\n")
        graph = {
            "sentence_words": {"0": [0, 1, 2]},
            "word_cooccurrence": {"0": [1, 2], "1": [0, 2], "2": [0, 1]},
            "word_next": {"0": {"1": 1}, "1": {"2": 1}},
            "sentence_similar": {"0": []},
            "context_to_sentences": {"context_length": 2, "index": {"0,1": [[0, 2]]}},
        }
        with open(corpus / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f)
        config = {
            "align_data_dir": str(tmp_path),
            "default_genre_id": "general",
            "corpus_hybrid_context_length": 2,
            "corpus_hybrid_beta": 0.7,
            "corpus_max_tokens": 20,
        }
        out, _ = generate(
            "q", {"terms": ["Hello"], "definitions": {}},
            ["Hello world"], config, generator_kind="corpus"
        )
        assert out
        assert "Hello" in out

    def test_graph_attention_missing_data_falls_back_to_stub(self):
        config = {"align_data_dir": None}
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {"x": "Def of x."}},
            [], config, generator_kind="graph_attention"
        )
        assert "x" in out or "Def" in out

    def test_scratchllm_missing_path_falls_back_to_stub(self):
        config = {"scratchllm_path": None}
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {"x": "Def of x."}},
            [], config, generator_kind="scratchllm"
        )
        assert "x" in out or "Def" in out

    def test_scratchllm_nonexistent_path_falls_back_to_stub(self, tmp_path: Path):
        config = {"scratchllm_path": str(tmp_path / "nonexistent_scratchllm")}
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {"x": "Def of x."}},
            [], config, generator_kind="scratchllm"
        )
        assert "x" in out or "Def" in out

    def test_scratchllm_import_failure_falls_back_to_stub(self, tmp_path: Path):
        """When scratchllm_path exists but base.retrieve is not there, fall back to stub."""
        (tmp_path / "base").mkdir(parents=True, exist_ok=True)
        config = {"scratchllm_path": str(tmp_path)}
        out, _ = generate(
            "q", {"terms": ["x"], "definitions": {"x": "Def of x."}},
            [], config, generator_kind="scratchllm"
        )
        assert "x" in out or "Def" in out
