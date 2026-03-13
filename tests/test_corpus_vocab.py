"""
Tests for anchor.corpus_vocab: tokenize, build_vocab, encode_sentences, load_vocab, run_build.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.corpus_vocab import (
    build_vocab,
    encode_sentences,
    load_vocab,
    run_build,
    tokenize,
)


class TestTokenize:
    def test_normal_sentence(self):
        # Tokenizer includes space as token; words and punctuation
        out = tokenize("The cat sat.")
        assert "The" in out and "cat" in out and "sat" in out and "." in out
        assert len(out) >= 4

    def test_empty_string(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_none_equivalent(self):
        assert tokenize(None or "") == []

    def test_punctuation_only(self):
        assert tokenize("...") == ["..."]

    def test_words_and_punctuation(self):
        out = tokenize("Hello, world!")
        assert "Hello" in out and "world" in out
        assert any("," in t or "!" in t for t in out)

    def test_numbers_in_words(self):
        assert "abc123" in tokenize("Word abc123 done.")


class TestBuildVocab:
    def test_missing_corpus_file(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        w2i, i2w = build_vocab(missing)
        assert w2i == {}
        assert i2w == {}

    def test_missing_file_with_dictionary_terms(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        w2i, i2w = build_vocab(missing, dictionary_terms=["foo", "bar"])
        assert w2i["foo"] == 0
        assert w2i["bar"] == 1

    def test_empty_corpus_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.touch()
        w2i, i2w = build_vocab(p)
        assert w2i == {}

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write('{"text": "Good sentence."}\n')
        w2i, _ = build_vocab(p)
        assert "Good" in w2i


class TestEncodeSentences:
    def test_empty_vocab_unk(self, tmp_path):
        corpus = tmp_path / "sents.jsonl"
        out = tmp_path / "encoded.jsonl"
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"text": "Hello.", "genre_id": "general", "sentence_id": 0}\n')
        w2i = {}
        n = encode_sentences(corpus, w2i, out, unk_id=999)
        assert n == 1
        data = json.loads(next(open(out, encoding="utf-8")))
        assert 999 in data["token_ids"]

    def test_malformed_lines_skipped(self, tmp_path):
        corpus = tmp_path / "sents.jsonl"
        out = tmp_path / "encoded.jsonl"
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("bad\n")
            f.write('{"text": "Good.", "genre_id": "g", "sentence_id": 0}\n')
        w2i = {"Good": 0, ".": 1}
        n = encode_sentences(corpus, w2i, out)
        assert n == 1

    def test_preserves_source_and_term_from_corpus(self, tmp_path):
        """Unified store: encode_sentences passes through source and term when present."""
        corpus = tmp_path / "sents.jsonl"
        out = tmp_path / "encoded.jsonl"
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"text": "apple: A fruit.", "genre_id": "definitional", "source": "dictionary", "term": "apple"}\n')
        w2i = {"apple": 0, ":": 1, "A": 2, "fruit": 3, ".": 4}
        n = encode_sentences(corpus, w2i, out)
        assert n == 1
        data = json.loads(next(open(out, encoding="utf-8")))
        assert data.get("source") == "dictionary"
        assert data.get("term") == "apple"


class TestLoadVocab:
    def test_missing_file(self, tmp_path):
        w2i, i2w = load_vocab(tmp_path / "missing.json")
        assert w2i == {}
        assert i2w == {}

    def test_word_to_id_format(self, tmp_path):
        p = tmp_path / "vocab.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"word_to_id": {"a": 0, "b": 1}}, f)
        w2i, i2w = load_vocab(p)
        assert w2i["a"] == 0
        assert i2w[0] == "a"


class TestRunBuild:
    def test_run_build_creates_vocab_and_encoded(self, sample_sentences_jsonl):
        data_dir = sample_sentences_jsonl
        vocab_size, n_encoded = run_build(data_dir)
        assert vocab_size > 0
        assert n_encoded == 3
        assert (data_dir / "corpus" / "vocab.json").exists()
        assert (data_dir / "corpus" / "encoded_sentences.jsonl").exists()

    def test_run_build_with_dictionary_terms(self, sample_sentences_jsonl):
        vocab_size, _ = run_build(
            sample_sentences_jsonl, dictionary_terms=["Retirement", "goal"]
        )
        assert vocab_size >= 2
        w2i, _ = load_vocab(sample_sentences_jsonl / "corpus" / "vocab.json")
        assert "Retirement" in w2i
