"""
Tests for scripts/encode_dictionary: encode dictionary definitions as token ID sequences.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from encode_dictionary import run as encode_dict_run, load_vocab, definitions_from_webster, definitions_from_file


def test_load_vocab(tmp_path: Path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"word_to_id": {"a": 0, "b": 1, "is": 2, "one": 3}}))
    w2i = load_vocab(vocab_path)
    assert w2i["a"] == 0
    assert w2i["b"] == 1


def test_definitions_from_webster(tmp_path: Path):
    webster = tmp_path / "d.json"
    webster.write_text(json.dumps({"cat": "A feline animal.", "dog": "A canine."}))
    pairs = definitions_from_webster(webster)
    assert len(pairs) == 2
    assert ("cat", "A feline animal.") in pairs
    assert ("dog", "A canine.") in pairs


def test_definitions_from_file(tmp_path: Path):
    f = tmp_path / "defs.txt"
    f.write_text("foo\tA placeholder.\nbar\tAnother.\n")
    pairs = definitions_from_file(f)
    assert len(pairs) == 2
    assert pairs[0] == ("foo", "A placeholder.")
    assert pairs[1] == ("bar", "Another.")


def test_run_writes_jsonl_with_token_ids(tmp_path: Path):
    # Tokenizer produces "A", " ", "feline", " ", "animal", "." for "A feline animal."
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({
        "word_to_id": {"A": 0, " ": 1, "feline": 2, "animal": 3, ".": 4, "canine": 5},
    }))
    webster = tmp_path / "d.json"
    webster.write_text(json.dumps({"cat": "A feline animal.", "dog": "A canine."}))
    out = tmp_path / "encoded_dictionary.jsonl"
    n = encode_dict_run(vocab_path, out, webster_path=webster)
    assert n >= 1
    assert out.exists()
    lines = [l for l in out.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    for line in lines:
        obj = json.loads(line)
        assert "term" in obj and "token_ids" in obj
        assert isinstance(obj["token_ids"], list)
        for tid in obj["token_ids"]:
            assert isinstance(tid, (int, float))


def test_run_missing_vocab_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        encode_dict_run(tmp_path / "missing.json", tmp_path / "out.jsonl", webster_path=tmp_path / "w.json")
