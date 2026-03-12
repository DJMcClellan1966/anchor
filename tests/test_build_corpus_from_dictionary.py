"""
Tests for scripts/build_corpus_from_dictionary: ingest dictionary repo compiled corpora into sentences.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from build_corpus_from_dictionary import run as build_dict_run


def test_run_writes_jsonl_from_compiled_corpus(tmp_path: Path):
    """GooAQ-style compiled_corpus.json -> sentences with genre_id gooaq."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    compiled = data_dir / "compiled_corpus.json"
    compiled.write_text(json.dumps({
        "neural_network": {
            "noun": {"def": "A type of computing system.", "definitions": ["Inspired by the brain."]},
            "source": "gooaq",
            "questions": ["What is a neural network?"],
        },
    }, ensure_ascii=False))
    n = build_dict_run(data_dir, tmp_path, append=False)
    assert n >= 2
    out = tmp_path / "corpus" / "sentences.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    genres = {obj.get("genre_id") for obj in lines}
    assert "gooaq" in genres
    assert any("neural" in (obj.get("text") or "") for obj in lines)


def test_run_append_adds_to_existing(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    (tmp_path / "corpus" / "sentences.jsonl").write_text(
        json.dumps({"text": "Existing.", "genre_id": "general"}) + "\n"
    )
    data_dir = tmp_path / "dict_data"
    data_dir.mkdir()
    (data_dir / "compiled_conceptnet.json").write_text(json.dumps({
        "cat": {"noun": {"def": "A feline animal.", "definitions": []}, "source": "conceptnet"},
    }, ensure_ascii=False))
    n = build_dict_run(data_dir, tmp_path, append=True)
    assert n >= 1
    lines = (tmp_path / "corpus" / "sentences.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 2
    assert "Existing." in lines[0]
    assert "conceptnet" in (json.loads(lines[-1]).get("genre_id") or "")


def test_run_missing_data_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_dict_run(tmp_path / "nonexistent", tmp_path)
