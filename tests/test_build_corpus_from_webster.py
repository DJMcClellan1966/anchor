"""
Tests for scripts/build_corpus_from_webster: ingest Webster JSON into sentences.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import the script's run() by path (script is not a package)
import sys
_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from build_corpus_from_webster import run as build_webster_run


def test_run_writes_jsonl_with_genre(tmp_path: Path):
    webster = tmp_path / "dict.json"
    webster.write_text(json.dumps({"apple": "A fruit.", "banana": "Another fruit."}))
    n = build_webster_run(webster, tmp_path, genre_id="definitional", append=False)
    assert n == 2
    out = tmp_path / "corpus" / "sentences.jsonl"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj.get("genre_id") == "definitional"
        assert "text" in obj
        assert ": " in obj["text"]

def test_run_append_adds_to_existing(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    existing = corpus_dir / "sentences.jsonl"
    existing.write_text(json.dumps({"text": "Prior sentence.", "genre_id": "general"}) + "\n")
    webster = tmp_path / "dict.json"
    webster.write_text(json.dumps({"x": "X def."}))
    n = build_webster_run(webster, tmp_path, genre_id="definitional", append=True)
    assert n == 1
    lines = existing.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first.get("text") == "Prior sentence."
    second = json.loads(lines[1])
    assert "x" in second["text"] and "definitional" == second["genre_id"]

def test_run_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_webster_run(tmp_path / "missing.json", tmp_path)
