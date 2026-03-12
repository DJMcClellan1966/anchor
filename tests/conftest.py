"""
Pytest configuration and shared fixtures for Anchor tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def anchor_root() -> Path:
    """Project root (parent of tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_corpus_dir(tmp_path: Path) -> Path:
    """Temporary dir with corpus/sentences.jsonl and optional vocab/graph."""
    return tmp_path


@pytest.fixture
def sample_sentences_jsonl(temp_corpus_dir: Path) -> Path:
    """Create corpus/sentences.jsonl with a few lines; return path to corpus dir (parent of file)."""
    corpus = temp_corpus_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    path = corpus / "sentences.jsonl"
    lines = [
        {"text": "Hello world.", "genre_id": "general", "sentence_id": 0},
        {"text": "Retirement is a goal.", "genre_id": "retirement", "sentence_id": 1},
        {"text": "This is a test.", "genre_id": "general", "sentence_id": 2},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return temp_corpus_dir


@pytest.fixture
def empty_engine():
    """Mock engine with no get_context_for_description (returns empty bundle)."""
    return None


@pytest.fixture
def mock_engine_with_context():
    """Mock engine that returns a concept bundle for get_context_for_description."""

    def get_context_for_description(query: str):
        if not query or not query.strip():
            return {}
        return {
            "concepts": [{"name": "function"}, {"name": "code"}],
            "definitions": {"function": "A reusable block of code."},
            "definition_map": {"code": "Instructions for a computer."},
        }

    class MockEngine:
        pass

    MockEngine.get_context_for_description = staticmethod(get_context_for_description)
    return MockEngine()
