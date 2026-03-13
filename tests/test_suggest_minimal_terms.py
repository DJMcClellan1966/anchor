"""
Tests for scripts/suggest_minimal_terms: minimal term set for a set of queries (Theorem 5).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from suggest_minimal_terms import run as suggest_minimal_terms_run


def test_run_collects_terms_from_queries(tmp_path: Path):
    """Given a Webster JSON and queries, returns sorted unique terms from concept bundles."""
    webster = tmp_path / "dict.json"
    webster.write_text(json.dumps({
        "apple": "A fruit.",
        "function": "A relation.",
        "fruit": "Edible plant product.",
    }))
    queries = ["What is an apple?", "Explain function and fruit."]
    terms = suggest_minimal_terms_run(queries, webster_path=webster)
    assert "apple" in terms
    assert "function" in terms
    assert "fruit" in terms
    assert terms == sorted(terms)


def test_run_writes_pruned_json_when_requested(tmp_path: Path):
    """When pruned_json_path is set, writes a Webster JSON with only the suggested terms."""
    webster = tmp_path / "dict.json"
    webster.write_text(json.dumps({
        "apple": "A fruit.",
        "banana": "Another fruit.",
        "function": "A relation.",
    }))
    queries = ["apple"]
    pruned_path = tmp_path / "pruned.json"
    terms = suggest_minimal_terms_run(
        queries, webster_path=webster, pruned_json_path=pruned_path
    )
    assert "apple" in terms
    assert pruned_path.exists()
    with open(pruned_path, encoding="utf-8") as f:
        pruned = json.load(f)
    assert "apple" in pruned or "Apple" in list(pruned.keys())
    assert len(pruned) == 1
