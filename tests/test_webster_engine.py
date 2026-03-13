"""
Tests for anchor.webster_engine: WebsterEngine and get_context_for_description.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.webster_engine import WebsterEngine


class TestWebsterEngine:
    def test_missing_file_returns_empty_on_lookup(self, tmp_path: Path):
        missing = tmp_path / "missing.json"
        eng = WebsterEngine(missing)
        out = eng.get_context_for_description("anything")
        assert out == {}

    def test_empty_json_returns_empty_on_lookup(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text("{}")
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("word")
        assert out == {}

    def test_known_word_returns_definition_map_and_key_words(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"anopheles": "A genus of mosquitoes."}))
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("anopheles")
        assert "definition_map" in out
        assert out["definition_map"]["anopheles"] == "A genus of mosquitoes."
        assert "key_words" in out
        assert any(kw.get("name") == "anopheles" for kw in out["key_words"])
        assert out.get("definitions", {}).get("anopheles") == "A genus of mosquitoes."

    def test_query_with_multiple_words_looks_up_each(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"foo": "Foo definition.", "bar": "Bar definition."}))
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("foo and bar")
        assert "foo" in out["definition_map"]
        assert "bar" in out["definition_map"]
        assert len(out["key_words"]) == 2

    def test_unknown_word_returns_empty(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"other": "Other def."}))
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("nonexistent")
        assert out == {}

    def test_empty_query_returns_empty(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"a": "A."}))
        eng = WebsterEngine(p)
        assert eng.get_context_for_description("") == {}
        assert eng.get_context_for_description("   ") == {}

    def test_case_insensitive_lookup(self, tmp_path: Path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"Anopheles": "A genus of mosquitoes."}))
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("ANOPHELES")
        assert "definition_map" in out
        assert "anopheles" in out["definition_map"] or "ANOPHELES" in out["definition_map"]

    def test_list_definition_multiple_senses(self, tmp_path: Path):
        """Webster JSON value may be a list of strings (multiple senses); pass-through in bundle."""
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({
            "function": ["A relation from inputs to outputs.", "A role or purpose."],
        }))
        eng = WebsterEngine(p)
        out = eng.get_context_for_description("function")
        assert "definition_map" in out
        val = out["definition_map"].get("function")
        assert isinstance(val, list)
        assert len(val) == 2
        assert "inputs to outputs" in val[0]
        assert "role or purpose" in val[1]
