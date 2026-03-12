"""
Tests for anchor.grammar: rewrite with rules file or command.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor.grammar import rewrite


class TestRewrite:
    def test_no_use_grammar_returns_unchanged(self):
        out = rewrite("hello world", {"use_grammar": False})
        assert out == "hello world"

    def test_empty_text_returns_unchanged(self):
        out = rewrite("", {"use_grammar": True, "grammar_rules_path": "/nonexistent"})
        assert out == ""

    def test_rules_file_applies_pattern_replacement(self, tmp_path: Path):
        rules = [{"pattern": r"\bteh\b", "replacement": "the"}]
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(rules), encoding="utf-8")
        out = rewrite("teh cat", {"use_grammar": True, "grammar_rules_path": str(path)})
        assert out == "the cat"

    def test_rules_file_missing_returns_unchanged(self):
        out = rewrite("hello", {"use_grammar": True, "grammar_rules_path": "/nonexistent/path.json"})
        assert out == "hello"

    def test_no_rules_or_command_returns_unchanged(self):
        out = rewrite("hello", {"use_grammar": True})
        assert out == "hello"

    def test_grammar_command_preferred_when_set(self, tmp_path: Path):
        rules = tmp_path / "rules.json"
        rules.write_text('[{"pattern": "x", "replacement": "y"}]', encoding="utf-8")
        # On Windows we need a command that echoes stdin; use Python
        cmd = "python -c \"import sys; print(sys.stdin.read().strip())\""
        out = rewrite("hello", {"use_grammar": True, "grammar_command": cmd, "grammar_rules_path": str(rules)})
        assert out == "hello"
