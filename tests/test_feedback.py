"""Tests for anchor.feedback: record, load_weights, apply_boosts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor import feedback


class TestRecord:
    def test_record_appends_line(self, tmp_path: Path):
        path = tmp_path / "feedback.jsonl"
        feedback.record("s1", "query one", "response one", True, path=path)
        feedback.record("s1", "query two", "response two", False, path=path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        assert r1["query"] == "query one" and r1["accepted"] is True
        r2 = json.loads(lines[1])
        assert r2["query"] == "query two" and r2["accepted"] is False


class TestLoadWeights:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        w = feedback.load_weights(path=tmp_path / "nonexistent.json")
        assert w == {}

    def test_valid_file_returns_dict(self, tmp_path: Path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"what is x": [0, 1], "other": [2]}), encoding="utf-8")
        w = feedback.load_weights(path=path)
        assert "what is x" in w
        assert w["what is x"] == [0, 1]
        assert w["other"] == [2]


class TestApplyBoosts:
    def test_boosts_matching_query(self):
        sentence_visits = {0: 1.0, 1: 2.0}
        weights = {"what is it": [0, 1]}
        feedback.apply_boosts(sentence_visits, weights, "what is it", boost=0.5)
        assert sentence_visits[0] == 1.5
        assert sentence_visits[1] == 2.5

    def test_no_boost_when_key_missing(self):
        sentence_visits = {0: 1.0}
        weights = {"other query": [0]}
        feedback.apply_boosts(sentence_visits, weights, "what is it", boost=0.5)
        assert sentence_visits[0] == 1.0
