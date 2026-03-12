"""
Tests for anchor.engine: AnchorEngine.query with edge cases.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anchor.engine import AnchorEngine


class TestAnchorEngineQuery:
    def test_returns_tuple_response_and_critic_info(self, mock_engine_with_context):
        config = {"align_data_dir": None, "default_genre_id": "retirement"}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info = engine.query("What is a function?")
        assert isinstance(response, str)
        assert isinstance(critic_info, dict)
        assert "score" in critic_info
        assert "decision" in critic_info

    def test_empty_question_still_returns(self, mock_engine_with_context):
        config = {}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info = engine.query("")
        assert isinstance(response, str)
        assert isinstance(critic_info, dict)

    def test_none_engine_empty_bundle_still_runs(self):
        config = {}
        engine = AnchorEngine(None, config, generator_kind="stub")
        response, critic_info = engine.query("anything")
        assert "No concepts" in response or "dictionary" in response.lower()
        assert critic_info["decision"] in ("accept", "warn", "reject")
