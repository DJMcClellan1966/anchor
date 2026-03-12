"""
Tests for anchor.engine: AnchorEngine.query with edge cases, streaming, effective_query.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anchor.engine import AnchorEngine, _chunk_response


class TestChunkResponse:
    def test_empty_returns_empty(self):
        assert _chunk_response("") == []
        assert _chunk_response("   ") == []

    def test_splits_on_sentence_boundaries(self):
        chunks = _chunk_response("First. Second. Third.")
        assert len(chunks) >= 1
        assert "First" in chunks[0]

    def test_long_segment_split_by_max_chunk_size(self):
        long = "a" * 250
        chunks = _chunk_response(long, max_chunk_size=100)
        assert len(chunks) >= 2
        assert sum(len(c) for c in chunks) >= 250


class TestAnchorEngineQuery:
    def test_returns_tuple_response_and_critic_info(self, mock_engine_with_context):
        config = {"align_data_dir": None, "default_genre_id": "retirement"}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info, _ = engine.query("What is a function?")
        assert isinstance(response, str)
        assert isinstance(critic_info, dict)
        assert "score" in critic_info
        assert "decision" in critic_info

    def test_empty_question_still_returns(self, mock_engine_with_context):
        config = {}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info, _ = engine.query("")
        assert isinstance(response, str)
        assert isinstance(critic_info, dict)

    def test_none_engine_empty_bundle_still_runs(self):
        config = {}
        engine = AnchorEngine(None, config, generator_kind="stub")
        response, critic_info, _ = engine.query("anything")
        assert "No concepts" in response or "dictionary" in response.lower()
        assert critic_info["decision"] in ("accept", "warn", "reject")

    def test_stream_true_returns_iterator_chunks_then_critic(self, mock_engine_with_context):
        config = {"align_data_dir": None, "default_genre_id": "retirement"}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        result = engine.query("What is X?", stream=True)
        chunks = list(result)
        assert len(chunks) >= 2
        for i, (c, crit, ext) in enumerate(chunks):
            if i < len(chunks) - 1:
                assert c is not None and isinstance(c, str)
                assert crit is None and ext is None
            else:
                assert c is None
                assert crit is not None and isinstance(crit, dict)
                assert "score" in crit and "decision" in crit

    def test_effective_query_includes_system_prompt_and_history(self):
        config = {"system_prompt": "You are helpful.", "conversation_turn_limit": 2}
        engine = AnchorEngine(None, config, generator_kind="stub")
        q = engine._build_effective_query("What is Y?", [("Q1", "A1")])
        assert "You are helpful" in q
        assert "Q1" in q and "A1" in q
        assert "What is Y" in q
