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

    def test_grammar_rewrite_applied_when_configured(self, mock_engine_with_context, tmp_path):
        import json
        rules = [{"pattern": r"\bteh\b", "replacement": "the"}]
        (tmp_path / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
        config = {
            "align_data_dir": None,
            "default_genre_id": "retirement",
            "use_grammar": True,
            "grammar_rules_path": str(tmp_path / "rules.json"),
        }
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        # Stub returns concepts + style; we need response to contain "teh" then get corrected.
        # Mock returns terms/code/function - stub with include_definitions false gives "Related: ..." or style.
        # So we need a response that has "teh" - the stub won't. So patch or use a custom that returns "teh word".
        # Simpler: just assert that when we have a rule and use_grammar, the engine runs and returns something.
        # To test correction we'd need to force the generator to return "teh" - e.g. mock generator to return "teh cat".
        from unittest.mock import patch
        with patch("anchor.engine.generator.generate", return_value=("teh cat", {"generator_actually_used": "stub"})):
            response, _, _ = engine.query("q")
        assert "the" in response
        assert "teh" not in response


class TestGoldenQueries:
    """Smoke tests: fixed queries return grounded output (concept_bundle or graph)."""

    def test_grounded_response_with_concept_bundle(self, mock_engine_with_context):
        """Query with mock dictionary returns non-empty response and uses concept terms."""
        config = {"align_data_dir": None, "default_genre_id": "retirement"}
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info, extras = engine.query("What is a function?", return_extras=True)
        assert isinstance(extras, dict)
        assert extras.get("generator_meta", {}).get("generator_actually_used") == "stub"
        assert response
        assert response.strip()
        concept_bundle = extras.get("concept_bundle") or {}
        terms = concept_bundle.get("terms") or []
        assert terms, "concept_bundle should have terms for this query"
        assert any(t in response for t in terms) or "function" in response.lower() or "code" in response.lower()

    def test_graph_attention_actually_used_when_data_present(self, mock_engine_with_context, tmp_path):
        """With mock graph data, generator_actually_used is graph_attention and response uses corpus."""
        import json
        corpus = tmp_path / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "vocab.json").write_text(
            json.dumps({"word_to_id": {"hello": 0, "world": 1}, "id_to_word": {"0": "hello", "1": "world"}}),
            encoding="utf-8",
        )
        (corpus / "encoded_sentences.jsonl").write_text(
            json.dumps({"sentence_id": 0, "genre_id": "general", "text": "Hello world.", "token_ids": [0, 1]}) + "\n",
            encoding="utf-8",
        )
        graph = {
            "sentence_words": {"0": [0, 1]},
            "word_cooccurrence": {"0": [1], "1": [0]},
            "word_next": {"0": {"1": 1}},
            "sentence_similar": {"0": []},
        }
        (corpus / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
        config = {
            "align_data_dir": str(tmp_path),
            "default_genre_id": "general",
        }
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="graph_attention")
        response, critic_info, extras = engine.query("hello", return_extras=True)
        assert isinstance(extras, dict)
        gen_meta = extras.get("generator_meta") or {}
        assert gen_meta.get("generator_actually_used") == "graph_attention"
        assert response and response.strip()
        assert "hello" in response.lower() or "Hello" in response or "world" in response.lower()
