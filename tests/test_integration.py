"""
Integration tests: full pipeline with temp data; optional run against real engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure anchor is importable
_ANCHOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANCHOR_ROOT))

from anchor.corpus_vocab import run_build
from anchor.corpus_graph import build_graph, save_graph, load_corpus_graph
from anchor.retrieval import get_style_sentences, get_style_sentences_from_graph
from anchor.engine import AnchorEngine
from anchor.wire import get_config, get_engine, get_generator_kind


class TestFullCorpusPipeline:
    """Build corpus -> vocab -> graph; then run retrieval and engine."""

    def test_build_corpus_vocab_graph_then_retrieve(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        with open(corpus_dir / "sentences.jsonl", "w", encoding="utf-8") as f:
            for i, (text, g) in enumerate([
                ("Hello world.", "general"),
                ("Retirement is a goal.", "retirement"),
                ("This is a test.", "general"),
            ]):
                f.write(json.dumps({"text": text, "genre_id": g, "sentence_id": i}) + "\n")

        vocab_size, n_enc = run_build(tmp_path)
        assert vocab_size > 0 and n_enc == 3
        assert (tmp_path / "corpus" / "vocab.json").exists()
        assert (tmp_path / "corpus" / "encoded_sentences.jsonl").exists()

        g = build_graph(tmp_path / "corpus" / "encoded_sentences.jsonl", top_similar_per_sentence=5)
        save_graph(g, tmp_path / "corpus" / "graph.json")
        cg = load_corpus_graph(tmp_path)
        assert cg is not None
        assert len(cg.sentence_ids()) == 3

        sentences = get_style_sentences(None, tmp_path, {"terms": ["Retirement"]}, genre_id="retirement")
        assert any("Retirement" in s for s in sentences)

        graph_sentences = get_style_sentences_from_graph(tmp_path, {"terms": ["Retirement"]}, "retirement")
        assert isinstance(graph_sentences, list)

    def test_engine_query_with_mock_returns_valid_structure(self, mock_engine_with_context):
        config = {
            "align_data_dir": None,
            "default_genre_id": "retirement",
            "use_corpus_graph": False,
        }
        engine = AnchorEngine(mock_engine_with_context, config, generator_kind="stub")
        response, critic_info = engine.query("What is a function?")
        assert isinstance(response, str)
        assert len(response) > 0
        assert "function" in response or "code" in response
        assert critic_info["decision"] in ("accept", "warn", "reject")
        assert "score" in critic_info
        assert isinstance(critic_info["score"], (int, float))


class TestAgainstRealEngine:
    """If dictionary is available, run a real query and assert response shape and grounding.
    Run with: pytest -m real_engine
    """

    @pytest.mark.real_engine
    def test_real_engine_returns_non_empty_response(self):
        engine = get_engine()
        if engine is None:
            pytest.skip("Dictionary not configured; set dictionary_path to run")
        config = get_config()
        kind = get_generator_kind()
        anchor_engine = AnchorEngine(engine, config, generator_kind=kind)
        response, critic_info = anchor_engine.query("What is a function?")
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        assert "score" in critic_info
        assert critic_info["decision"] in ("accept", "warn", "reject")
        assert "num_grounded" in critic_info
        assert "num_content" in critic_info
