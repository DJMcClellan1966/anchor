"""
Tests for anchor.wire: get_generator_kind with use_scratchllm priority.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from anchor import wire


class TestGetGeneratorKind:
    def test_use_scratchllm_prefers_scratchllm_over_graph(self, tmp_path: Path):
        """When use_scratchllm is true and scratchllm_path exists, return scratchllm even if graph exists."""
        (tmp_path / "corpus").mkdir(exist_ok=True)
        (tmp_path / "corpus" / "graph.json").write_text("{}")
        scratch_dir = tmp_path / "scratchllm"
        scratch_dir.mkdir(exist_ok=True)

        with patch.object(wire, "get_config", return_value={
            "use_scratchllm": True,
            "scratchllm_path": str(scratch_dir),
            "use_corpus_graph": True,
            "align_data_dir": str(tmp_path),
            "use_attention_loop": True,
        }):
            kind = wire.get_generator_kind()
        assert kind == "scratchllm"

    def test_without_use_scratchllm_graph_wins(self, tmp_path: Path):
        """When use_scratchllm is false, graph is preferred when present."""
        (tmp_path / "corpus").mkdir(exist_ok=True)
        (tmp_path / "corpus" / "graph.json").write_text("{}")
        scratch_dir = tmp_path / "scratchllm"
        scratch_dir.mkdir(exist_ok=True)

        with patch.object(wire, "get_config", return_value={
            "use_scratchllm": False,
            "scratchllm_path": str(scratch_dir),
            "use_corpus_graph": True,
            "align_data_dir": str(tmp_path),
            "use_attention_loop": True,
        }):
            kind = wire.get_generator_kind()
        assert kind == "graph_attention"

    def test_use_scratchllm_but_missing_path_uses_graph(self, tmp_path: Path):
        """When use_scratchllm is true but scratchllm_path does not exist, graph wins."""
        (tmp_path / "corpus").mkdir(exist_ok=True)
        (tmp_path / "corpus" / "graph.json").write_text("{}")

        with patch.object(wire, "get_config", return_value={
            "use_scratchllm": True,
            "scratchllm_path": str(tmp_path / "nonexistent"),
            "use_corpus_graph": True,
            "align_data_dir": str(tmp_path),
            "use_attention_loop": True,
        }):
            kind = wire.get_generator_kind()
        assert kind == "graph_attention"
